import argparse
import os
import sys
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor, AdafactorSchedule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions, pos_label=1)
    return auc(fpr, tpr)


class BalanceType(Enum):
    NON = 1
    CUTMAJORITY = 2
    DUPLICATEMINORITY = 3
    COMBINATION = 3


class MultidomaindeDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        train_split: bool,
        balance: BalanceType,
        max_length: Optional[int] = None,
    ):

        self.inputs, self.targets = [], []
        if train_split:
            data = data[len(data) // 10 :]

            if balance == BalanceType.NON:
                data = data
            elif balance == BalanceType.CUTMAJORITY:
                data = (
                    data.groupby(["label"])
                    .apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().min())]])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
            elif balance == BalanceType.DUPLICATEMINORITY:
                data = (
                    data.groupby(["label"])
                    .apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().max())]])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
            elif balance == BalanceType.COMBINATION:
                df_max = data.label.value_counts().max()
                df_min = data.label.value_counts().min()
                data = (
                    data.groupby(["label"])
                    .apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().max())]])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
                data = (
                    data.groupby(["label"])
                    .apply(lambda x: x.iloc[[i % len(x) for i in range((df_max - df_min) // 2)]])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )

        else:
            data = data[: len(data) // 10]

        if max_length is not None:
            data = data[:max_length]

        print(f"{'Train' if train_split else 'Validation'} dataset size:", len(data))
        self.inputs = tokenizer(
            data["text"].to_list(), padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )
        self.targets = data["label"].apply(lambda x: int(x)).to_list()

    def __len__(self):
        return self.inputs["input_ids"].shape[0]

    def __getitem__(self, index):
        input_ids = self.inputs["input_ids"][index].squeeze()
        input_mask = self.inputs["attention_mask"][index].squeeze()
        return {"input_ids": input_ids, "input_mask": input_mask, "target": self.targets[index]}


class Stats:
    def __init__(self):
        self.loss = []
        self.targets = []
        self.predictions = []


class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            learning_rate = 2e-4
            lrs = [
                learning_rate
            ]  # just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs  # [lrs]


class TrainerForSequenceClassification(pl.LightningModule):
    def __init__(self, args):
        super(TrainerForSequenceClassification, self).__init__()
        self.args = args
        os.makedirs(self.args.output_dir, exist_ok=True)

        self.model = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        try:
            self.model.config.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        except:
            print("Warning: Exception occured while setting pad_token_id")

        self._best_validation_loss = sys.float_info.max
        self._training_stats, self._validation_stats = Stats(), Stats()
        self._first_validation_finished = False

    def forward(self, batch):
        return self.model(
            batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["target"],
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._update_stats(self._training_stats, batch, outputs)
        self.lr_scheduler.step()
        if batch_idx % 50 == 0:
            loss, auc_value, accuracy, f1 = self._compute_stats(self._training_stats)
            self._training_stats = Stats()
            self.log_dict(
                {
                    "train_loss": loss,
                    "AUC": auc_value,
                    "ACC": accuracy,
                    "f1": f1,
                    "lr": self.lr_scheduler.get_last_lr()[-1],
                }
            )
        return outputs.loss

    def on_train_epoch_end(self):
        if len(self._training_stats.loss) < 50:
            return
        loss, auc_value, accuracy, f1 = self._compute_stats(self._training_stats)
        self._training_stats = Stats()
        self.log_dict(
            {"train_loss": loss, "AUC": auc_value, "ACC": accuracy, "f1": f1, "lr": self.lr_scheduler.get_last_lr()[-1]}
        )
        if self.args.log_to_console:
            print("\nTraining results: ")
            print(f"    loss: {loss}")
            print(f"     ACC: {accuracy}")
            print(f"     AUC: {auc_value}")

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._update_stats(self._validation_stats, batch, outputs)
        # Input - Output, visual check
        if batch_idx == 0:
            print("\nInput: ", self.tokenizer.batch_decode(batch["input_ids"])[0])
            print("Target: ", batch["target"][0])
            print("Output: ", outputs.logits[0])

    def on_validation_epoch_end(self):
        if self._first_validation_finished == False:
            self._first_validation_finished = True
            self._validation_stats = Stats()
            return

        loss, auc_value, accuracy, f1 = self._compute_stats(self._validation_stats)
        self._validation_stats = Stats()
        self.log_dict(
            {"validation_loss": loss, "validation_AUC": auc_value, "validation_ACC": accuracy, "validation_f1": f1}
        )
        if self.args.log_to_console:
            print("\nValidation results: ")
            print(f"    loss: {loss}")
            print(f"     ACC: {accuracy}")
            print(f"     AUC: {auc_value}")

        if loss < self._best_validation_loss and self.current_epoch % self.args.model_save_period_epochs == 0:
            self._save_model()
            self._best_validation_loss = loss

    def configure_optimizers(self):
        if self.args.using_peft:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.opt = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.my_params.adam_epsilon,
                weight_decay=self.my_params.weight_decay,
            )
        else:
            self.opt = Adafactor(
                self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None
            )
            self.lr_scheduler = MyAdafactorSchedule(self.opt)
        return self.opt

    def train_dataloader(self):
        train_dataset = MultidomaindeDataset(
            data=self.args.data,
            tokenizer=self.tokenizer,
            train_split=True,
            balance=BalanceType.CUTMAJORITY,
        )

        dataloader = DataLoader(
            train_dataset, batch_size=self.args.train_batch_size, drop_last=True, shuffle=True, num_workers=4
        )
        if self.args.using_peft:
            t_total = self.args.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
            )
        return dataloader

    def val_dataloader(self):
        val_dataset = MultidomaindeDataset(
            data=self.args.data,
            tokenizer=self.tokenizer,
            train_split=False,
            balance=BalanceType.NON,
        )
        return DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=4, shuffle=True)

    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        logits = outputs.logits.detach().cpu()
        outputs = torch.nn.Softmax(dim=-1)(logits)[:, 1]
        stats.targets.extend(batch["target"].cpu())
        stats.predictions.extend(outputs)

    def _compute_stats(self, stats: Stats):
        round_predictions = [torch.round(p) for p in stats.predictions]
        loss = torch.stack(stats.loss).mean()
        auc = auc_from_pred(stats.targets, stats.predictions)
        accuracy = accuracy_score(stats.targets, round_predictions)
        f1 = f1_score(stats.targets, round_predictions)
        return loss, auc, accuracy, f1

    def _decode_logits(self, logits):
        argmax = torch.argmax(logits, -1, keepdim=False)
        return self.tokenizer.batch_decode(argmax)

    def _save_model(self):
        print("### Saving model ###")
        # Save best model (only adapters)
        best_model_path = os.path.join(self.args.output_dir, "best")
        os.makedirs(best_model_path, exist_ok=True)
        self.model.save_pretrained(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)


### Example
# python finetuning_reproduce.py --data_path "multidomain.csv" --model microsoft/mdeberta-v3-base --domain social_media --language en es ru --generator gemini
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model_name",
        choices=[
            "microsoft/mdeberta-v3-base",
            "FacebookAI/xlm-roberta-large",
            "tiiuae/falcon-rw-1b",
            "tiiuae/falcon-11B",
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Meta-Llama-3-8B",
            "bigscience/bloomz-3b",
            "google/mt5-small",
            "CohereForAI/aya-101",
        ],
        nargs="?",
        required=True,
    )
    parser.add_argument("--domain", choices=["all", "news", "social_media"], nargs="?")
    parser.add_argument(
        "--language",
        choices=[
            "en",
            "pt",
            "de",
            "nl",
            "es",
            "ru",
            "pl",
            "ar",
            "bg",
            "ca",
            "uk",
            "ro",
            "zh",
            "hr",
            "cs",
            "et",
            "el",
            "hu",
            "ga",
            "gd",
            "sk",
            "sl",
        ],
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--source",
        choices=["telegram", "twitter", "gab", "discord", "whatsapp"],
        default=None,
    )
    parser.add_argument(
        "--generator",
        choices=[
            "gemini",
            "Llama-2-70b-chat-hf",
            "gpt-3.5-turbo-0125",
            "opt-iml-max-30b",
            "aya-101",
            "v5-Eagle-7B-HF",
            "Mistral-7B-Instruct-v0.2",
            "vicuna-13b",
        ],
        default=[
            "gpt-3.5-turbo-0125",
            "opt-iml-max-30b",
            "aya-101",
            "v5-Eagle-7B-HF",
            "Mistral-7B-Instruct-v0.2",
            "vicuna-13b",
        ],
        nargs="+",
    )
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--job_name", type=str, default="default")
    parser.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--saved_model_root", type=str, default="saved_models")
    parser.add_argument("--logging_root", type=str, default="lightning_logs")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    # 1) Create datatset
    df = pd.read_csv(args.data_path, index_col=0)
    df = df[df["split"] == "train"]
    df["source"] = [x.replace("multisocial_", "") for x in df["source"]]
    if args.language:
        df = df[df["language"].isin(args.language)]
    if args.generator:
        df = df[df["multi_label"].isin(args.generator + ["human"])]
    if args.domain and args.domain != "all":
        df = df[df["domain"] == args.domain]
    if args.source:
        df = df[df["source"] == args.source]

    print("Training dataset:")
    print(df)

    # 2) Prepare model
    if args.use_peft:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        target_map = {
            "microsoft/mdeberta-v3-base": ["query_proj", "key_proj", "value_proj"],
            "FacebookAI/xlm-roberta-large": [
                "query",
                "key",
                "value",
            ],
            "tiiuae/falcon-rw-1b": ["query_key_value"],
            "tiiuae/falcon-11B": ["query_key_value"],
            "mistralai/Mistral-7B-v0.1": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            "meta-llama/Meta-Llama-3-8B": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            "bigscience/bloomz-3b": ["query_key_value"],
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(task_type="SEQ_CLS", target_modules=target_map.get(args.model_name, None), r=4),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, ignore_mismatched_sizes=True
        )
    print(model)

    train_args = argparse.Namespace(
        output_dir=os.path.join(args.saved_model_root, args.job_name, args.model_name.split("/")[-1]),
        model=model,
        data_path=args.data_path,
        data=df,
        model_name=args.model_name,
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=100,
        train_batch_size=(1 if "aya" in args.model_name else 2),
        eval_batch_size=(1 if "aya" in args.model_name else 2),
        model_save_period_epochs=2,
        num_train_epochs=7,
        using_peft=args.use_peft,
        log=True,
        log_to_console=True,
    )
    model_trainer = TrainerForSequenceClassification(train_args)

    train_params = dict(
        accumulate_grad_batches=8,
        max_epochs=train_args.num_train_epochs,
        precision="32",
        # val_check_interval=0.2,
        deterministic=True,
        logger=TensorBoardLogger(
            save_dir=os.path.join(args.logging_root, args.job_name),
            name=args.model_name.split("/")[-1] if train_args.log else None,
        ),
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=8)],
    )
    pl.Trainer(**train_params).fit(model_trainer)
