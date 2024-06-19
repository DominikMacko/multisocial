# MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts
Source code for replication of the experiments in the paper submitted to ACL Rolling Review.

## Install Dependencies
Install the [IMGTB framework](https://github.com/kinit-sk/IMGTB) along with its dependencies.

## Data Preparation
Download the human texts from the existing datasets, described in the paper.
- [Gab](https://zenodo.org/records/1418347) - download the dataset and put it into "dataset/gab_posts_jan_2018.json.tar.gz"
- [WhatsApp](https://github.com/gvrkiran/whatsapp-public-groups) - request the non-anonymised version of the dataset and put it into "dataset/non_anonymised_data_to_share.tsv.gz"
- [Telegram](https://zenodo.org/records/3607497) - download the messages part of the dataset and put it into "dataset/messages.ndjson.zst"
- [Twitter](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/tree/main/task1) - download all the ".jsonl" and ".tsv" files and put them into the folder "dataset/clef2022-checkthat-lab-main-task1-data/"
- [Twitter](https://www.kaggle.com/datasets/kazanova/sentiment140/data) - download the dataset and ZIP it into the archive "dataset/sentiment140.zip"
- [Discord](https://www.kaggle.com/datasets/jef1056/discord-data) - download the dataset and all three versions put into the folder "dataset/discord-data/" (so there will be e.g. "dataset/discord-data/v1/content" subfolder structure)

## Source Code Structure
| # | Description |
| :-: | :-: |
| 01 | A script for combination of the downloaded datasets and selection of the subset. |
| 02a | A script to generate texts using paraphrasing by various large language models (HuggingFace pretrained models as well as OpenAI API accessible models). |
| 02b | A script for automated text-similarity metrics calculation. |
| 02c | A script for combination of the generated texts from different models. |
| 02d | A script for pre-processing (filtering) for the experiments. |
| 03a | A configuration file to run external [IMGTB framework](https://github.com/kinit-sk/IMGTB). |
| 03b | A text file to update environment requirements for fine-tuning process. |
| 03c | A script for fine-tuning of detection models.|
| 04 | Google Colab notebook for results analysis. |

## Experiments
1. Run the 01 script three times (1. to combine all dataset, 2. to sample 10k texts per FastText detected language and platform, 3. to create the final deduplicated subset with a more reliable language detection) to create "dataset/multisocial.csv.gz" containing human-written texts sampled from the downloaded datasets.

2. Use the 02 scripts in their alphabetic order to create machine-generated texts and combine them into "multisocial_clean.csv.gz" dataset.

3. Use the 03a configuration file to run external [IMGTB framework](https://github.com/kinit-sk/IMGTB) for predictions by the statistical and pre-trained detectors.
   ```
   python benchmark.py --from_config=./03a_IMGTB_config.yaml
   ```
5. Use the 03c script to fine-tune individual versions of the fine-tuned detectors described in the paper. Exemplar usage:
   ```
   python 03c_finetune.py --data_path "dataset/multisocial_clean.csv.gz" --model microsoft/mdeberta-v3-base --domain social_media --language en es --source telegram
   ```
6. Modify the 03a config file to run external [IMGTB framework](https://github.com/kinit-sk/IMGTB) for predictions by the fine-tuned models.

7. Run Colab notebook of 04 to calculate and analyze the results.
