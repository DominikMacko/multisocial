openai_organization = "" #use your openai organization id
openai_api_key = "" #use your openai api key
hf_token = "" #use your huggingface access token
CACHE = "./cache/" #change to your huggingface cache folder (where the models will be downloaded)
offload_folder = "./offload_folder/" #change to your offload folder (empty temporary dir for offloading huge models)
selected = ['ar', 'ca', 'cs', 'de', 'en', 'es', 'nl', 'pt', 'ru', 'uk', 'zh'] + ['ro', 'gd', 'et', 'el', 'sk', 'bg', 'pl', 'ga'] + ['hu', 'sl', 'hr']
testing=False

import sys

MODEL = sys.argv[1]
DATASET = sys.argv[2]

import os
os.environ['HF_HOME'] = CACHE

import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import re
import gc
import time
import shutil
import nvidia_smi, psutil
from langcodes import *
from tqdm import tqdm
import backoff
import openai
openai.organization = openai_organization
openai.api_key = openai_api_key

import random
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda:0

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.ServiceUnavailableError, openai.error.Timeout))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

## data loading

model_name = MODEL.split("/")[-1]
if os.path.isfile(DATASET.replace('.csv', f'_{model_name}.csv')):
  shutil.copy(DATASET.replace('.csv', f'_{model_name}.csv'), DATASET.replace('.csv', f'_{model_name}.csv') + '_' + str(int(time.time())))
  try:
    df = pd.read_csv(DATASET, lineterminator='\n', escapechar='\\')
  except:
    df = pd.read_csv(DATASET, lineterminator='\n')
  try:
    df2 = pd.read_csv(DATASET.replace('.csv', f'_{model_name}.csv'), lineterminator='\n', escapechar='\\')
  except:
    df2 = pd.read_csv(DATASET.replace('.csv', f'_{model_name}.csv'), lineterminator='\n')
  if "generated" not in df.columns:
    df['generated'] = ""
  df['text'] = df['text'].astype(str)
  df['generated'] = df['generated'].astype(str)
  df2['text'] = df2['text'].astype(str)
  df2['generated'] = df2['generated'].astype(str)
  df.fillna("", inplace=True)
  df2.fillna("", inplace=True)
  #df.loc[(df.text == df2.text), 'generated'] = df2['generated']
  df2.loc[df2.duplicated(), 'text'] = ""
  df2.drop_duplicates(subset=['text'], inplace=True)
  df = pd.merge(df, df2[['text','generated']], on=['text'], how='left', suffixes=('','_y'))
  if 'generated_y' in df.columns:
    df.loc[((df.generated == "") | (df.generated == "nan")) & ((df.generated_y != "") & (df.generated_y != "nan")), 'generated'] = df['generated_y']
    df.drop(columns=['generated_y'], inplace=True)
else:
  df = pd.read_csv(DATASET, lineterminator='\n', escapechar='\\')
  df['text'] = df['text'].astype(str)
  if "generated" in df.columns:
    df['generated'] = df['generated'].astype(str)
df.fillna("", inplace=True)
df = df[df.label != 'label'].reset_index(drop=True)

df['text'] = ["nan" if x=="" else x for x in df['text']]
if "generated" in df.columns:
    df['generated'] = ["nan" if x=="" else x for x in df['generated']]

if testing:
    #subsampling for testing purpose
    subset = df.groupby(['source', 'language']).apply(lambda x: x.sample(min(20, len(x)), random_state = 0)).reset_index(drop=True)
    subset['selected'] = [x in selected for x in subset.language]
    subset = subset[subset.selected]
    subset = subset.drop(columns=['selected'])

    #just single sample for testing
    #subset = subset[10:11]
    #subset = subset[subset.language == 'sk'].reset_index(drop=True)[:2]
else:
    subset = df
    subset['selected'] = [x in selected for x in subset.language]
    subset = subset[subset.selected]
    subset = subset.drop(columns=['selected'])

## model loading

use4bit = True
use8bit = False
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=use4bit, load_in_8bit=use8bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

text2text = False
if 'aya-101' in model_name: text2text = True
  
model = None
tokenizer = None
task = "text-generation"
if ("gpt-3.5-turbo" in model_name):
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", trust_remote_code=True, cache_dir=CACHE)
  pass
elif "rwkv" in MODEL.lower():
  tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE)
  model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=CACHE)
elif text2text:
  tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE)
  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE)
  task = "text2text-generation"
else:
  tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE, token=hf_token)
  model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE, token=hf_token)

if model is not None:
  model = model.eval()

if testing: model_name = model_name + "_testing4"

## generating texts

def generate_rwkv_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

generated = [""] * len(subset)
with torch.no_grad():
  for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
    if ("generated" in row.index) and (row['generated'] is not np.NaN) and (str(row['generated']) != "nan"):
      generated[index] = row['generated']
      #print(index, 'skipping')
      continue
    #for testing purpose
    else:
      #print(index, 'processing')
      #continue
      pass
    if not isinstance(row.text, str): continue
    if row.text == "nan": continue
    n_try = 0
    result = ""
    while n_try < 3:
      language = row.language
      language_name = Language.make(language=row.language).display_name()
      text = row.text.replace('\n\n','\n')
      prompt = f'You are a helpful assistent.\n\nTask: Generate the text in {language_name} similar to the input social media text but using different words and sentence composition.\n\nInput: {text}\n\nOutput:'
      if "rwkv" in MODEL.lower(): prompt = generate_rwkv_prompt(f'You are a helpful assistent. Generate the text in {language_name} similar to the input social media text but using different words and sentence composition.', input=text)
      if model is None:
        result = chat_completions_with_backoff(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200, top_p=0.95, seed=42).choices[0].message.content
        time.sleep(2) #to not reach openai rate limit of requests per minute, or use backoff
      else:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
        #with torch.cuda.amp.autocast():
        if "gemma" in MODEL:
            generated_ids = model.generate(input_ids=input_ids.to(device), min_new_tokens=5, max_new_tokens=200, num_return_sequences=1, do_sample=False)
            result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            result = result[0]
        else:
          try:
            generated_ids = model.generate(input_ids=input_ids.to(device), min_new_tokens=5, max_new_tokens=200, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95)
            result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            result = result[0]
          except Exception as e:
            print('Error', str(e), device, index, text)
            n_try += 1
            continue
      #print(index, len(subset), len(generated))
      result = result.replace(prompt, '').split('\n\n')[0].split('\nor\n')[0].split('\nor \n')[0].strip() #clearing redundant parts of the output
      result = re.split(r'\S+Task:', result)[0].strip() #problem with artificial instructions in Eagle
      result = re.split(r'\S+ranslation:', result)[0].strip() #problem with translations in Mistral
      result = tokenizer.decode(tokenizer(result)['input_ids'][:len(tokenizer(text)['input_ids'])+10], skip_special_tokens=True) #max 10 tokens more than input text
      if (result != "") and (result != text):
        generated[index] = result
        break
      else:
        n_try += 1
        if n_try >= 3: break
    if (index % 100) == 0:
      subset['generated'] = generated
      subset.to_csv(DATASET.replace('.csv', f'_{model_name}.csv'), index=False, escapechar='\\')

subset['generated'] = generated
subset.to_csv(DATASET.replace('.csv', f'_{model_name}.csv'), index=False, escapechar='\\')

#modify to make it ready as the input to the next iteration of paraphrasing
subset['text'] = subset['generated']
subset['generated'] = ""
subset.to_csv(DATASET.replace('.csv', f'_{model_name}_paraphrased.csv'), index=False, escapechar='\\')
