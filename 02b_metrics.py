datapath = './dataset' #path to the dataset folder
precheck = False #sample generation
fasttext_model_path = "./lid.176.bin" #path to the downloaded fasttext model

import os
import sys

models = [sys.argv[1]]#['vicuna-13b', 'aya-101', 'v5-Eagle-7B-HF', 'Mistral-7B-Instruct-v0.2', 'opt-iml-max-30b', 'gpt-3.5-turbo-0125']#, 'Llama-2-70b-chat-hf']

import glob
import pandas as pd
import numpy as np
import torch
#from ftlangdetect import detect
import fasttext
from tqdm import tqdm
from collections import Counter
from langcodes import *
from polyglot.text import Text, Word
import regex
import tensorflow_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import ngram
import editdistance
import mauve
from mauve.compute_mauve import compute_mauve
from nltk.translate import meteor
from nltk import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from evaluate import load
bertscore = load("bertscore")

import tensorflow_hub as hub
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
random.seed(42)
np.random.seed(42)

pd.set_option('display.max_rows', 100)
tqdm.pandas()

device = 0 if torch.cuda.is_available() else "CPU"

#remove whitespaces around texts
def clear_dataset(df):
  df_string_columns = df.select_dtypes(['object'])
  df[df_string_columns.columns] = df_string_columns.apply(lambda x: x.str.strip())
  return df

#remove some unicode chars making problems in polyglot
#https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790
def remove_bad_chars(text):
  RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
  return RE_BAD_CHARS.sub("", text)

#remove prompts from generated text
def remove_prompts(row):
  language = row.language
  language_name = Language.make(language=row.language).display_name()
  headline = row.title
  prompt = f'You are a multilingual journalist.\n\nTask: Write a news article in {language_name} for the following headline: "{headline}". Leave out the instructions, return just the text of the article.\n\nOutput:'
  #alpaca prompt
  prompt2 = f'<unk>### Instruction:\nYou are a multilingual journalist.\n\nTask: Write a news article in {language_name} for the following headline: "{headline}". Leave out the instructions, return just the text of the article.\n\n\n\n### Response:'
  text = str(row.generated).strip()
  text = text.replace(prompt2, '').strip()
  text = text.replace(''.join(prompt2.split()), '').strip()
  #text = text.replace(remove_bad_chars(prompt2), '').strip()
  text = text.replace(prompt, '').strip()
  text = text.replace(''.join(prompt.split()), '').strip()
  #text = text.replace(remove_bad_chars(prompt), '').strip()
  text = text.replace(f'"{row.title}"', '').strip()
  text = text.replace(row.title, '').strip()
  #text = text.replace(remove_bad_chars(row.title), '').strip()
  return text
  text = text.replace('###', '').strip()
  text = text.replace('Instruction:', '').strip()
  text = text.replace('You are a multilingual journalist.', '').strip()
  text = text.replace('Task:', '').strip()
  text = text.replace(f'Write a news article in {language_name} for the following headline:', '').strip()
  text = text.replace('\"\".', '').strip()
  text = text.replace('Leave out the instructions, return just the text of the article.', '').strip()
  text = text.replace('Response:', '').strip()
  return text

#remove unfinished final sentence from generated text
def remove_unended_sentence(row):
  text = Text(row.generated, hint_language_code=row.language)
  if (row.generated != '') and (len(text.sentences) > 1):
    if (text.sentences[-1].words[-1] not in ['?', '?', '!', '?', '.']): #final sentence not ended by any of these characters
      return row.generated.removesuffix(str(text.sentences[-1]))
  return row.generated

#detect language of generated text
fasttext_model = fasttext.load_model(fasttext_model_path)
def detect(text: str, low_memory=False):
    if (text.strip() == "") or (text.strip() == "nan"): return {"lang": "unknown", "score": 0.0,}
    labels, scores = fasttext_model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
def fasttext_detect_language(dataset):
  generated_languages = []
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if (str(row.generated) != "nan") and (row.generated != ""):
      #lines = row.generated.split('\n')
      #detected = []
      #for line in lines:
      #  detected.append(detect(text=line, low_memory=False)['lang'])
      ##HOW TO REPRESENT PER-LINE DETECTION? ALL/MAJORITY MUST MATCH?
      #c = Counter(detected)
      #detected_language = c.most_common()[0][0] #majority
      #generated_languages.append(detected_language)
      generated_languages.append(detect(text=row.generated.replace('\n', ' '), low_memory=False)['lang'])
    else:
      generated_languages.append(row.language)
  return generated_languages

#shorten generated texts
def shorten_generated(row):
  generated = str(row.generated).strip()
  if (generated == '') or (generated == 'nan'):
    return generated
  generated_length = len(row.generated.split())
  if (row.language == 'zh'):
    generated_length = len(Text(row.generated, hint_language_code=row.language).words)
  human_length = len(row.text.split())
  if (row.language == 'zh'):
    human_length = len(Text(row.text, hint_language_code=row.language).words)

  if (human_length == 0):
    return generated

  while (human_length < (generated_length - 5)): #remove last sentence while more than 5 words longer
    #print(human_length, '<', generated_length)
    text = Text(generated, hint_language_code=row.language)
    if (len(text.sentences) < 2): #single sentence will not be removed
      return generated
    generated = generated.removesuffix(str(text.sentences[-1])).strip()
    generated_length = len(generated.split())
    if (row.language == 'zh'):
      generated_length = len(Text(generated, hint_language_code=row.language).words)
  return generated

#unify dataset form
def unify_form(dataset, model):
  dataset = clear_dataset(dataset)
  dataset['label'] = model
  dataset['text'] = dataset['generated']
  #ToDo: list() for Chinese to obtain letters or use some NLP library to get words
  #dataset['length'] = [len(x.split()) for x in dataset.text]
  dataset['length'] = [len(x.split()) if (y != 'zh') or (x == '') or (x == 'nan') else len(Text(x, hint_language_code=y).words) for (x, y) in zip(dataset.text, dataset.language)]
  dataset['source'] = [f'multisocial_{x}' for x in dataset.source]
  #dataset.drop(columns=['url', 'title', 'generated', 'generated_languages_fasttext'], inplace=True)
  #dataset = dataset[column_order]
  return dataset

#uniqueness/repetitiveness - get number of unique sentences in row.text
def unique_sentences(row):
  if (row.text == '') or (row.text == 'nan'):
    return 0
  sentences = Text(row.text, hint_language_code=row.language).sentences
  return len(set(sentences)) / len(sentences)

#uniqueness/repetitiveness - get number of unique words in row.text
def unique_words(row):
  if (row.text == '') or (row.text == 'nan'):
    return 0
  words = Text(row.text, hint_language_code=row.language).words
  return len(set(words)) / len(words)

#evaluate similarity of generated text to original human text
def get_ngram(dataset, n=3):
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      metric[index] = round(ngram.NGram.compare(original, obfuscated, N=n), 4)
    except:
      metric[index] = 0.0
  return metric

def custom_tokenizer(text, language=None):
  #return word_tokenize(text)
  return list(Text(text, hint_language_code=language).words)

#evaluate similarity of generated text to original human text
#nltk tokenizer can be changed to polyglot for better language support
def get_tf_cosine_similarity(dataset):
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      original_tokens = custom_tokenizer(original, row.language)
      obfuscated_tokens = custom_tokenizer(obfuscated, row.language)
      words = set(original_tokens).union(set(obfuscated_tokens))
      vectorizer = CountVectorizer(tokenizer = custom_tokenizer, vocabulary = words)
      original_vector = vectorizer.transform([original])
      obfuscated_vector = vectorizer.transform([obfuscated])
      metric[index] = round(cosine_similarity(original_vector.toarray(), obfuscated_vector.toarray())[0][0], 4)
    except:
      metric[index] = 0.0
  return metric

#evaluate similarity of generated text to original human text
def get_meteor(dataset):
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      #metric[index] = round(meteor([word_tokenize(original)],word_tokenize(obfuscated)), 4)
      metric[index] = round(meteor([custom_tokenizer(original, row.language)],custom_tokenizer(obfuscated, row.language)), 4)
    except:
      metric[index] = 0.0
  return metric

#evaluate similarity of generated text to original human text
def get_bertscore(dataset):
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      results = bertscore.compute(predictions=[obfuscated], references=[original], model_type="bert-base-multilingual-cased")
      metric[index] = sum(results['f1']) / len(results['f1'])
    except:
      metric[index] = 0.0
  return metric

embed = None
#evaluate similarity of generated text to original human text
def get_use_cosine_similarity(dataset):
  global embed
  if embed is None:
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual-large/versions/2")
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      results = cosine_similarity(embed(original), embed(obfuscated), dense_output=False)
      metric[index] = results.mean()
    except:
      metric[index] = 0.0
  return metric

#evaluate similarity of generated text to original human text
def get_editdistance(dataset):
  metric = [""] * len(dataset)
  for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    if ("metric" in row.index) and (row['metric'] is not np.NaN) and (str(row['metric']) != "nan"):
      metric[index] = row['metric']
      continue
    original = row.text
    obfuscated = row.generated
    try:
      metric[index] = editdistance.eval(original, obfuscated)
    except:
      metric[index] = 0.0
  return metric

#evaluate similarity of generated text to original human text
def get_diff_charlen(dataset):
  df = dataset
  prev = ''
  text_charlength = [len(''.join([y if (y != prev) & ((prev:=y) == y) else '' for y in x]).strip()) for x in df['text']]
  generated_charlength = [len(''.join([y if (y != prev) & ((prev:=y) == y) else '' for y in x]).strip()) for x in df['generated']]
  #result = np.divide(text_charlength, generated_charlength)
  generated_charlength_inv = np.array([1/i if i!=0 else 0 for i in generated_charlength])
  result = text_charlength * generated_charlength_inv
  return result


datasets = {}
for model in models:
  #filename = f'{datapath}/multisocial_{model}_paraphrased_{model}_paraphrased_{model}.csv.gz'
  filename = f'{datapath}/multisocial_3x_{model}.csv.gz' #data with original (text) and 3x paraphrased text (generated)
  print(filename)
  temp = pd.read_csv(filename, lineterminator='\n')
  #filename = filename.replace('_3x_', '_')
  datasets[filename.split('multisocial_')[-1].split('.csv')[0]] = temp

multisocial = pd.DataFrame()
for model, dataset in datasets.items():
  print(f'Processing {model}')

  dataset['multi_label'] = model
  dataset.fillna("", inplace=True)

  if not precheck:
    #dataset['generated'] = dataset.apply(lambda x: remove_prompts(x), axis=1)
    dataset['generated'] = dataset['generated'].apply(lambda x: remove_bad_chars(x))
    #dataset['generated'] = dataset.apply(lambda x: remove_unended_sentence(x), axis=1)
    #dataset['generated'] = dataset.progress_apply(lambda x: shorten_generated(x), axis=1)
  
  empty_generation = len(dataset[dataset.generated.str.strip() == ''])
  
  dataset['mauve'] = compute_mauve(p_text=dataset['text'], q_text=dataset['generated'], featurize_model_name='google-bert/bert-base-multilingual-cased',  device_id=device, max_text_length=512, verbose=False).mauve
  dataset['meteor'] = get_meteor(dataset)
  dataset['bertscore'] = get_bertscore(dataset)
  dataset['fasttext'] = fasttext_detect_language(dataset)
  dataset['ngram'] = get_ngram(dataset, n=3)
  dataset['tf'] = get_tf_cosine_similarity(dataset)
  dataset['editdistance'] = get_editdistance(dataset)
  #dataset['use'] = get_use_cosine_similarity(dataset)
  dataset['ED-norm'] = dataset['editdistance'] / [len(x) for x in dataset['text']]
  dataset['diff_charlen'] = get_diff_charlen(dataset)
  dataset['diff_charlen'] = np.array([1/i if i!=0 else 0 for i in dataset['diff_charlen']])
  dataset['changed_language'] = dataset.language != dataset.fasttext
  dataset['LangCheck'] = len(dataset[dataset['changed_language']]) / len(dataset)

  dataset.to_feather(datapath + f'/metrics_{model}.feather')
  multisocial = pd.concat([multisocial, dataset], ignore_index=True, copy=False)

multisocial = multisocial.reset_index(drop=True)
if precheck:
  multisocial.to_feather(datapath + f'/metrics_testing3.feather')
else:
  multisocial.to_feather(datapath + f'/metrics.feather')

#multisocial = pd.read_feather(datapath + f'/metrics.feather')
#metrics = ['meteor', 'bertscore', 'use', 'ngram', 'tf', 'ED-norm', 'diff_charlen', 'LangCheck']
#multisocial.groupby(['multi_label'])[metrics].agg(['mean', 'std']).style.format(na_rep=0, precision=4).highlight_max(props='font-weight: bold;', axis=0)
  