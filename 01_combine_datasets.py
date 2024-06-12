datapath = "./dataset/" #path where the datasets are downloaded and where the combined dataset will be
selected = ['ar', 'ca', 'cs', 'de', 'en', 'es', 'nl', 'pt', 'ru', 'uk', 'zh'] + ['ro', 'gd', 'et', 'el', 'sk', 'bg', 'pl', 'ga'] + ['hu', 'sl', 'hr']
column_order = ['text', 'label', 'length', 'source', 'language', 'domain', 'topic', 'split']
fasttext_model_path = "./lid.176.bin" #path to the downloaded fasttext model

from tqdm import tqdm
import re
#from ftlangdetect import detect
import fasttext
import regex
import unicodedata
from polyglot.text import Text, Word
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")
from lingua import LanguageDetectorBuilder, IsoCode639_1
from LanguageIdentifier import rank
import numpy as np
import pandas as pd
import glob
import os
from os import listdir
from os.path import isfile, join
import io
import csv
import tarfile
import json
import ujson
import psutil
import random
random.seed(42)
np.random.seed(42)

#zstd stream reader, adjusted from https://raw.githubusercontent.com/pushshift/zreader/master/zreader.py
import zstandard as zstd

class Zreader:
    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
        self.chunk_size = chunk_size
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''

    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            try:
              chunk = self.reader.read(self.chunk_size).decode()
            except:
              buffer = ''
              continue
            if not chunk: break 
            lines = (self.buffer + chunk).split("\n")
            for line in lines[1:-1]:
                yield line
            self.buffer = lines[-1]

# language check

#remove some unicode chars making problems in polyglot (https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790)
def remove_bad_chars(text):
  text = ''.join([l for l in text if unicodedata.category(str(l))[0] not in ('S', 'M', 'C')])
  RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
  return RE_BAD_CHARS.sub("", text)

fasttext_model = fasttext.load_model(fasttext_model_path)
def detect(text: str, low_memory=False):
    labels, scores = fasttext_model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
def get_fasttext_lang(text):
    text = remove_bad_chars(text).lower().replace('im', '') #remove im due to language detection confusing of I'm without apostrophe often in messages
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove url for language detection
    fasttext_res = detect(text=text.replace('\n', ' '), low_memory=False)
    return fasttext_res['lang'], fasttext_res['score']

def get_LanguageIdentifier_lang(text):
    text = remove_bad_chars(text).lower().replace('im', '') #remove im due to language detection confusing of I'm without apostrophe often in messages
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove url for language detection
    LI = rank(text.replace('\n', ' '))
    LI_langs, LI_scores = zip(*LI)
    LI_langs = np.array(LI_langs)
    LI_scores = np.array(LI_scores)
    return LI_langs[np.argmax(LI_scores)], LI_scores[np.argmax(LI_scores)], list(LI_langs[LI_scores > 0.01])
    
linguadetector = LanguageDetectorBuilder.from_all_languages().build()
def get_lang(text):
    text = remove_bad_chars(text).lower().replace('im', '') #remove im due to language detection confusing of I'm without apostrophe often in messages
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove url for language detection
    fasttext_res = detect(text=text.replace('\n', ' '), low_memory=False)
    if fasttext_res['score'] > 0.8: return fasttext_res['lang'], fasttext_res['score']

    polyglot_res = Detector(text, quiet=True)
    polglot_langs = []
    for l in polyglot_res.languages:
      polglot_langs.append(l.code)
    if fasttext_res['lang'] in polglot_langs: return fasttext_res['lang'], fasttext_res['score']

    lingua_res = linguadetector.compute_language_confidence_values(text)[0]
    if lingua_res.language.iso_code_639_1.name.lower() == fasttext_res['lang']: return fasttext_res['lang'], fasttext_res['score']
    if lingua_res.language.iso_code_639_1.name.lower() == polyglot_res.language.code: return polyglot_res.language.code, polyglot_res.language.confidence/100.0

    if (polyglot_res.language.confidence/100.0) > 0.95: return polyglot_res.language.code, polyglot_res.language.confidence/100.0
    if lingua_res.value > 0.8: return lingua_res.language.iso_code_639_1.name.lower(), lingua_res.value

    return 'unknown', 0.0

#get four language detection results with probability scores (+ most likely languages where available)
def get_langs(text):
    text = remove_bad_chars(text).lower().replace('im', '') #remove im due to language detection confusing of I'm without apostrophe often in messages
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove urls
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)"," ",text, flags=re.MULTILINE).split()) #remove hashtags and user referencences
    if ' '.join(text.split()).strip() == "": return 'unknown', 1.0, ['unknown'], 'unknown', 1.0, 'unknown', 1.0, 'unknown', 1.0, ['unknown']
    fasttext_res = detect(text=text.replace('\n', ' '), low_memory=False)
    polyglot_res = Detector(text, quiet=True)
    polyglot_langs = []
    for l in polyglot_res.languages:
      polyglot_langs.append(l.code)
    lingua_res = linguadetector.compute_language_confidence_values(text)[0]
    LI = rank(text.replace('\n', ' '))
    LI_langs, LI_scores = zip(*LI)
    LI_langs = np.array(LI_langs)
    LI_scores = np.array(LI_scores)
      
    return fasttext_res['lang'], fasttext_res['score'], polyglot_langs, polyglot_res.language.code, polyglot_res.language.confidence/100.0, lingua_res.language.iso_code_639_1.name.lower(), lingua_res.value, LI_langs[np.argmax(LI_scores)], LI_scores[np.argmax(LI_scores)], list(LI_langs[LI_scores > 0.01])

#json item finder
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)

#process subset data

if(os.path.isfile(datapath + 'combined_subset.csv.gz')):

  df = pd.read_csv(datapath + 'combined_subset.csv.gz', lineterminator='\n', escapechar='\\')
  
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')
  print(df.info())

  ## language selection
  df['language_backup'] = df['language']
  df['language'] = 'unknown'
  df['fasttext'] = 'unknown'
  df.loc[df.fasttext_score >= 0.2, 'fasttext'] = df['fasttext_lang']
  df['polyglot'] = 'unknown'
  df.loc[df.polyglot_score >= 0.9, 'polyglot'] = df['polyglot_lang']
  df['lingua'] = 'unknown'
  df.loc[df.lingua_score >= 0.2, 'lingua'] = df['lingua_lang']
  df['li'] = 'unknown'
  df.loc[df.li_score >= 0.8, 'li'] = df['li_lang']
  
  df.loc[(df.language_backup == df.fasttext) | (df.language_backup == df.lingua) | (df.language_backup == df.polyglot) | (df.language_backup == df.li), 'language'] = df.language_backup #intended language is detected by some detector
  
  df.loc[(df.fasttext == df.lingua) & (df.fasttext == df.polyglot), 'language'] = df.fasttext #three detectors match (fasttext, polyglot, lingua)
  df.loc[(df.fasttext == df.li) & (df.fasttext == df.polyglot), 'language'] = df.fasttext #three detectors match (fasttext, polyglot, li)
  df.loc[(df.fasttext == df.lingua) & (df.fasttext == df.li), 'language'] = df.fasttext #three detectors match (fasttext, lingua, li)
  df.loc[(df.polyglot == df.li) & (df.lingua == df.polyglot), 'language'] = df.polyglot #three detectors match (polyglot, lingua, li)

  df.loc[(df.language == 'unknown') & (df.fasttext_score >= 0.9) & ((df.fasttext == df.polyglot) | (df.lingua == df.fasttext) | (df.li == df.fasttext)), 'language'] = df.fasttext_lang #high confidence fasttext detected by another detector
  df.loc[(df.language == 'unknown') & (df.polyglot_score >= 0.98) & ((df.fasttext == df.polyglot) | (df.lingua == df.polyglot) | (df.li == df.polyglot)), 'language'] = df.polyglot_lang #high confidence polyglot detected by another detector
  df.loc[(df.language == 'unknown') & (df.lingua_score >= 0.95) & ((df.fasttext == df.lingua) | (df.lingua == df.polyglot) | (df.li == df.lingua)), 'language'] = df.lingua_lang #high confidence lingua detected by another detector
  df.loc[(df.language == 'unknown') & (df.li_score >= 0.9) & ((df.fasttext == df.li) | (df.li == df.polyglot) | (df.li == df.lingua)), 'language'] = df.li_lang #high confidence li detected by another detector
  
  df.loc[(df.language == 'unknown') & ((df.source == 'discord') | (df.source == 'whatsapp')) & (df.li_lang == df.fasttext_lang) & (df.polyglot_lang != 'en') & (df.lingua_lang != 'en'), 'language'] = df.fasttext_lang #lesser confidence li and fasttext match for non english by polyglot or lingua
  
  ## unified form, additional fields
  df['label'] = 'human'
  df['length'] = [len(x.split()) if (y != 'zh') or (x == '') else len(Text(x, hint_language_code=y).words) for (x, y) in zip(df.text, df.language)]
  df['domain'] = 'social_media'
  df['topic'] = 'unknown'
  df = df[df.length > 2].reset_index(drop=True) #at least 3 words, above 5 used for news
  
  ## sampling per platform/source and per language
  social = df[df.language.isin(selected)].groupby(['source', 'language']).apply(lambda x: x.sample(min(1300, len(x)), random_state = 0)).sample(frac=1., random_state = 0).reset_index(drop=True)
  social['split'] = "train"
  test_split = social.groupby(['source', 'language']).apply(lambda x: x.sample(min(300, len(x)), random_state = 0))
  social.loc[social.index.isin(test_split.index.get_level_values(2)), 'split'] = "test"
  
  social = social[column_order]
  social.drop_duplicates(keep=False, inplace=True)
  social.drop_duplicates(subset=['text'], keep=False, inplace=True)
  
  
  ## print some basic statistics
  print('Train:\n', social[social.split == "train"].language.value_counts())
  print('Train:\n', social[social.split == "test"].language.value_counts())
  print(social.language.value_counts())
  print(social.source.value_counts())
  print(social.info())
  
  social.to_csv(datapath + 'multisocial.csv.gz', index=False, escapechar='\\')
  
  exit(0)

# process combined data if available and make small subset (per platform per language)

if(os.path.isfile(datapath + 'combined.csv.gz')):

  df = pd.read_csv(datapath + 'combined.csv.gz', lineterminator='\n', escapechar='\\')
  
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')
  print(df.info())

  df[['source', 'fasttext_lang']].value_counts(sort=False).to_csv(datapath + 'languages_counts.csv')
  
  df = df.groupby(['source', 'fasttext_lang']).apply(lambda x: x.sample(min(10000, len(x)), random_state = 42)).sample(frac=1., random_state = 42).reset_index(drop=True)
  
  #langcheck by three detectors
  results = [get_langs(text) for text in tqdm(df['text'])]
  fasttext_lang, fasttext_score, polyglot_langs, polyglot_lang, polyglot_score, lingua_lang, lingua_score, li_lang, li_score, li_langs = zip(*results)
  df['fasttext_lang'] = fasttext_lang
  df['fasttext_score'] = fasttext_score
  df['polyglot_langs'] = polyglot_langs
  df['polyglot_lang'] = polyglot_lang
  df['polyglot_score'] = polyglot_score
  df['lingua_lang'] = lingua_lang
  df['lingua_score'] = lingua_score
  df['li_langs'] = li_langs
  df['li_lang'] = li_lang
  df['li_score'] = li_score
  
  df.to_csv(datapath + 'combined_subset.csv.gz', index=False, escapechar='\\')
  
  exit(0)

# parse original datasets and combine them (requires 150GB of RAM)

#whatsapp
if(not os.path.isfile(datapath + 'whatsapp.csv.gz')):
  df = pd.read_csv(datapath + 'non_anonymised_data_to_share.tsv.gz', sep='\t', lineterminator='\n', on_bad_lines='warn', quoting=csv.QUOTE_NONE)
  clean_data = [x.strip() for x in set(df.message_text) if isinstance(x, str) and len(x.split()) > 2]
  df = pd.DataFrame({'text': clean_data})
  df.drop_duplicates(subset=['text'], inplace=True)
  df = df[~df.text.str.contains(b'\xe2\xac\x86\xef\xb8\x8f'.decode('utf-8')) & ~df.text.str.contains(b'\xe2\x9a\xbd'.decode('utf-8')) & ~df.text.str.contains(b'\xf0\x9f\x94\xb6'.decode('utf-8'))].reset_index(drop=True) #remove undesired content (due to high amount of unusable text)
  df.to_csv(datapath + 'whatsapp.csv.gz', index=False, escapechar='\\')
whatsapp = pd.read_csv(datapath + 'whatsapp.csv.gz', lineterminator='\n', escapechar='\\')
print('whatsapp:', len(whatsapp), '\n', whatsapp[:10])

#gab
if(not os.path.isfile(datapath + 'gab.csv.gz')):
  languages = []
  texts = []
  with tarfile.open(datapath + 'gab_posts_jan_2018.json.tar.gz', mode='r') as tarf:
   for member in tarf:
    i = 0
    for line in tqdm(tarf.extractfile(member)):
        temp = json.loads(line)
        for item in item_generator(temp, 'post'):
          text = item['body'].strip()
          if (text != "") and (len(text.split()) > 2):
            languages.append(item['language'])
            texts.append(text)
        i += 1
        #if i > 10: break #for testing
    break
  df = pd.DataFrame({'text': texts, 'language': languages})
  df.drop_duplicates(subset=['text'], inplace=True)
  df.to_csv(datapath + 'gab.csv.gz', index=False, escapechar='\\')
gab = pd.read_csv(datapath + 'gab.csv.gz', lineterminator='\n', escapechar='\\')
print('gab:', len(gab), '\n', gab[:10])

#telegram
if(not os.path.isfile(datapath + 'telegram4part.csv.gz')):
  reader = Zreader(datapath + "messages.ndjson.zst", chunk_size=16384)
  texts = set()
  i = 0
  for line in tqdm(reader.readlines()):
    obj = json.loads(line)
    for item in item_generator(obj, 'message'):
      text = item.strip()
      if (text != "") and (len(text.split()) > 2):
        texts.add(text)
    i += 1
    if i == 100000000:
      df = pd.DataFrame({'text': list(texts)})
      texts = set()
      df.drop_duplicates(subset=['text'], inplace=True)
      df.to_csv(datapath + 'telegram1part.csv.gz', index=False, escapechar='\\')
    elif i == 200000000:
      df = pd.DataFrame({'text': list(texts)})
      texts = set()
      df.drop_duplicates(subset=['text'], inplace=True)
      df.to_csv(datapath + 'telegram2part.csv.gz', index=False, escapechar='\\')
    elif i == 300000000:
      df = pd.DataFrame({'text': list(texts)})
      texts = set()
      df.drop_duplicates(subset=['text'], inplace=True)
      df.to_csv(datapath + 'telegram3part.csv.gz', index=False, escapechar='\\')
  df = pd.DataFrame({'text': list(texts)})
  df.drop_duplicates(subset=['text'], inplace=True)
  df.to_csv(datapath + 'telegram4part.csv.gz', index=False, escapechar='\\')
telegram1 = pd.read_csv(datapath + 'telegram1part.csv.gz', lineterminator='\n', escapechar='\\')
telegram2 = pd.read_csv(datapath + 'telegram2part.csv.gz', lineterminator='\n', escapechar='\\')
#telegram3 = pd.read_csv(datapath + 'telegram3part.csv.gz', lineterminator='\n', escapechar='\\')
telegram3 = pd.DataFrame()
telegram4 = pd.read_csv(datapath + 'telegram4part.csv.gz', lineterminator='\n', escapechar='\\')
telegram = pd.concat([telegram1, telegram2, telegram3, telegram4], copy=False, ignore_index=True)
print('telegram:', len(telegram), '\n', telegram[:10])

#twitter
if(not os.path.isfile(datapath + 'twitter.csv.gz')):
  twitter_ar = pd.read_json(datapath + 'clef2022-checkthat-lab-main-task1-data/CT22_arabic_json_objects.jsonl', lines=True)
  twitter_ar['language'] = 'ar'
  twitter_ar['text'] = twitter_ar['full_text'].str.strip()
  twitter_bg = pd.read_json(datapath + 'clef2022-checkthat-lab-main-task1-data/covid19_infodemic_bulgarian_data_multiclass_all.jsonl', lines=True)
  twitter_bg['language'] = 'bg'
  twitter_bg['text'] = twitter_bg['full_text'].str.strip()
  twitter_nl = pd.read_json(datapath + 'clef2022-checkthat-lab-main-task1-data/covid19_infodemic_dutch_data_multiclass_final_all.jsonl', lines=True)
  twitter_nl['language'] = 'nl'
  twitter_nl['text'] = twitter_nl['text'].str.strip()
  twitter_en = pd.read_json(datapath + 'clef2022-checkthat-lab-main-task1-data/covid19_infodemic_english_data_multiclass_final_all.jsonl', lines=True)
  twitter_en['language'] = 'en'
  twitter_en['text'] = twitter_en['full_text'].str.strip()
  twitter_es = pd.read_json(datapath + 'clef2022-checkthat-lab-main-task1-data/CT22_spanish_json_objects.jsonl', lines=True)
  twitter_es['language'] = 'es'
  twitter_es['text'] = twitter_es['text'].str.strip()
  twitter_tr = pd.read_csv(datapath + 'clef2022-checkthat-lab-main-task1-data/CT22_turkish_1A_checkworthy_train.tsv', sep='\t')
  twitter_tr['language'] = 'tr'
  twitter_tr['text'] = twitter_tr['tweet_text'].str.strip()
  df = pd.concat([twitter_ar, twitter_bg, twitter_nl, twitter_en, twitter_es, twitter_tr], copy=False, ignore_index=True)[['text', 'language']]
  df.drop_duplicates(subset=['text'], inplace=True)
  df.dropna(subset=['text'], inplace=True, ignore_index=True)
  lengths = [len(x.split()) for x in df['text']]
  df['length'] = lengths
  df = df[(df.text != "") & (df['length'] > 2)]
  df.drop(columns=['length'], inplace=True)
  df = df.reset_index(drop=True)
  df.to_csv(datapath + 'twitter.csv.gz', index=False, escapechar='\\')
twitter = pd.read_csv(datapath + 'twitter.csv.gz', lineterminator='\n', escapechar='\\')
print('twitter:', len(twitter), '\n', twitter[:10])

if(not os.path.isfile(datapath + 'twitter2.csv.gz')):
  df = pd.read_csv(datapath + 'sentiment140.zip', header=None, lineterminator='\n', encoding_errors='ignore')
  df['text'] = df[5].str.strip()
  df.drop_duplicates(subset=['text'], inplace=True)
  lengths = [len(x.split()) for x in df['text']]
  df['length'] = lengths
  df = df[(df.text != "") & (df['length'] > 2)]
  df.drop(columns=['length'], inplace=True)
  df.dropna(inplace=True)
  df = df.reset_index(drop=True)
  df[['text']].to_csv(datapath + 'twitter2.csv.gz', index=False, escapechar='\\')
twitter2 = pd.read_csv(datapath + 'twitter2.csv.gz', lineterminator='\n', escapechar='\\')
print('twitter2:', len(twitter2), '\n', twitter2[:10])

#discord
if(not os.path.isfile(datapath + 'discord.csv.gz')):
  folder_path = datapath + 'discord-data/v1/content/drive/Shareddrives/Datasets/cleaned-v4/discord-v1'
  files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".txt")]
  folder_path = datapath + 'discord-data/v2/content/drive/Shareddrives/Datasets/cleaned-v4/discord-v2'
  files += [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".txt")]
  folder_path = datapath + 'discord-data/v3/content/drive/Shareddrives/Datasets/cleaned-v4/discord-v3'
  files += [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".txt")]
  texts = set()
  for file in tqdm(files, total=len(files)):
    line = ""
    with io.open(file, mode="r", encoding="utf-8") as f:
      line = f.read()
    line = line.replace('\\n', '\n')
    for message in line.split('\t'):
      author =  message.split(':')[0]
      text = ':'.join(message.split(':')[1:])
      text = ' '.join(text.split(' ')[:512]).strip()
      if (text != "") and (len(text.split()) > 2):
        texts.add(text)
    #break
  df = pd.DataFrame({'text': list(texts)})
  df.drop_duplicates(subset=['text'], inplace=True)
  df.dropna(inplace=True)
  df = df.reset_index(drop=True)
  df.to_csv(datapath + 'discord.csv.gz', index=False, escapechar='\\')
discord = pd.read_csv(datapath + 'discord.csv.gz', lineterminator='\n', escapechar='\\')
print('discord:', len(discord), '\n', discord[:10])

####################
#exit()
  
whatsapp['source'] = 'whatsapp'
gab['source'] = 'gab'
telegram['source'] = 'telegram'
twitter['source'] = 'twitter'
twitter2['source'] = 'twitter'
discord['source'] = 'discord'

df = pd.concat([whatsapp, gab, telegram, twitter, twitter2, discord], ignore_index=True, copy=False)
df.drop_duplicates(subset=['text'], inplace=True)
df.dropna(subset=['text'], inplace=True)

#langcheck by fasttext detector (the fastest detection to indicate language in tons of data)
results = [get_fasttext_lang(text) for text in tqdm(df['text'])]
lang, score = zip(*results)
df['fasttext_lang'] = lang
df['fasttext_lang_score'] = score

df.to_csv(datapath + 'combined.csv.gz', index=False, escapechar='\\')

print(df.fasttext_lang.value_counts())
