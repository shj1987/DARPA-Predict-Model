#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Collect results from previous rolls
'''
import json
url2results = dict()
with open('news_text_en_phrased.json') as fin:
    for line in fin:
        js = json.loads(line)
        url2results[js['url']] = js
'''

# In[7]:


from tqdm import tqdm
import json
import gzip
import sys

# path_i_file = './data/NewsArticles/cp6.ea.newsarticles.training.v1.json.gz'
# path_o_file = 'news_text_raw_append.json'

# path_i_file = './data/0830appended/eval1-cp6.ea.newsarticles.jamii.json.gz'
# path_o_file = './data/0830appended/cleaned_eval1-cp6.ea.newsarticles.jamii.json'

# path_i_file = './data/0830appended/eval1-cp6.ea.newsarticles.reddit.json.gz'
# path_o_file = './data/0830appended/cleaned_eval1-cp6.ea.newsarticles.reddit.json'

# path_i_file = './data/0830appended/eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json.gz'
# path_o_file = './data/0830appended/cleaned_eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json'

# path_i_file = './data/0830appended/eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json.gz'
# path_o_file = './data/0830appended/cleaned_eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json'

# path_i_file = './data/0830appended/eval3_cp6.ea.newsarticles.twitter.2020-12-21_2021-01-10.json.gz'
# path_o_file = './data/0830appended/cleaned_eval3_cp6.ea.newsarticles.twitter.2020-12-21_2021-01-10.json'

path_i_file = sys.argv[1]
path_o_file = sys.argv[2]


if True:
    # Read appended raw news
    url2news = dict()
    news = []
    with gzip.open(path_i_file, 'rt') as fin:
        for line in tqdm(fin):
            obj = json.loads(line)
            if 'title' not in obj or 'text' not in obj or obj['title'] is None or obj['text'] is None:
                continue
            nid = len(news)
            news.append(obj)
            url2news[news[-1]['url']] = len(news) - 1
            #if len(news) > 1000:
            #    break
    # Read gdelt events
    import json
    from collections import defaultdict
    events = []
    for line in gzip.open(sys.argv[3], 'rt'):
        events.append(json.loads(line.strip()))
        #if len(events) > 10000:
        #    break
    #print(len(events))
    url2code = defaultdict(list)
    for idx, js in enumerate(events):
        url2code[js['sourceurl']].append(idx)

    #print(len(set(url2code) & set(url2news)))

    # Link gdelt event and news by URL
    for u in set(url2code) & set(url2news):
        if len(url2code[u]) > 0:
            news[url2news[u]]['events'] = url2code[u]


    # In[9]:


    # Extract dates
    for i in news:
        ti = None
        if 'article_extracted_date' in i['extension']:
            ti = i['extension']['article_extracted_date']
        if not ti:
            ti = i['extension']['earliest_reference_datetime']
            if ti:
                ti = ti.split('T')[0]
        if not ti:
            ti = i['extension']['gdelt_reference_datetime']
            if ti:
                ti = ti.split('T')[0]
        i['date'] = ti
    #print(news)


#     print(len(narratives))

    # Extract language
    import fasttext
    #fasttext.FastText.eprint = print
    lid_model = fasttext.load_model('lid.176.ftz')
    for x in tqdm(news):
        try:
            text = x['title'].replace('\n', ' ')
            lang = lid_model.predict(text)[0][0][-2:]
            assert len(lang) == 2
        except:
            lang = 'unkown'
        x['lang'] = lang


    # In[10]:

    # Dump raw news (all language) to json with some meta data
    with open(path_o_file, 'w') as fout:
        for i in tqdm(news):
            if 'title' in i and 'text' in i:
                fout.write(json.dumps({'title':i['title'],
                            'article':i['text'],
                            'date':i['date'],
                            'url':i['url'],
                            'lang':i['lang']}) + '\n')
else:
    news = []
    with open(path_o_file, 'r') as fin:
        for line in fin:
            tmp = json.loads(line)
            tmp['text'] = tmp['article']
            news.append(tmp)
    print(len(news))


# In[18]:

import nltk
import string
from nltk.corpus import wordnet
# nltk.download('words')
# tc = nltk.classify.textcat.TextCat()
import fasttext
#fasttext.FastText.eprint = print
import sys
from multiprocessing import Pool

ascii = set(string.printable)

from collections import defaultdict
import spacy
nlp = spacy.load('en_core_web_sm', exclude=["ner", "tagger", "parser", "lemmatizer"])
nlp.add_pipe('sentencizer')
#nlp.add_pipe(nlp.create_pipe('sentencizer'))
#import en_core_web_sm
#nlp = en_core_web_sm.load()

clean_text = []
spliter = [' | ', ' - ', ' ??? ', ' -- ', ' ??? ', ' ?? ']

import re
import string

printable = set(string.printable)

def clean0(i, newline=True): # make ' and " consistent
    i = re.sub(r'???', '"', i)
    i = re.sub(r'???', '"', i)
    i = re.sub(r'???', "'", i)
    i = re.sub(r'???', "'", i)
    i = ''.join(filter(lambda x: x in printable, i))
    return i.strip(' ')
def clean1(text, title_raw): # remove html codes in text
    tmp = []
    ban_words = ['(http)', '(\.com)', '( = )', '(==)', '(&&)', '(\|\|)', '(?? Copyright )', 'Copyright ??', 'function\s.+\(.*\)\s*\{'] # '( \| )', 
    for i in text.split('\n'):
        if len(i.split()) >= 5:
            if not re.search('|'.join(ban_words), i) and i != title_raw:
                tmp.append(clean0(i))
    return tmp
def clean3(doc, spliter='\n'): # segement into sentences
    if type(doc) == list:
        doc = '\n'.join(doc)
    doc = nlp(doc)
    tmp = []
    for i in doc.sents:
        tmp.append(' '.join(map(str, i)))
    return spliter.join(tmp)
def clean4(texts): # remove duplicated sentences in article
    url_set = set()
    title_set = set()
    tmp = []
    for js in tqdm(texts):
        u = js['title']
        if js['url']:
            u = js['url']
        t = js['title']
        if u in url_set or t in title_set:
            continue
        url_set.add(u)
        title_set.add(t)
        tmp.append(js)
    return tmp    
def clean2(texts): # remove duplicated sentences in article
    count = defaultdict(int)
    for js in tqdm(texts):
        for sent in js['article']:
            _ = re.sub('\d+', '@num@', sent).lower()
            count[_] += 1
    for js in tqdm(texts):
        doc = []
        for sent in js['article']:
            _ = re.sub('\d+', '@num@', sent).lower()
            if count[_] <= 1:
                doc.append(sent)
        title = clean3(js['title'], ' ')
        article = clean3(doc)
        js['title'] = title
        js['article'] = article

err = 0
for idx, i in enumerate(tqdm(news)):
    if 'title' in i and i['title'] and i['lang'] == 'en':
        text = i['title'].replace('\n', ' ')
        for s in spliter:
            if text.find(s) != -1:
                tmp = []
                for j in text.split(s):
                    tmp.append(j)
                tmp = sorted(tmp, key=lambda x:-len(x))
                text = tmp[0]
        if len(text.split()) >= 5 and len(text) >= 5:
            clean_text.append({'title':clean0(text[:10000], newline=False),
                               'article':clean1(i['text'][:10000], i['title'][:10000]),
                               'date':i['date'],
                               'url':i['url'],
                               'lang':i['lang']
                              })
    else:
        err += 1
clean_text = clean4(clean_text)
clean2(clean_text)

print('error rate:', err / len(news))
import json
# with open(folder + f'news_text_en_append.json', 'w', encoding='utf-8') as fout:
#     for js in clean_text:
#         fout.write(json.dumps(js) + '\n')


# In[22]:

# # Phrasing
# phrases = []
# from flashtext import KeywordProcessor
# keyword_processor = KeywordProcessor()
# phrase2score = dict()
# with open('AutoPhrase.txt') as fin:
#     for line in fin:
#         score, phrase = line.strip().split('\t')
#         keyword_processor.add_keyword(phrase, phrase.replace(' ', '_'))
#         phrase = phrase.replace(' ', '_')
#         phrases.append(phrase)
#         phrase2score[phrase] = float(score)
# print('#phrases', len(phrases))
# import re

# for js in clean_text:
#     text = (js['title'].strip()).replace('\n', ' ').replace('\r', ' ')
#     if len(text) > 0:
#         js['phrased_title'] = keyword_processor.replace_keywords(text)
#     else:
#         js['phrased_title'] = ''
#     text = (js['article'].strip()).replace('\n', ' ').replace('\r', ' ')
#     if len(text) > 0:
#         js['phrased_article'] = keyword_processor.replace_keywords(text)
#     else:
#         js['phrased_article'] = ''

# # Dump cleaned and phrase (english) news to json file
# import json
# with open(f'news_text_en_phrased_append.json', 'w', encoding='utf-8') as fout:
#     for js in clean_text:
#         fout.write(json.dumps(js) + '\n')

