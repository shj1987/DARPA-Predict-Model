#!/usr/bin/env python
# coding: utf-8

# In[3]:

# Collect news classification results, compute the news-based input time-series for model group
# All the news articles are indexed with URL

import json
data = dict()
cnt = 0
# Load raw news
with open('news_text_raw_append.json', encoding='utf-8') as fin:
    for line in fin:
        js = json.loads(line)
        u = js['url']
        data[u] = {'date': js['date'], 'lang': js['lang'] if js['lang'] else 'unknown', 'title': js['title'] if js['title'] else '', 'article': js['article'] if js['article'] else ''}
        cnt += 1
print(cnt)
cnt = 0

# News frame derived from tweet using URL linking
# with open('url2tweet_frame.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         u = js['url']
#         if u in data:
#             data[u]['frames'] = js['frames']
#             cnt += 1
# print(cnt)
# cnt = 0

# News event code from gdelt using URL linking
import json
import gzip
from collections import defaultdict
for line in gzip.open('cp5-cpec.exogenous.gdelt.events.v1.json.gz', 'rt'):
    js = json.loads(line.strip())
    u = js['sourceurl']
    if u in data:
        if 'eventcode' not in data[u]:
            data[u]['eventcode'] = list()
        if type(js['EventCode']) == str:
            js['EventCode'] = [js['EventCode']]
        for ec in js['EventCode']:
            if ec not in data[u]['eventcode']:
                data[u]['eventcode'].append(ec)
        cnt += 1
print(cnt)
cnt = 0

# Load cleaned and phrased text
with open('news_text_en_phrased_append.json') as fin:
    for line in fin:
        js = json.loads(line)
        u = js['url']
        assert u in data
        data[u]['cleaned_title'] = js['title']
        data[u]['cleaned_article'] = js['article']
        data[u]['phrased_title'] = js['phrased_title']
        data[u]['phrased_article'] = js['phrased_article']
        cnt += 1
print(cnt)
cnt = 0

# Load LEIDOS BERT result
from collections import defaultdict
frame_cnt = defaultdict(int)
with open('frame_classification/url2bert_prob_append.json') as fin:
    for line in fin:
        js = json.loads(line)
        u = js['url']
        if u in data:
            data[u]['leidos_bert_score'] = js['prob']
            data[u]['leidos_bert_prediction'] = sorted(js['prob'], key=js['prob'].get, reverse=True)[:1]
#             data[u]['leidos_bert_prediction'] = [i for i in js['prob'] if js['prob'][i] >= 0.5]
            if len(data[u]['leidos_bert_prediction']) > 0:
                for f in data[u]['leidos_bert_prediction']:
                    frame_cnt[f] += 1
            cnt += 1
print(cnt)
cnt = 0

# Load WeSTClass result
from collections import defaultdict
frame_cnt = defaultdict(int)
with open('frame_classification/url2west_prob_append.json') as fin:
    for line in fin:
        js = json.loads(line)
        u = js['url']
        if u in data and js['prob']:
            data[u]['westclass_score'] = js['prob']
            data[u]['westclass_prediction'] = sorted(js['prob'], key=js['prob'].get, reverse=True)[:1]
#             data[u]['leidos_bert_prediction'] = [i for i in js['prob'] if js['prob'][i] >= 0.5]
            if len(data[u]['westclass_prediction']) > 0:
                for f in data[u]['westclass_prediction']:
                    frame_cnt[f] += 1
            cnt += 1
print(cnt)
cnt = 0

# Load Tuned-BERT result
# from collections import defaultdict
# frame_cnt = defaultdict(int)
# with open('frame_classification/url2tuned_bert_prob_append.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         u = js['url']
#         if u in data and js['prob']:
#             data[u]['tuned_bert_score'] = js['prob']
#             data[u]['tuned_bert_prediction'] = sorted(js['prob'], key=js['prob'].get, reverse=True)[:1]
# #             data[u]['leidos_bert_prediction'] = [i for i in js['prob'] if js['prob'][i] >= 0.5]
#             if len(data[u]['tuned_bert_prediction']) > 0:
#                 for f in data[u]['tuned_bert_prediction']:
#                     frame_cnt[f] += 1
#             cnt += 1
# print(cnt)
# cnt = 0

# Load LOTClass result
# from collections import defaultdict
# frame_cnt = defaultdict(int)
# with open('frame_classification/url2lot_prob_append.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         u = js['url']
#         if u in data and js['prob']:
#             data[u]['lotclass_score'] = js['prob']
#             data[u]['lotclass_prediction'] = sorted(js['prob'], key=js['prob'].get, reverse=True)[:1]
# #             data[u]['leidos_bert_prediction'] = [i for i in js['prob'] if js['prob'][i] >= 0.5]
#             if len(data[u]['lotclass_prediction']) > 0:
#                 for f in data[u]['lotclass_prediction']:
#                     frame_cnt[f] += 1
#             cnt += 1
# print(cnt)
# print(frame_cnt)
# cnt = 0

# Load string match result
frame_cnt = defaultdict(int)
with open('frame_classification/url2string_match_append.json') as fin:
    for line in fin:
        js = json.loads(line)
        u = js['url']
        if u in data:
            data[u]['string_match_score'] = js['string_match_score']
            data[u]['string_match_prediction'] = sorted(js['string_match_score'], key=js['string_match_score'].get, reverse=True)[:1]
            if len(data[u]['string_match_prediction']) > 0:
                for f in data[u]['string_match_prediction']:
                    frame_cnt[f] += 1
            cnt += 1
print(cnt)
print(frame_cnt)
cnt = 0

# Load relevance result
# with open('relevance_classification/url2relevance_append.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         u = js['url']
#         assert u in data
#         data[u]['relevance_score'] = js['relevance_score']
#         data[u]['relevance_prediction'] = js['relevance_prediction']
#         cnt += 1
# print(cnt)
# cnt = 0

# Prepare default values, 21 is the #frame for CP5
default_value = {'date' : None, # e.g. 2020-03-01
                 'lang' : 'unknown', # >70% are 'en'
#                  'frames': None, # list of str, derived from tweet-news url matching
                 'eventcode': None, # list of str, derived from gdelt-news url matching
                 
                 'title' : '',
                 'cleaned_title': None,
                 'phrased_title': None,
                 
#                  'relevance_prediction': None, # bool
                 'leidos_bert_prediction': None, # list of str
                 'string_match_prediction': None, # list of str
                 'westclass_prediction': None, # list of str
#                  'lotclass_prediction': None, # list of str
#                  'tuned_bert_prediction': None, # list of str
                 
#                  'relevance_score': None, # float
                 'leidos_bert_score': None, # dict, size=21
                 'string_match_score': None, # dict, size=21
                 'westclass_score': None, # dict, size=21
#                  'lotclass_score': None, # dict, size=21
#                  'tuned_bert_score': None, # dict, size=21                 
                 
                 'article' : '',
                 'cleaned_article': None,
                 'phrased_article': None}
for i in default_value:
    for u in data:
        if i not in data[u]:
            data[u][i] = default_value[i]
            # if i.endswith('_score') and data[u][i] and type(data[u][i]) == dict:
            #     assert len(data[u][i]) == 21

# The date format can be tricky
def format_date(d):
    if not d:
        return d
    if d.find('T') != -1:
        d = d.split('T')[0]
    tmp = list(d.split('-'))
    assert len(tmp) == 3
    if len(tmp[0]) == 4 and len(tmp[1]) == 2 and len(tmp[2]) == 2:
        pass
    elif len(tmp[0]) == 2 and len(tmp[1]) == 2 and len(tmp[2]) == 2:
        tmp[0] = '20' + tmp[0]
    elif len(tmp[0]) == 2 and len(tmp[1]) == 2 and len(tmp[2]) == 4:
        tmp = [tmp[2], tmp[1], tmp[0]]
    else:
        print(d, tmp)
        print()
    if tmp[0] > '2020' or tmp[0] < '2010':
        tmp[0] = '2020'
    return '-'.join(tmp)

for u in data:
    data[u]['url'] = u
    data[u]['date'] = format_date(data[u]['date'])
    


# In[4]:



import csv
# with open('news_results_append.csv', 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=['url'] + list(default_value.keys()))
#     writer.writeheader()
#     for u in data:
#         writer.writerow(data[u])

# Dump the aggregated classification result
with open('news_results_append.json', 'w', encoding='utf-8', errors='ignore') as fout:
    for u in data:
        fout.write(json.dumps(data[u]) + '\n')

