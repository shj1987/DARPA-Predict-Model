#!/usr/bin/env python
# coding: utf-8

# In[9]:

# This is a simple classifier using frame keyword matching.

input_file = 'news_text_raw_append.json'
output_file = 'url2string_match_append.json'

url2results = dict()
# import json
# with open('url2string_match.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         url2results[js['url']] = js

# Frames here
narratives = ['benefits/connections/afghanistan', 'benefits/covid', 'benefits/development/energy', 'benefits/development/maritime', 'benefits/development/roads', 'benefits/jobs', 'controversies/china/border', 'controversies/china/debt', 'controversies/china/exploitation', 'controversies/china/funding', 'controversies/china/naval', 'controversies/china/uighur', 'controversies/pakistan/army', 'controversies/pakistan/bajwa', 'controversies/pakistan/baloch', 'controversies/pakistan/students', 'leadership/bajwa', 'leadership/khan', 'leadership/sharif', 'opposition/kashmir', 'opposition/propaganda']
import json
from collections import defaultdict
url2title = dict()
url2string_match_prediction = dict()
url2string_match_score = dict()
# keywords = dict()
# with open('WeSTClass/news_en/keywords.txt') as fin:
#     for nar, line in zip(narratives, fin):
#         keywords[nar] = list(line.strip().split(':')[1].split(','))
import json

# Keywords are manually set
keywords = json.load(open('keywords.txt'))
# print(keywords)

# Countings are used for tie break
global_cnt = {"benefits/covid": 485344, "benefits/development/roads": 74575, "benefits/development/maritime": 44606, "benefits/development/energy": 55211, "leadership/bajwa": 18469, "opposition/propaganda": 179989, "opposition/kashmir": 484026, "benefits/jobs": 34899, "leadership/khan": 164996, "controversies/china/uighur": 49202, "controversies/china/border": 1280130, "controversies/china/debt": 55764, "controversies/china/exploitation": 118370, "controversies/pakistan/students": 10867, "controversies/pakistan/baloch": 172937, "controversies/pakistan/army": 82715, "benefits/connections/afghanistan": 5863, "leadership/sharif": 92467, "controversies/pakistan/bajwa": 15371, "controversies/china/funding": 1425, "controversies/china/naval": 256}
# print(len(global_cnt))
# with open('../url2tweet_frame.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         for f in js['frames']:
#             global_cnt[f] += 1
cnt = 0
with open(f'../{input_file}') as fin:
    for line in fin:
        js = json.loads(line)
#         text = (js['phrased_title'] if js['phrased_title'] else ' ').replace('\r', ' ').replace('\n', ' ').strip()
        text = (js['title'] if js['title'] else ' ').replace('\r', ' ').replace('\n', ' ').strip()
        url2title[js['url']] = js['title']
        nar_cnt = defaultdict(int)
        found = 0
        for nar in narratives:
            for key in keywords[nar]:
                key = key.replace('_', ' ')
                if text.lower().find(key) != -1:
                    nar_cnt[nar] += 1
                    found += 1
        if found >= 1:
            cnt += 1
            url2string_match_prediction[js['url']] = sorted(nar_cnt, key=lambda x: (nar_cnt[x], global_cnt[x]), reverse=True)[0]
            url2string_match_score[js['url']] = {i: nar_cnt[i] for i in narratives}
for u in url2string_match_score:
    url2results[u] = {'url': u, 'string_match_score': url2string_match_score[u], 'string_match_prediction': url2string_match_prediction[u]}
            
with open(output_file, 'w') as fout:
    for u in url2results:
        fout.write(json.dumps(url2results[u]) + '\n')
        
print(cnt / len(url2title))

