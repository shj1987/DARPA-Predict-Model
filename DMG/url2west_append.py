#!/usr/bin/env python
# coding: utf-8

# In[3]:

# Collect all the news text
input_files = []

import os
for root, dirs, files in os.walk('/shared/data2/qiz3/socialsim/data/0830appended/', topdown=False):
    for name in files:
        if name.startswith('ucphrase_eval_append'):
            input_files.append('/shared/data2/qiz3/socialsim/data/0830appended/' + name)
    
print(input_files)


# In[5]:



url2results = dict()
#narratives = ['benefits/connections/afghanistan', 'benefits/covid', 'benefits/development/energy', 'benefits/development/maritime', 'benefits/development/roads', 'benefits/jobs', 'controversies/china/border', 'controversies/china/debt', 'controversies/china/exploitation', 'controversies/china/funding', 'controversies/china/naval', 'controversies/china/uighur', 'controversies/pakistan/army', 'controversies/pakistan/bajwa', 'controversies/pakistan/baloch', 'controversies/pakistan/students', 'leadership/bajwa', 'leadership/khan', 'leadership/sharif', 'opposition/kashmir', 'opposition/propaganda']
narratives = ['covid', 'covid/assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', 'travel', 'un'] 
selected_nar = []
import json
from collections import defaultdict
import itertools
url2title = dict()

def open_all():
    global file_tmp
    file_tmp = [open(f'{input_file}') for input_file in input_files]
    return itertools.chain.from_iterable(file_tmp)

def close_all():
    global file_tmp
    for i in file_tmp:
        i.close()

if True:
    # Prepare WeSTClass corpus
    with open('WeSTClass/news_manual/corpus.txt', 'w') as fout:
        for line in open_all():
            js = json.loads(line)
            text = ((js['phrased_title'] if js['phrased_title'] else ' ') + ' ' + (js['phrased_article'] if js['phrased_article'] else ' ')).replace('\r', ' ').replace('\n', ' ').strip()
            url2title[js['url']] = js['phrased_title']
            if len(text) > 10 and len(text.split(' ')) > 5:
                fout.write(' '.join(text.split(' ')[:480]) + '\n')
        close_all()


# In[6]:

# Clean up previous runs
#get_ipython().system('rm -rf WeSTClass/results/*')
#get_ipython().system('rm -f WeSTClass/news_manual/embedding')
os.system('rm -f WeSTClass/news_manual/out.txt')


# In[7]:

# Run WeSTClass
os.system('cd WeSTClass && python main.py --dataset news_manual --sup_source keywords --model cnn --with_evaluation False')


# In[6]:

url2west = dict()
selected_nar = ['covid', 'assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', 'travel', 'un'] 
A = ['covid', 'assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', '#', 'un']
B = [['covid', 'coronavirus', 'covid19', 'pandemic', 'corona_virus', 'lockdowns'], ['assistance',
 'chinese_assistance',
 'china_assistance',
 'financial_assistance',
 'financial_support',
 'resource_mobilization'],['debt', 'debts', 'chinese_loans', 'african_debt', 'repayments'],['environment',
 'environmental',
 'pollution',
 'national_park',
 'preservation',
 'wildlife'],['infrastructure',
 'road',
 'roads',
 'railways',
 'ports',
 'railway',
 'public_infrastructure'],['mistreatment',
 'maltreatment',
 'abuse',
 'abusement',
 'ill_treatment',
 'detention',
 'detainee'],
['prejudice',
 'racial_prejudice',
 'racism',
 'racist',
 'xenophobia',
 'discrimination',
 'religious_discrimination'], 
['travel',
 'flights',
 'flight',
 'international_travel',
 'traveler',
 'travelers',
 'vacations'], ['un', 'united_nations', 'un_secretary', 'security_council']]
topic2keywords_manual = {i : j for i, j in zip(A, B)}
nar2key = {i:i for i in narratives}
nar2keyweight = set()
import numpy as np
cnt = 0
valid_url = []
# with open(f'../{input_file}') as fin1:
fin1 = open_all()
for line in fin1:
    js = json.loads(line)
    text = ((js['phrased_title'] if js['phrased_title'] else ' ') + ' ' + (js['phrased_article'] if js['phrased_article'] else ' ')).replace('\r', ' ').replace('\n', ' ').strip()
    url2title[js['url']] = js['phrased_title']
    #if len(text) > 10 and len(text.split(' ')) > 5:
    valid_url.append(js['url'])
close_all()

# Collect results
fin1 = open_all()
with open('WeSTClass11/news_manual/out.txt') as fin2:
    for u, line2 in zip(valid_url, fin2):
        classes = ['covid', 'assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', 'travel', 'un']
        raw = {i : j for i, j in zip(classes, list(map(float, line2.strip().split(','))))}
        west = dict()
        for nar in selected_nar:
            if nar in nar2keyweight:
                score = 1.0
                w = 0
                for key in nar2keyweight[nar]:
                    _s = raw[key]
                    score *= np.power(_s, nar2keyweight[nar][key])
                    w += nar2keyweight[nar][key]
                score = np.power(score, 1.0 / w)
            else:
                nk = topic2keywords_manual[nar] if nar in topic2keywords_manual else nar2key[nar]
                score = 1.0
                weight = [0.0 for i in range(len(nk))]
                weight[-1] += 0.8
                for i in range(len(nk) - 1):
                    weight[i] += 0.2 / (len(nk) - 1)
                for key, w in zip(nk, weight):
                    _s = raw[key]
                    score *= np.power(_s, w)
                score = np.power(score, 1.0 / sum(weight))
            west[nar] = score
        url2west[u] = west
        cnt += 1
    close_all()
print(cnt)


# In[7]:

json.dump(url2west, open('./data/ft_retrieval_westclass_append.json', 'w'))
# Only top 300 most confident results are kept for a frame
if False:
    chosen_url = set()
    for nar in selected_nar:
        urls = list(url2west)
        urls = sorted(urls, key=lambda u:url2west[u][nar], reverse=True)
        for u in urls[:1000]:
            chosen_url.add(u)