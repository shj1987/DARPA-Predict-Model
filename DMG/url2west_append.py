#!/usr/bin/env python
# coding: utf-8

# Collect all the news text
input_files = []
import os
import sys
for file in os.listdir(sys.argv[1]):
    #print(files)
    if os.path.isfile(os.path.join(sys.argv[1], file)):
        if file.startswith(sys.argv[2]):
            print(file)
            input_files.append(sys.argv[1] + file)
    
print(input_files)

url2results = dict()
#narratives = ['benefits/connections/afghanistan', 'benefits/covid', 'benefits/development/energy', 'benefits/development/maritime', 'benefits/development/roads', 'benefits/jobs', 'controversies/china/border', 'controversies/china/debt', 'controversies/china/exploitation', 'controversies/china/funding', 'controversies/china/naval', 'controversies/china/uighur', 'controversies/pakistan/army', 'controversies/pakistan/bajwa', 'controversies/pakistan/baloch', 'controversies/pakistan/students', 'leadership/bajwa', 'leadership/khan', 'leadership/sharif', 'opposition/kashmir', 'opposition/propaganda']
#narratives = ['covid', 'covid/assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', 'travel', 'un'] 
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
os.system('rm -f WeSTClass/news_manual/embedding')
os.system('rm -f WeSTClass/news_manual/out.txt')


# In[7]:

# Run WeSTClass
os.system('cd WeSTClass && python main.py --dataset news_manual --sup_source keywords --model cnn --with_evaluation False')


# In[6]:

url2west = []
with open('WeSTClass/news_manual/classes.txt') as IN:
    for line in IN:
        selected_nar.append(line.strip().split(':')[1])

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
with open('WeSTClass/news_manual/out.txt') as fin2, open('./data/ft_retrieval_westclass_append.json', 'w') as OUT:
    for u, line2 in zip(valid_url, fin2):
        raw = {i : j for i, j in zip(selected_nar, list(map(float, line2.strip().split(','))))}
        #url2west.append({'url': u, 'prob':raw})
        OUT.write(json.dumps({'url': u, 'prob':raw})+'\n')
        cnt += 1
    close_all()
print(cnt)

#json.dump(url2west, open('./data/ft_retrieval_westclass_append.json', 'w'))
# Only top 300 most confident results are kept for a frame
if False:
    chosen_url = set()
    for nar in selected_nar:
        urls = list(url2west)
        urls = sorted(urls, key=lambda u:url2west[u][nar], reverse=True)
        for u in urls[:1000]:
            chosen_url.add(u)