#!/usr/bin/env python
# coding: utf-8

# In[3]:

# Collect all the news text
input_files = ['news_text_en_phrased.json']

import os
for root, dirs, files in os.walk('./', topdown=False):
    for name in files:
        if name.endswith('news_text_en_phrased_append.json'):
            input_files.append(name)
print(input_files)


# In[5]:



url2results = dict()
narratives = ['benefits/connections/afghanistan', 'benefits/covid', 'benefits/development/energy', 'benefits/development/maritime', 'benefits/development/roads', 'benefits/jobs', 'controversies/china/border', 'controversies/china/debt', 'controversies/china/exploitation', 'controversies/china/funding', 'controversies/china/naval', 'controversies/china/uighur', 'controversies/pakistan/army', 'controversies/pakistan/bajwa', 'controversies/pakistan/baloch', 'controversies/pakistan/students', 'leadership/bajwa', 'leadership/khan', 'leadership/sharif', 'opposition/kashmir', 'opposition/propaganda']
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

# Prepare WeSTClass corpus
with open('frame_classification/WeSTClass11/news_manual/corpus.txt', 'w') as fout:
    for line in open_all():
        js = json.loads(line)
        text = ((js['phrased_title'] if js['phrased_title'] else ' ') + ' ' + (js['phrased_article'] if js['phrased_article'] else ' ')).replace('\r', ' ').replace('\n', ' ').strip()
        url2title[js['url']] = js['phrased_title']
        if len(text) > 10 and len(text.split(' ')) > 5:
            fout.write(' '.join(text.split(' ')[:480]) + '\n')
    close_all()


# In[6]:

# Clean up previous runs
get_ipython().system('rm -rf frame_classification/WeSTClass11/results/*')
get_ipython().system('rm -f frame_classification/WeSTClass11/news_manual/embedding')
get_ipython().system('rm -f frame_classification/WeSTClass11/news_manual/out.txt')


# In[7]:

# Run WeSTClass
get_ipython().system('cd frame_classification/WeSTClass11/ && bash test.sh')


# In[6]:

url2west = dict()
selected_nar = ['controversies/pakistan/students', 'leadership/sharif', 'leadership/bajwa', 'controversies/china/uighur', 'controversies/china/border', 'benefits/development/roads', 'controversies/pakistan/baloch', 'benefits/jobs', 'opposition/propaganda', 'benefits/development/energy', 'controversies/pakistan/bajwa']
A = ['bajwa', 'baloch', 'benefits', 'border', 'china', 'controversies', 'development', 'energy', 'jobs', 'leadership', 'opposition', 'pakistan', 'propaganda', 'roads', 'sharif', 'students', 'uighur', 'cpec']
B = [['gen_bajwa','qamar_javed_bajwa','gen_qamar_javed_bajwa'],['baloch','balochistan','balochi','baluch'],['win-win','benefit','benefits'],['siachen_glacier','karakoram_pass','boundary','borders','frontier','indo','along','lac','crossing','sikkim','line_of_actual_control'],['china','beijing','chinese','xi','taiwan','xi_jinping','chinas'],['backlash_against','controversy','rumors','rumours','speculation','backlash','bizarre_claim','conspiracy_theories','fake_news'],['construction','development','projects'],['dam','petroleum','gas','oil','natural_gas','power_generation','renewable_energy','clean_energy','crude_oil','fertilizer','mining'],['employment','jobs','pakistan'],['coalition','alliance','leader','factions','political','leaders','party','supporters'],['opposition_leader','opposition_parties'],['pakistan','pak','pm_imran','imran_khan','islamabad','pakistani'],['propaganda','disinformation','fake_news','misinformation','false_information'],['underpasses','underpass','mortal_way','expressway','bridge','bridges'],['shehbaz_sharif','shahbaz_sharif','nawaz_sharif'],['students','online_classes','classes','university','universities','school','schools','classmates'],['uighurs','uyghur','uyghurs','xinjiang','minority','racial','discrimination'],['cpec','china','pakistan','china-pakistan','economic_corridor','india']]
topic2keywords_manual = {i : j for i, j in zip(A, B)}
nar2key = {i: [j for j in i.split('/')] for i in narratives}
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
    if len(text) > 10 and len(text.split(' ')) > 5:
        valid_url.append(js['url'])
close_all()

# Collect results
fin1 = open_all()
with open('frame_classification/WeSTClass11/news_manual/out.txt') as fin2:
    for u, line2 in zip(valid_url, fin2):
        classes = ['bajwa', 'baloch', 'benefits', 'border', 'china', 'controversies', 'development', 'energy', 'jobs', 'leadership', 'opposition', 'pakistan', 'propaganda', 'roads', 'sharif', 'students', 'uighur']
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


# Only top 300 most confident results are kept for a frame
chosen_url = set()
for nar in selected_nar:
    urls = list(url2west)
    urls = sorted(urls, key=lambda u:url2west[u][nar], reverse=True)
    for u in urls[:300]:
        chosen_url.add(u)

# The remaining articles are using keywords matching
import json
keywords = json.load(open('frame_classification/keywords.txt'))
global_cnt = {"benefits/covid": 485344, "benefits/development/roads": 74575, "benefits/development/maritime": 44606, "benefits/development/energy": 55211, "leadership/bajwa": 18469, "opposition/propaganda": 179989, "opposition/kashmir": 484026, "benefits/jobs": 34899, "leadership/khan": 164996, "controversies/china/uighur": 49202, "controversies/china/border": 1280130, "controversies/china/debt": 55764, "controversies/china/exploitation": 118370, "controversies/pakistan/students": 10867, "controversies/pakistan/baloch": 172937, "controversies/pakistan/army": 82715, "benefits/connections/afghanistan": 5863, "leadership/sharif": 92467, "controversies/pakistan/bajwa": 15371, "controversies/china/funding": 1425, "controversies/china/naval": 256}

def string_match(text):
    nar_cnt = defaultdict(int)
    found = 0
    for nar in narratives:
        for key in keywords[nar]:
            if text.lower().find(key) != -1:
                nar_cnt[nar] += 1
                found += 1
    if found >= 1:
        return {i: nar_cnt[i] for i in narratives}
    else:
        return None
    
for u in url2west:
    string_match_result = string_match(url2title[u] if u in url2title else '')
    if u in chosen_url:
        for nar in narratives:
            if nar not in url2west[u]:
                url2west[u][nar] = 0.0
        if 'nar' in url2west[u]:
            url2west[u].pop('nar')
        if len(url2west[u]) != 21:
            print(len(url2west[u]), url2west[u])
            assert False
    else:
        url2west[u] = string_match_result
#         assert len(url2west[u]) == 21
        
# Dump the results
with open('frame_classification/url2west_prob_append.json', 'w') as fout:
    for u in url2west:
        i = url2west[u]
        if i:
            fout.write(json.dumps({'url': u, 'prob': i}) + '\n')
        


# In[8]:


keywords


# In[57]:



# for nar in selected_nar:
#     urls = list(url2west)
#     urls = sorted(urls, key=lambda u:url2west[u][nar] if url2west[u] else 0.0, reverse=True)
#     for u in urls[:30]:
#         print(nar, url2title[u])

