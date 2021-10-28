#!/usr/bin/env python
# coding: utf-8

# In[14]:

# Using the official provided BERT (also called LEIDOS BERT) for frmae classification
# The model should be downloaded from SocialSim wiki, this code is tje inference step
import json
import gzip
import string
urls = set()
id2urls = dict()
# Prepare text
with gzip.open('../../append_news_raw.json.gz','rt') as fin, open('data/input_append.csv', 'w') as fout:
    fout.write('\"id\",\"comment_text\"\n')
    for idx, line in enumerate(fin):
        js = json.loads(line)
        if js['url'] in urls:
            continue
        urls.add(js['url'])
        try:
            if 'text' in js and js['text']:
                text = js['text']
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = text.split()
                x = ''
                for token in text:
                    if len(token) > 0:
                        x += token+' '
                fout.write('\"'+str(idx)+'\",\"'+x.strip()+'\"\n')
                id2urls[idx] = js['url']
        except:
            pass


# In[23]:

# Run inference
get_ipython().system('bash run_bert.sh')


# In[11]:


narratives = ['benefits/connections/afghanistan', 'benefits/covid', 'benefits/development/energy', 'benefits/development/maritime', 'benefits/development/roads', 'benefits/jobs', 'controversies/china/border', 'controversies/china/debt', 'controversies/china/exploitation', 'controversies/china/funding', 'controversies/china/naval', 'controversies/china/uighur', 'controversies/pakistan/army', 'controversies/pakistan/bajwa', 'controversies/pakistan/baloch', 'controversies/pakistan/students', 'leadership/bajwa', 'leadership/khan', 'leadership/sharif', 'opposition/kashmir', 'opposition/propaganda']
url_done = dict()
# with open('../url2bert_prob.json') as fin:
#     for line in fin:
#         js = json.loads(line)
#         url_done[js['url']] = js


# In[12]:

# Collect result
import csv
import pandas as pd
scores = []
with open('data/output_append.csv') as fin:
    csv = pd.read_csv(fin) # , delimiter=',', quoting=csv.QUOTE_NONE)
    for row in csv.iterrows():
        _id = row[1]['id']
        prob = dict()
        for label in narratives:
            prob[label] = float(row[1][label])
        url_done[id2urls[_id]] = {'url': id2urls[_id], 'prob': prob}
# print(len(scores), len(id2urls))
# assert len(urls_list) == len(scores)


# In[13]:

# Dump the result
with open('../url2bert_prob_append.json', 'w') as fout:
    for u in url_done:
        fout.write(json.dumps(url_done[u]) + '\n')

