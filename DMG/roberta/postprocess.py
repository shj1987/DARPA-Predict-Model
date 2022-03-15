import json
import gzip
import string
import argparse
from pathlib import Path

urls = set()
id2urls = dict()


parser = argparse.ArgumentParser(description='Apply SocialSim BERT Models')
parser.add_argument('--path_raw_gz', type=str, required=True)
parser.add_argument('--path_id2url', type=str, required=True)
parser.add_argument('--path_roberta_output', type=str, required=True)
parser.add_argument('--path_roberta_url2prob', type=str, required=True)
parser.add_argument('--path_frame_names', type=str, default=Path(__file__).parent.joinpath('frame','tmp', 'classes.txt'))
args = parser.parse_args()

# path_raw_gz = '../../data/0830appended/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz'
# path_id2url = 'id2url_eval3-youtube.json'
# path_roberta_output = 'frame/0830eval3-youtubet.csv'
# path_roberta_url2prob = './frame/0830eval3-youtube_url2roberta_prob_append.json'

path_raw_gz = args.path_raw_gz
path_id2url = args.path_id2url
path_roberta_output = args.path_roberta_output
path_roberta_url2prob = args.path_roberta_url2prob


with gzip.open(path_raw_gz,'rt') as fin, open(path_id2url, 'w') as fout:
    for idx, line in enumerate(fin):
        js = json.loads(line)
        if js['url'] in urls:
            continue
        if 'title' not in js or 'text' not in js or js['title'] is None or js['text'] is None:
            continue
        urls.add(js['url'])
        try:
            if 'text' in js and js['text']:
                id2urls[idx] = js['url']
        except:
            pass
    json.dump(id2urls, fout, indent=4)
    
with open(args.path_frame_names) as rf:
    narratives = rf.readline().strip().split(',')
url_done = dict()

import csv
import json
import pandas as pd

with open(path_id2url) as rf:
    id2urls = json.load(rf)
    
scores = []
with open(path_roberta_output) as fin:
    csv = pd.read_csv(fin) # , delimiter=',', quoting=csv.QUOTE_NONE)
    for row in csv.iterrows():
        _id = str(row[1]['id'])
        prob = dict()
        for label in narratives:
            prob[label] = float(row[1][label])
        url_done[id2urls[_id]] = {'url': id2urls[_id], 'prob': prob}
        
        
with open(path_roberta_url2prob, 'w') as fout:
    for u in url_done:
        fout.write(json.dumps(url_done[u]) + '\n')