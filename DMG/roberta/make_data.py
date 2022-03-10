import json
import gzip
import string
import argparse


parser = argparse.ArgumentParser(description='Apply SocialSim BERT Models')
parser.add_argument('--i_path', type=str, required=True)
parser.add_argument('--o_path', type=str, required=True)
args = parser.parse_args()


urls = set()
id2urls = dict()
# with gzip.open('../../data/NewsArticles/cp6.ea.newsarticles.training.v1.json.gz','rt') as fin, open('data/input_append.csv', 'w') as fout:
# with gzip.open('../../data/0830appended/eval1-cp6.ea.newsarticles.jamii.json.gz','rt') as fin, open('data/0830jamii_input_append.csv', 'w') as fout:
# with gzip.open('../../data/0830appended/eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json.gz','rt') as fin, open('data/0830youtube_input_append.csv', 'w') as fout:
# with gzip.open('../../data/0830appended/eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json.gz','rt') as fin, open('data/0830twitter_input_append.csv', 'w') as fout:
# with gzip.open('../../data/0830appended/eval1-cp6.ea.newsarticles.reddit.json.gz','rt') as fin, open('data/0830reddit_input_append.csv', 'w') as fout:
# with gzip.open('../../data/0830appended/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz','rt') as fin, open('data/0830eval3-youtube_input_append.csv', 'w') as fout:
with gzip.open(args.i_path,'rt') as fin, open(args.o_path, 'w') as fout:
    fout.write('\"id\",\"comment_text\"\n')
    for idx, line in enumerate(fin):
        js = json.loads(line)
        if js['url'] in urls:
            continue
        if 'title' not in js or 'text' not in js or js['title'] is None or js['text'] is None:
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