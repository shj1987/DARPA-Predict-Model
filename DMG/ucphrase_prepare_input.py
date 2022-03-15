import json
from pprint import pprint
from tqdm.auto import tqdm
import argparse


# path_i_file = '/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval1-cp6.ea.newsarticles.jamii.json'
# path_o_file = '/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval1-cp6.ea.newsarticles.jamii.json'

# path_i_file = '/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval1-cp6.ea.newsarticles.reddit.json'
# path_o_file = '/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval1-cp6.ea.newsarticles.reddit.json'

# path_i_file = '/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json'
# path_o_file = '/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json'

# path_i_file = '/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json'
# path_o_file = '/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json'

# path_i_file = '/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
# path_o_file = '/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'

parser = argparse.ArgumentParser()
parser.add_argument("--i_path", "-i", type=str, required=True)
parser.add_argument("--o_path", "-o", type=str, required=True)
args = parser.parse_args()


with open(args.i_path) as rf:
    lines = rf.read().splitlines()
    docs = [json.loads(l) for l in lines]
    docs = [d for d in docs if d['article']]
    

def clean(doci, doc):
    article = doc['article']
    sents = article.split('\n')
    sents = [s.strip(' |').replace('  ', ' ') for s in sents]
    sents = [s for s in sents if s and len(s.split()) > 3]
    if not sents:
        return None
    newdoc = doc.copy()
    newdoc.pop('article')
    newdoc.pop('title')
    newdoc['_id_'] = str(doci)
    newdoc['sents'] = [doc['title']] + sents
    return newdoc

newdocs = [clean(i, d) for i, d in tqdm(list(enumerate(docs)))]
newdocs = [d for d in newdocs if d]
newlines = [json.dumps(d) for d in newdocs]
with open(args.o_path, 'w') as wf:
    wf.write('\n'.join(newlines))