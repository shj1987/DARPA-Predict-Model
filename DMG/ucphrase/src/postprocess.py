import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path_cleaned_corpus", type=str)
parser.add_argument("--path_ucphrase_decoded", type=str)
parser.add_argument("--path_output", type=str)
args = parser.parse_args()
    
    
with open(args.path_cleaned_corpus) as rf:
    raw_docs = [json.loads(l) for l in rf.readlines()]
    raw_docs = [d for d in raw_docs if d['article']]
    
with open(args.path_ucphrase_decoded) as rf:
    ucphrase_id2doc = json.load(rf)
    
from tqdm.auto import tqdm

GTOKEN = 'Ġ'

def sents2text(sents):
    text = ''
    for sent in sents:
        tokens = sent['tokens']
        for i_l, i_r, _ in sent['spans']:
            for i in range(i_l + 1, i_r + 1):
                if tokens[i].startswith(GTOKEN):
                    tokens[i] = '_' + tokens[i][1:]
        sent_text = ''.join(tokens).replace(GTOKEN, ' ').lower().strip()
        text += sent_text + ' '
    return text

new_docs = []
for i, doc in tqdm(enumerate(raw_docs), total=len(raw_docs)):
    new_doc = doc.copy()
    if str(i) not in ucphrase_id2doc:
        continue
    ucphrase_sents = ucphrase_id2doc[str(i)]
    new_doc['phrased_title'] = sents2text(ucphrase_sents[:1])
    new_doc['phrased_article'] = sents2text(ucphrase_sents[1:])
    new_docs.append(new_doc)
    
    
with open(args.path_output, 'w') as wf:
    wf.write('\n'.join([json.dumps(d) for d in new_docs]))