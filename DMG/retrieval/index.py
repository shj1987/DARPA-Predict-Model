import os
import json
import whoosh
import argparse
from tqdm import tqdm
from whoosh.index import create_in

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ix_name", type=str, required=True)
parser.add_argument("--ix_dir", type=str, required=True)
parser.add_argument("--dir_retrieved_docs", type=str, required=True)
parser.add_argument("--path_input_cleaned_corpus", type=str, required=True)
parser.add_argument("--path_twitter_timeseries_pickle", type=str, required=True)
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
args = parser.parse_args()


schema = whoosh.fields.Schema(
    url=whoosh.fields.ID(stored=True),
    title=whoosh.fields.TEXT(stored=True),
    article=whoosh.fields.TEXT(stored=True),
    date=whoosh.fields.TEXT(stored=True))


# ix_name = ''
# ix_dirname = 'index'
# path_input_file = '../dmg_prob_append.json'

# ix_name = '0830jammi'
# ix_dirname = 'index_' + ix_name
# path_input_file = '../../data/0830appended/cleaned_eval1-cp6.ea.newsarticles.jamii.json'

# ix_name = '0830reddit'
# ix_dirname = 'index_' + ix_name
# path_input_file = '../../data/0830appended/cleaned_eval1-cp6.ea.newsarticles.reddit.json'

# ix_name = '0830twitter'
# ix_dirname = 'index_' + ix_name
# path_input_file = '../../data/0830appended/cleaned_eval2-cp6.ea.newsarticles.twitter.2020-11-30_2020-12-20.json'

# ix_name = '0830youtube'
# ix_dirname = 'index_' + ix_name
# path_input_file = '../../data/0830appended/cleaned_eval2-cp6.ea.newsarticles.youtube.2020-11-30_2020-12-20.json'

# ix_name = '0830eval3-youtube'
# ix_dirname = 'index_' + ix_name
# path_input_file = '../../data/0830appended/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'

ix_name = args.ix_name
ix_dirname = args.ix_dir
path_input_file = args.path_input_cleaned_corpus

# assert False, 'index finished, no need to run again'

os.system('rm -rf ' + ix_dirname)
if not os.path.isdir(ix_dirname):
    print('indexing')
    os.mkdir(ix_dirname)
    ix = create_in(ix_dirname, schema)
    writer = ix.writer()
    with open(path_input_file) as rf:
        for line in tqdm(list(rf.readlines())):
            doc = json.loads(line)
            if not doc['article']:
                continue
            writer.add_document(
                url=doc['url'],
                date=doc['date'],
                title=doc['title'],
                article=doc['article'])
    writer.commit()
    
    
    
import json
from datetime import datetime
import whoosh.index as index
from whoosh.qparser import QueryParser
from whoosh.qparser import MultifieldParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections

ix = index.open_dir(ix_dirname)

from datetime import timedelta


def normed(df):
    return (df-df.min())/(df.max()-df.min())

def getdt(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d")


twitter_df = pd.read_pickle(args.path_twitter_timeseries_pickle)


MIN_DATE = getdt(args.start_date)
MAX_DATE = getdt(args.end_date)

frame2keywords = {
    'covid': 'covid OR covid19 OR coronavirus OR corona',
    'debt': 'china OR chinese tanzania loan OR loans',
    'environmentalism': 'china OR chinese Zimbabwe OR Nigeria "mining" OR "mine"',
    'infrastructure': 'china africa OR tanzania OR kenya OR Zimbabwe SGR',  # can improve
    'mistreatment': 'mistreatment',
    'prejudice': 'prejudice OR discrimination OR xenophobia OR racist OR racism',
    'travel': 'travel OR flights OR travelers OR traveler',
    'un': '"united nations" OR "un secretary" OR un'
    # 'covid/assistance': 'china africa covid donate OR dontation OR help OR assistance',
    # 'trade': 'china africa OR kenya trade OR imports OR exports',
}
# frame2keywords = {
#     'debt': 'china OR chinese tanzania loan OR loans',
#     'infrastructure': 'china africa OR tanzania OR kenya OR Zimbabwe SGR',  # can improve
# }


url2frame2prob = collections.defaultdict(dict)


dir_retrieved_docs = args.dir_retrieved_docs
os.makedirs(dir_retrieved_docs, exist_ok=True)

def draw(frame):
    dates = [0 for _ in range((MAX_DATE - MIN_DATE).days + 1)]
    docs = []
    with ix.searcher() as searcher:
        query = MultifieldParser(["article", "title"], ix.schema).parse(frame2keywords[frame])
        print(query)
        results = searcher.search(query, limit=None)
        print(len(results))
        for result in results[:2000]:
            date = getdt(result['date'])
            if MIN_DATE <= date <= MAX_DATE:
                dates[(date - MIN_DATE).days] += 1
                docs.append(dict(result))
                url2frame2prob[result['url']][frame] = 1

    framename = frame.replace('/', '-')
    with open(f'{dir_retrieved_docs}/docs_{framename}.json', 'w') as wf:
        json.dump(docs, wf, indent=4)

    dti = pd.to_datetime([ str((MIN_DATE+timedelta(days=delta)).date()) for delta in range(0, (MAX_DATE-MIN_DATE).days + 1)])

    df = pd.DataFrame(dates, columns=[''])
    df.index = dti
    sns.lineplot(data=normed(df), label="retrieval")
    
    # if frame == 'assistance':
    #     if "covid/assistance" in twitter_df:
    #         sns.lineplot(data=normed(twitter_df["covid/assistance"]), label='twitter')
    # elif frame in twitter_df:
    if frame in twitter_df:
        sns.lineplot(data=normed(twitter_df[frame]), label='twitter')
    
    plt.xticks(rotation=70)
    plt.show()
    
for frame in frame2keywords.keys():
    print(frame)
    draw(frame)
    

for url, frame2prob in url2frame2prob.items():
    for frame in frame2keywords.keys():
        if frame not in frame2prob:
            url2frame2prob[url][frame] = 0
with open(Path(__file__).parent.joinpath('..', f'retrieved_url2frame2prob_{ix_name}.json'), 'w') as wf:
    json.dump(url2frame2prob, wf, indent=4)
