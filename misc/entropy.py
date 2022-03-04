from collections import OrderedDict, defaultdict, deque, Counter
import gzip
import json
from math import sqrt
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def calcEntropy(path, nodepath, start_date, end_date, out_path):
    logging.info(f'Reading node file')
    with open(nodepath, 'r') as f:
        twt_nodes = f.read().strip().split('\n')
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1))
    
    logging.info(f'Reading news articles')
    json_data = []
    with open(path, 'r') as f:
        for l in f:
            tmp = json.loads(l)
            try:
                if pd.to_datetime(tmp['date']) >= pd.to_datetime(start_date) and pd.to_datetime(tmp['date']) < pd.to_datetime(end_date):
                    json_data.append(tmp)
            except:
                pass

    logging.info(f'Counting')
    articles = defaultdict(lambda: defaultdict(dict))
    for r in tqdm(json_data):
        if r['Leidos_infoids'] is not None and r['phrased_article'] is not None and r['phrased_article'] != '':
            for infoid in r['Leidos_infoids']:
                if r['phrased_title'] not in articles[infoid][pd.to_datetime(r['date']).date()]:
                    articles[infoid][pd.to_datetime(r['date']).date()][r['phrased_title']] = [1, r['phrased_article']]
                else:
                    articles[infoid][pd.to_datetime(r['date']).date()][r['phrased_title']][0] += 1

    logging.info(f'Calculating frequencies')
    tokens = defaultdict(dict)
    for k, v in articles.items():
        for kk, vv in tqdm(v.items(), total=len(v)):
            tokens[k][kk] = word_tokenize(' '.join([' '.join([vvv[1]] * int(sqrt(vvv[0]))) for vvv in vv.values()]))

    stop_words = set(stopwords.words('english') + pd.read_csv('news_stop.csv').term.to_list()) 
    words = set(nltk.corpus.words.words())

    ctss = defaultdict(dict)
    for k, v in tokens.items():
        for kk, vv in tqdm(v.items(), total=len(v)):
            ctss[k][kk] = nltk.FreqDist([vvv.lower() for vvv in vv if len(vvv) > 3 and vvv.lower() not in stop_words and vvv.lower() in words])

    def recifunc(x, s, c):
        return (1 / np.power(x, s)) / np.sum([1 / np.power(i, s) for i in range(1, 101)]) + c

    logging.info(f'Fitting Zipf curve')
    etps2 = defaultdict(dict)
    for k1, v1 in ctss.items():
        for k, v in sorted(v1.items()):
            try:
                tmp = v.most_common(100)
                s = sum([x[1] for x in tmp])
                ps0 = [x[1] for x in tmp]
                ps = [x[1] / s for x in tmp]
                popt, pcov = curve_fit(recifunc, np.linspace(1, len(ps), len(ps)), ps)
                etps2[k1][k] = popt[0]
            except:
                print(k1, k, tmp)

    for k, v in etps2.items():
        if len(v) == 0:
            logging.info(f'"{k}" is empty')

    etps2_df = {}
    for k, v in etps2.items():
        tmpmean = np.mean(list(v.values()))
        tmp = pd.Series(v).reindex(idx, fill_value=tmpmean)
        tmp[tmp < 1e-3] = tmpmean
        etps2_df[k] =tmp

    for k in set(twt_nodes) - set(etps2_df.keys()):
        etps2_df[k] = pd.Series().reindex(idx, fill_value=1)

    logging.info(f'Writing to file')
    etps2_df_json = {k: v.to_json() for k, v in etps2_df.items()}
    with open(out_path, 'w') as f:
        f.write(json.dumps(etps2_df_json))
    
def main(args):
    calcEntropy(args.path, args.nodes, args.startdate, args.enddate, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Zipf timeseries from annotated news articles.')
    parser.add_argument('-i', '--path', required=True, type=str, help='Path to the annotated news articles (.json)')
    parser.add_argument('-n', '--nodes', required=True, type=str, help='Path to the node file (.txt)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the Zipf timeseries (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
