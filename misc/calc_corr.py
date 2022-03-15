import argparse
import dateutil.parser
import json
from collections import OrderedDict
import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import pandas as pd

def getCorr(a, akeys, b, idx, offset=0):
    corr = {}
    for na in akeys:
        z = np.array(a[na].reindex(idx, fill_value=0).EventCount.to_list())
        if np.linalg.norm(z) > 0:
            z = z / np.linalg.norm(z)
        tmp = []
        for nb in b.keys():
            x = np.pad(np.array(b[nb].reindex(idx, fill_value=0).to_list()), offset, mode='constant')
            tmp.append((nb, pearsonr(x[:len(z)], z)[0]))
        tmp.sort(key=lambda x: -x[1])
        corr[na] = OrderedDict(tmp)
    return corr
    
def main(args):
    logging.info(f'Reading node file')
    with open(args.nodes, 'r') as f:
        nodes = f.read().strip().split('\n')
    idx = pd.date_range(pd.to_datetime(args.startdate), pd.to_datetime(args.enddate) - pd.Timedelta(days=1))

    with open(args.exo_input, 'r') as f:
        edf = {k: pd.read_json(v, typ='series').resample(args.granularity).sum().reindex(idx, fill_value=0) for k, v in json.load(f).items()}
    with open(args.platform_input, 'r') as f:
        pdf = {k: pd.read_json(v, orient='columns').resample(args.granularity).sum().reindex(idx, fill_value=0) for k, v in json.load(f).items()}
    with open(args.out, 'w') as f:
        json.dump(getCorr(pdf, nodes, edf, idx, args.offset), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract timeseries and statistics from extracted groundtruth data.')
    parser.add_argument('-pi', '--platform_input', required=True, type=str, help='Path to the platform timeseries (.json)')
    parser.add_argument('-ei', '--exo_input', required=True, type=str, help='Path to the exogenous timeseries (.json)')
    parser.add_argument('-d', '--offset', required=False, type=int, help='How many days to shift the exogenous data', default=0)
    parser.add_argument('-n', '--nodes', required=True, type=str, help='Path to the node file (.txt)')
    parser.add_argument('-g', '--granularity', default='D', type=str, choices=['D', 'W'], help='Activity counting granularity (D or W)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the correlation results (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
