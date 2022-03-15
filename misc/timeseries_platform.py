import argparse
import dateutil.parser
import pathlib
import json
from collections import defaultdict
import logging
import pickle as pkl

import numpy as np
import pandas as pd
import gzip
from tqdm import tqdm

def extractTimeseries(paths, nodepath, platform, start_date, end_date, out_path, stat_out_path):
    logging.info(f'Reading node file')
    with open(nodepath, 'r') as f:
        twt_nodes = f.read().strip().split('\n')
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1))
    
    tdict_count = defaultdict(lambda: defaultdict(int))
    tdict_userset = defaultdict(lambda: defaultdict(set))
    tdict_activateduserset = defaultdict(lambda: defaultdict(set))
    tdict_timedist = defaultdict(lambda: defaultdict(lambda: [0] * 24))
    tdict_lambda = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    t_userset = defaultdict(set)
    for datapath in paths:
        logging.info(f'Reading {datapath}')
        if pathlib.Path(datapath).suffix == '.gz':
            f = gzip.open(datapath, 'rt')
        elif pathlib.Path(datapath).suffix == '.json':
            f = open(datapath, 'r')
        else:
            raise ValueError(f'Unknown file type: {datapath}')
        for line in f:
            tmp = json.loads(line)
            if tmp['platform'] != platform:
                continue
            if pd.to_datetime(tmp['nodeTime']).tz_localize(None) < pd.to_datetime(start_date).tz_localize(None):
                continue
            if pd.to_datetime(tmp['nodeTime']).tz_localize(None) >= pd.to_datetime(end_date).tz_localize(None):
                continue
            date = pd.to_datetime(tmp['nodeTime']).date()
            tt = pd.to_datetime(tmp['nodeTime']).time()
            user = tmp['nodeUserID']
            infoId = tmp['informationID']

            tdict_count[infoId][date] += 1
            tdict_userset[infoId][date].add(user)
            tdict_timedist[infoId][date][tt.hour] += 1
            tdict_lambda[infoId][tt.hour][date].append(tt.hour * 3600 + tt.minute * 60 + tt.second)
            if user not in t_userset[infoId]:
                tdict_activateduserset[infoId][date].add(user)
            t_userset[infoId].add(user)
        f.close()
    
    for k in set(twt_nodes) - set(tdict_count.keys()):
        logging.info(f'"{k}" does not exist in the dataset, ignoring')
        twt_nodes.remove(k)

    logging.info('Calculating parameters')
    tdict_lambda_whole = defaultdict(lambda: defaultdict(list))
    for node in twt_nodes:
        for h in range(24):
            for d, l in tdict_lambda[node][h].items():
                tdict_lambda_whole[node][d].extend(l)
    tdict_lambda_whole = {k: 
                          np.median(np.concatenate(list(map(np.diff, map(sorted, filter(lambda x: len(x) > 1, v.values())))))) \
                          for k, v in tdict_lambda_whole.items()}
    for node in twt_nodes:
        for h in range(24):
            if len(list(filter(lambda x: len(x) > 1, tdict_lambda[node][h].values()))) == 0 and -1 not in tdict_lambda[node][h]:
                tdict_lambda[node][h][-1].extend([0, tdict_lambda_whole[node]])
    tlambda = {}
    for k, v in tdict_lambda.items():
        tlambda[k] = [100] * 24
        for h in range(24):
            lst = list(map(np.diff, map(sorted, filter(lambda x: len(x) > 1, v[h].values()))))
            if len(lst) > 0:
                tlambda[k][h] = np.concatenate(lst).mean()

    logging.info('Counting events and users')
    tdict_usercount = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in tdict_userset.items()}
    tdict_activateduser = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in tdict_activateduserset.items()}

    tmerge = defaultdict(dict)
    for k, v in tdict_count.items():
        for kk, vv in v.items():
            if kk not in tdict_activateduser[k]:
                tdict_activateduser[k][kk] = 0
            tmerge[k][kk] = (vv, tdict_usercount[k][kk], tdict_activateduser[k][kk])

    tmerge_df = {
        k: pd.DataFrame.from_dict(v, orient='index', columns=['EventCount', 'UserCount', 'NewUserCount']).reindex(idx, fill_value=0) \
        for k, v in tmerge.items()
    }

    logging.info('Writing timeseries')
    tmerge_df_json = {k: v.to_json() for k, v in tmerge_df.items()}
    out_path = pathlib.Path(out_path)
    with open(out_path.parent.joinpath(f'{out_path.stem}.json'), 'w') as f:
        f.write(json.dumps(tmerge_df_json))
    pd.to_pickle({k: v.EventCount for k, v in tmerge_df.items()}, out_path.parent.joinpath(f'{out_path.stem}.pkl'))

    logging.info('Counting distributions')
    tdist = defaultdict(lambda: [[], [], [], []])
    for node in twt_nodes:
        if node not in tmerge_df:
            continue
        tmp = max(tmerge_df[node].EventCount)
        for date in tmerge_df[node].index:
            cur = tmerge_df[node].loc[date].EventCount
            if cur < tmp / 4:
                tdist[node][0].append(tdict_timedist[node][date.date()])
            elif cur < tmp * 2 / 4:
                tdist[node][1].append(tdict_timedist[node][date.date()])
            elif cur < tmp * 3 / 4:
                tdist[node][2].append(tdict_timedist[node][date.date()])
            else:
                tdist[node][3].append(tdict_timedist[node][date.date()])
        for kk in range(4):
            if len(tdist[node][kk]) == 0:
                tdist[node][kk].append([1] * 24)
    tdist = {k: [np.sum(v[0], axis=0), np.sum(v[1], axis=0), np.sum(v[2], axis=0), np.sum(v[3], axis=0)] for k, v in tdist.items()}
    for k, v in tdist.items():
        for i in range(4):
            nonz = v[i].nonzero()[0]
            vv = v[i]
            copy = vv.tolist()
            for ii in np.where(v[i] == 0)[0]:
                copy[ii] = (np.exp(-0.7 * np.abs(nonz - ii)) * vv[nonz]).sum()
            tdist[k][i] = copy

    tprob = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    maxes = {}
    for node in twt_nodes:
        tmp = max(tmerge_df[node].EventCount)
        maxes[node] = tmp
        for date in tmerge_df[node].index:
            cur = tmerge_df[node].loc[date].EventCount
            if cur < tmp / 4:
                for u in tdict_userset[node][date.date()]:
                    tprob[node][u][0] += 1
            elif cur < tmp * 2 / 4:
                for u in tdict_userset[node][date.date()]:
                    tprob[node][u][1] += 1
            elif cur < tmp * 3 / 4:
                for u in tdict_userset[node][date.date()]:
                    tprob[node][u][2] += 1
            else:
                for u in tdict_userset[node][date.date()]:
                    tprob[node][u][3] += 1

    logging.info('Writing statistics')
    with open(stat_out_path, 'w') as f:
        json.dump({'span': len(idx), 'max': maxes, 'prob': tprob, 'dist': tdist, 'lambda': tlambda}, f)

def main(args):
    extractTimeseries(args.paths, args.nodes, args.platform, args.startdate, args.enddate, args.out, args.statout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract timeseries and statistics from extracted groundtruth data.')
    parser.add_argument('-i', '--paths', required=True, type=str, nargs='+', help='Path to the extracted groundtruth file(s) (.json|.gz)')
    parser.add_argument('-n', '--nodes', required=True, type=str, help='Path to the node file (.txt)')
    parser.add_argument('-p', '--platform', required=True, type=str, help='Which platform to extract (twitter|youtube|reddit|jamii)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the extract timeseries (.json)')
    parser.add_argument('--statout', required=True, type=str, help='Path to save the extract statistics (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
