#!/usr/bin/env python
# coding: utf-8
import argparse
import dateutil.parser
import pathlib
import sys
from math import ceil
import json
from collections import defaultdict
import random
import logging
import pickle
# import datetime
# import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from load import load_data

pd.options.mode.chained_assignment = None

columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime', 'platform', 'rootUserID', 'parentUserID', 'informationID']

def networkFilling(history, nodes, preserve_user_nodes, pred, prob, idx):
#     for infoID in nodeList:
    def inner(infoID):
        content = pred[infoID][['EventCount', 'UserCount', 'NewUserCount']].astype(int)
        totalEvent = sum(content.EventCount)
        print(infoID, 'totalEvent', totalEvent)
        if totalEvent == 0:
            return pd.DataFrame(columns=columns)
        
        duplicate = history[history.informationID == infoID]
        if len(duplicate) == 0:
            return pd.DataFrame(columns=columns)
        active_level = pd.DataFrame(prob['prob'][infoID]).T.reset_index().replace(0, np.inf)
        
        print(infoID, len(duplicate), int(ceil(totalEvent / len(duplicate))))
        def mod(df, cnt):
            tmp = df.copy()
            tmp.nodeID = tmp.nodeID.apply(lambda x: x + '_' + str(cnt))
            tmp.parentID = tmp.parentID.apply(lambda x: x if x == '?' else x + '_' + str(cnt))
            tmp.rootID = tmp.rootID.apply(lambda x: x if x == '?' else x + '_' + str(cnt))
            if infoID not in preserve_user_nodes:
                tmp.nodeUserID = tmp.nodeUserID.apply(lambda x: x + '_' + str(cnt))
                tmp.parentUserID = tmp.parentUserID.apply(lambda x: x if x == '?' else x + '_' + str(cnt))
                tmp.rootUserID = tmp.rootUserID.apply(lambda x: x if x == '?' else x + '_' + str(cnt))
            return tmp
        copies = [mod(duplicate, i) for i in range(1, int(ceil(totalEvent / len(duplicate))))]
        duplicate = pd.concat(list(reversed(copies)) + [duplicate], axis=0)
        duplicate = duplicate.iloc[-totalEvent:]

        counter = 0
        maxEvent = prob['max'][infoID]
        sortCols = {}
        # history data
        hist_data = history[history.informationID == infoID]
        lam = prob['lambda'][infoID]
        #
        for index, (timestamp, (volume, _, newUser)) in enumerate(content.iterrows()):
            if volume < maxEvent / 4:
                sortCols[timestamp] = 'c0'
                dist = prob['dist'][infoID][0]
            elif volume < maxEvent * 2 / 4:
                sortCols[timestamp] = 'c1'
                dist = prob['dist'][infoID][1]
            elif volume < maxEvent * 3 / 4:
                sortCols[timestamp] = 'c2'
                dist = prob['dist'][infoID][2]
            else:
                sortCols[timestamp] = 'c3'
                dist = prob['dist'][infoID][3]
                
            print (index, infoID, timestamp, volume, sortCols[timestamp], counter, counter + volume)
            if volume == 0: continue
            
            # NodeTime Shift
            #duplicate.nodeTime.iloc[counter: counter + volume] = list(pd.Series(pd.date_range(timestamp + pd.Timedelta(seconds=1), timestamp + pd.Timedelta(hours=23, minutes=59, seconds=59), periods=volume)).astype('datetime64[s]'))
            dist = np.array(dist)
            st_dist = dist / dist.sum()
            shuffle = np.random.choice(24, volume, p=st_dist)
            shuffle_volume = pd.Series(shuffle).value_counts().sort_index().reindex(range(24), fill_value=0)
            time_after_process= []
            diffs = []
            for i in range(24):
                if shuffle_volume[i] == 0:
                    continue
                diffs.extend(np.random.exponential(lam[i], shuffle_volume[i]).tolist())
            diffs = np.array(diffs)
            diffs = ((diffs / diffs.sum()) * (24 * 60 * 60 - 1)).cumsum()
            duplicate.nodeTime.iloc[counter: counter + volume] = list(pd.Series([timestamp + pd.Timedelta(seconds=round(dt)) for dt in diffs]).astype('datetime64[s]'))
            
            # NodeID Shift
            duplicate.nodeID.iloc[counter: counter + volume] = duplicate.nodeID.iloc[counter: counter + volume].apply(lambda x: x + '_')
            duplicate.parentID.iloc[counter: counter + volume] = duplicate.parentID.iloc[counter: counter + volume].apply(lambda x: x if x == '?' else x + '_')
            duplicate.rootID.iloc[counter: counter + volume] = duplicate.rootID.iloc[counter: counter + volume].apply(lambda x: x if x == '?' else x + '_')

            counter += volume
        
        if infoID not in preserve_user_nodes:
            involved_users = []
            grouped_data = duplicate[['nodeID', 'nodeUserID', 'nodeTime']].set_index('nodeTime').groupby([pd.Grouper(freq='1D'), 'nodeUserID'])\
                .nodeID.nunique().to_frame().reset_index().sort_values(['nodeTime', 'nodeID'], ascending=[True, True])\
                .drop_duplicates('nodeUserID')
            grouped_data['_nodeUserID'] = grouped_data.nodeUserID.apply(lambda x: x[:22])
            active_level.columns = ['_nodeUserID', 'c0', 'c1', 'c2', 'c3']
            grouped_data = grouped_data.merge(active_level, how='left', on='_nodeUserID').set_index('nodeTime').groupby([pd.Grouper(freq='1D')])

            for name, group in grouped_data:
                #if name not in data[list(data.keys())[0]].index:
                #    break
                group = group.sort_values(['nodeID', sortCols[name]])
                involved_users.extend(list(group.head(content.loc[name].NewUserCount).nodeUserID.values))

            involved_users = set(involved_users)
            duplicate.nodeUserID = duplicate.nodeUserID.apply(lambda x: x + '_' if x in involved_users else x)
            duplicate.parentUserID = duplicate.parentUserID.apply(lambda x: x if x == '?' else (x + '_' if x in involved_users else x))
            duplicate.rootUserID = duplicate.rootUserID.apply(lambda x: x if x == '?' else (x + '_' if x in involved_users else x))
        return duplicate

#     final = []
#     for infoID in nodes:
#         final.append(inner(infoID))
    final = Parallel(n_jobs=8, verbose=50)(delayed(inner)(infoID) for infoID in nodes)
    final = pd.concat(final).sort_values(['nodeTime', 'informationID']).reset_index(drop=True)
    return final


def check(data, df, nodes, idx):
    for infoID in nodes:
        if data[infoID].EventCount.sum() == 0:
            print('Pass', infoID)
            continue
        print(np.all(data[infoID].EventCount == df[df.informationID == infoID].nodeTime.value_counts().resample('D').sum().reindex(idx, fill_value=0)), infoID)


def fill(predictions, nodes, pnodes, probs, gts, train_start_date, train_end_date, pred_start_date, pred_end_date, team_name, model_name, sim_period, out_path):
    idx = pd.date_range(pd.to_datetime(pred_start_date), pd.to_datetime(pred_end_date) - pd.Timedelta(days=1))
    results = []
    for p, n, pn, pb, gt in zip(predictions, nodes, pnodes, probs, gts):
        logging.info(f'Working on {p}')
        logging.info(f'Reading history')
        history_data = pd.concat([
            load_data(fn, ignore_first_line=False, verbose=False)[columns] \
            for fn in gt.split(';')
        ], ignore_index=True).sort_values(['nodeTime', 'nodeID']).reset_index(drop=True)
        history_data = history_data[~history_data.nodeUserID.isnull()]
        history_data.loc[history_data.parentUserID.isnull(), 'parentUserID'] = '?'
        history_data.loc[history_data.rootUserID.isnull(), 'rootUserID'] = '?'
        history_data = history_data[(history_data.nodeTime >= pd.to_datetime(train_start_date)) & (history_data.nodeTime < pd.to_datetime(train_end_date))]
        
        logging.info(f'Reading node file')
        with open(n, 'r') as f:
            nodes = f.read().strip().split('\n')
        with open(pn, 'r') as f:
            preserve_user_nodes = set(f.read().strip().split('\n'))
            
        logging.info(f'Reading statistics file')
        with open(pb, 'r') as f:
            prob = json.load(f)
            
        logging.info(f'Reading prediction file')
        with open(p, 'r') as f:
            d = json.loads(f.read())
        pred = {k: pd.read_json(v, orient='columns').reindex(idx, fill_value=0) for k, v in d.items()}

        df = networkFilling(history_data, nodes, preserve_user_nodes, pred, prob, idx)
        check(pred, df, nodes, idx)
        results.append(df)
    final = pd.concat(results, axis=0).sort_values(by='nodeTime').reset_index(drop=True)
    final.nodeTime = final.nodeTime.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ'))
    
    identifier = {"team": team_name, "model_identifier": model_name, "simulation_period": sim_period}
    with open(out_path, 'w') as f:
        f.write(json.dumps(identifier) + '\n')
        for row in tqdm(final.iterrows(), total=final.shape[0]):
            f.write(row[1].to_json() + '\n')

def main(args):
    assert len(args.predictions) == len(args.nodes)
    assert len(args.predictions) == len(args.preserve_user_nodes)
    assert len(args.predictions) == len(args.groundtruths)
    assert len(args.predictions) == len(args.probs)
    fill(args.predictions, args.nodes, args.preserve_user_nodes, args.probs, args.groundtruths, \
        args.train_startdate, args.train_enddate, args.pred_startdate, args.pred_enddate, \
        args.team_name, args.model_name, args.simulation_period, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network fill the messages from number prediction using old data.')
    parser.add_argument('-i', '--predictions', required=True, type=str, nargs='+', help='List of paths to prediction files (.json)')
    parser.add_argument('-n', '--nodes', required=True, type=str, nargs='+', help='List of paths to the node files (.txt)')
    parser.add_argument('-pn', '--preserve_user_nodes', required=True, type=str, nargs='+', help='List of paths to the node files that contains frames that need to preserve user names (.txt)')
    parser.add_argument('-p', '--probs', required=True, type=str, nargs='+', help='List of paths to statistics files (.json)')
    parser.add_argument('-g', '--groundtruths', required=True, type=str, nargs='+', help='List groundtruth files (.json), separate files of the same platform by ;')
    parser.add_argument('-t', '--team_name', required=True, type=str, help='Team name')
    parser.add_argument('-m', '--model_name', required=True, type=str, help='Model name')
    parser.add_argument('-sp', '--simulation_period', required=True, type=str, help='Simulation period string')
    parser.add_argument('-ts', '--train_startdate', required=True, type=dateutil.parser.isoparse, help='The Training Start Date (format YYYY-MM-DD)')
    parser.add_argument('-te', '--train_enddate', required=True, type=dateutil.parser.isoparse, help='The Training End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-ps', '--pred_startdate', required=True, type=dateutil.parser.isoparse, help='The Prediction Start Date (format YYYY-MM-DD)')
    parser.add_argument('-pe', '--pred_enddate', required=True, type=dateutil.parser.isoparse, help='The Prediction End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the filled output (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
