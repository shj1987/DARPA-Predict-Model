import os, sys
import argparse
import json
from sklearn import linear_model
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from datetime import timedelta
from math import floor
from collections import OrderedDict


def train(narratives, main_data, leidos):
    p = 0
    narrative_coef = {}
    train_num = len(main_data[narratives[0]])
    for na in narratives:
        current_main = main_data[na][:train_num].to_numpy()
        current_x = leidos[na].to_list()
        data_y_event = []
        data_y_user = []
        data_y_newuser = []
        data_x = []
        for k in range(p, len(current_main)):
            y = current_main[k]  # data on kth day of narrative i
            data_y_event.append(y[0])
            data_y_user.append(y[1])
            data_y_newuser.append(y[2])
            x = current_x[k - p:k + 1]
            data_x.append(x)

        ar_coef = []
        regr = linear_model.LinearRegression()
        regr.fit(np.array(data_x), np.array(data_y_event))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)  # append intercept
        ar_coef.append(coef)
        regr.fit(np.array(data_x), np.array(data_y_user))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)
        ar_coef.append(coef)
        regr.fit(np.array(data_x), np.array(data_y_newuser))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)
        ar_coef.append(coef)
        narrative_coef[na] = ar_coef
    
    return narrative_coef

def main(argv):
    parser = argparse.ArgumentParser(description='LR PLUS TC model.')
    parser.add_argument('-m', '--main_file', required=True, type=str)
    parser.add_argument('-g', '--tc_file', required=True, type=str)
    parser.add_argument('--nodes', required=True, type=str, help='Path to the node file (.txt)')
    # parser.add_argument('-p', '--platform', required=True, type=str)
    parser.add_argument('-n', '--num_days_test', required=True, type=int)
    parser.add_argument('-o', '--output_file', required=True, type=str)
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    
    with open(args.main_file, 'r') as f:
        d = json.loads(f.read())
    main_data = {k: pd.read_json(v, orient='columns')for k, v in d.items()}

    with open(args.tc_file, 'r') as f:
        leidos_ts = json.loads(f.read())
    leidos = {k: pd.read_json(v, typ='series') for k, v in leidos_ts.items()}

    with open(args.nodes, 'r') as f:
        narratives = f.read().strip().split('\n')

    train_num = len(main_data[narratives[0]])
    test_num = args.num_days_test
    test_idx = leidos[narratives[0]][train_num : train_num + test_num].index

    columns_name = main_data[narratives[0]].columns.values
    
    narrative_coef = train(narratives, main_data, leidos)

    rst_dict = {}
    p = 0
    for na in narratives:
        re = []
        coef = narrative_coef[na]
        current_x = leidos[na][train_num - p: train_num + test_num].to_list()
        for d in range(test_num):
            leidos_predict = current_x[d : d + p + 1]
            y_hat_event = np.dot(coef[0][:-1], leidos_predict) + coef[0][-1]
            y_hat_user = np.dot(coef[1][:-1], leidos_predict) + coef[1][-1]
            y_hat_newuser = np.dot(coef[2][:-1], leidos_predict) + coef[2][-1]
            y_hat_newuser = max(y_hat_newuser, 1)
            y_hat_user = max(y_hat_user, y_hat_newuser, 1)
            y_hat_event = max(y_hat_event, y_hat_user, y_hat_newuser, 1)

            re.append([int(round(y_hat_event)), int(round(y_hat_user)), int(round(y_hat_newuser))])
        
        re = np.array(re)
        rst_dict[na] = pd.DataFrame(re, columns=columns_name, index=test_idx).to_json()

    with open(args.output_file, 'w') as out:
        json.dump(rst_dict, out)
        print('Done!')
    
if __name__ == "__main__":
    main(sys.argv[1:])