from scipy import optimize, integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import time
import random
import math

from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression,Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

import copy
from math import log

import collections
 
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

from utils import *
from arg_parser import parse_args


def run_model_cp6(X_train, X_test, Y_train, Y_test, platform):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    rmse, ape, size, size_con = 0, 0, 0, 0
    train_size_per_frame = len(X_train)

    regression = LinearRegression(fit_intercept=False, normalize=True)
    regression.fit(X_train, Y_train)
    pred = postprocess(regression.predict(X_test))

    if sum(pred) == 0:
        print('1 frame all zeros')
        if (platform == 'youtube'):
            pred = [1] + [0 for _ in range(len(Y_test) - 1)]
        else:
            pred = [1] + [1 for _ in range(len(Y_test) - 1)]
        print('GT is ', sum(Y_test))

    rmse, ape = evaluation(Y_test, pred)
    size += sum(Y_test)
    size_con += sum(pred)
    return rmse, ape, size, size_con, pred


# Used to apply post-processing to generated prediction
def postprocess_cp6(pred):
    pred = np.array([np.round(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred


# Used to compare the prediction and the ground truth by calculating RMSE and APE
def evaluation_cp6(Y_test, Y_pred):
    if sum(Y_test) == 0:
        return 0, 0
    rmse = np.sqrt(mse(np.array(Y_test).cumsum() / (sum(Y_test)), np.array(Y_pred).cumsum() / (sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / (sum(Y_test))
    return rmse, ape

def main_CP6(args):
    platform = args.platform
    input_source = args.cp6_input_source

    data_path = args.cp6_data_path
    date = args.cp6_date
    data = load_global_data_cp6(platform, input_source, data_path, date)
    twitter, frames, gdelt, ent, corr = data
    dev_test_len = args.cp6_dev_test_len

    # get optimal k for using top k corrs for training
    most_corr_N = {}
    abnormal_use_last_X = {}
    for idx in trange(len(frames)):
        frame = frames[idx]
        nuc_apes = []
        nuc_totals = []
        x_axis = range(5, 100, 5)
        for i in x_axis:
            X_train, X_test, Y_train, Y_test = dataloader_cp6(twitter, frame, gdelt, corr, i, ent, 'NewUserCount',
                                                          366 - dev_test_len - dev_test_len, dev_test_len, False,
                                                          'method_1', abnormal_use_last_X)
            nuc_rmse, nuc_ape, nuc_size, nuc_size_con, nuc_pred = run_model_cp6(X_train, X_test, Y_train, Y_test, platform)
            nuc_apes.append(nuc_ape)
            nuc_totals.append(nuc_ape + nuc_rmse)
        min_ape = min(nuc_apes)
        min_x = x_axis[nuc_apes.index(min_ape)]
        most_corr_N[frame] = min_x

    abnormal_use_last_X = {}
    x_axis = range(1, 300, 5)
    for idx in trange(len(frames)):
        frame = frames[idx]
        ape_result = []
        total_result = []
        for length in x_axis:
            X_train, X_test, Y_train, Y_test = dataloader_cp6(twitter, frame, gdelt, corr, most_corr_N[frame], ent,
                                                          'NewUserCount', 366 - dev_test_len - dev_test_len,
                                                          dev_test_len, False, 'method_1', {})
            rmse, ape, size, size_con, pred = run_model_cp6(X_train[-length:], X_test, Y_train[-length:], Y_test, platform)
            ape_result.append(ape)
            total_result.append(rmse + ape)
        min_total = min(total_result)
        npArray = np.array(total_result)
        occurance = np.where(npArray == min_total)
        min_length = x_axis[occurance[0][0]]
        abnormal_use_last_X[frame] = min_length

    file_name = args.cp6_file_name

    result = {name: {'EventCount': [], 'UserCount': [], 'NewUserCount': []} for name in frames}
    for option in ['EventCount', 'UserCount', 'NewUserCount']:
        for frame in frames:
            train_len = 366 - dev_test_len
            X_train, X_test, Y_train = dataloader_cp6(twitter, frame, gdelt, corr, most_corr_N[frame], ent, option,
                                                  366 - dev_test_len - dev_test_len, dev_test_len, True, 'method_1',
                                                  abnormal_use_last_X)
            regression = LinearRegression(fit_intercept=False, normalize=True)
            regression.fit(X_train, Y_train)
            pred = postprocess(regression.predict(X_test))
            if sum(pred) == 0:
                if (platform == 'youtube'):
                    pred = [1] + [0 for _ in range(len(Y_test) - 1)]
                else:
                    pred = [1] + [1 for _ in range(len(Y_test) - 1)]
            result[frame][option] = np.array(list(map(int, pred)))

    output_cp6(result, dev_test_len, file_name)

def main_CP5(args):

    # load data:
    # predicted frame:
    with open(args.cp5_eval_nodes,'r') as wf:
        eval_frame = [line.strip() for line in wf]
    
    with open(args.cp5_other_nodes,'r') as wf:
        other_frame = [line.strip() for line in wf]

    frame = eval_frame[:-1] + other_frame + ['empty']

    # hyperparameter:
    platform = args.platform # youtube
    twitter_shift = args.twitter_shift

    split = args.split
    pred_len = args.pred_len
    val_len = args.val_len
    debug = args.debug
    condition = args.condition
    entropy = args.entropy

    idx = pd.date_range(pd.to_datetime(args.start_date), pd.to_datetime(args.end_date))
    text_path = args.text_path
    data_path = args.data_path

    # social ground truth data:
    with open(data_path + platform + "_time_series_to_8_03.json") as f:
        twitter_file = json.loads(f.read())
        twitter = {k: pd.read_json(v, orient='columns') for k, v in twitter_file.items()}

    # input data:
    if platform == "twitter":
        with open(text_path + args.twitter_corr, 'r') as f:
            corr = json.loads(f.read())
            corr['empty'] = corr['controversies/china/border']
        with open(data_path + args.twitter_input,encoding="utf-8") as f:
            gdelt_file = json.loads(f.read())
            gdelt = {k: pd.read_json(v, typ='series').reindex(idx, fill_value=0) for k, v in gdelt_file.items()}
            gdelt['empty'] = gdelt['controversies/china/border']
        with open(args.entropy_input, 'r') as f:
            zipf_ts = json.load(f)
            zipf = OrderedDict(sorted({k: pd.read_json(v, typ='series').reindex(idx, fill_value=0) for k, v in zipf_ts.items()}.items(), key=lambda kv: kv[1].sum(), reverse=True))
            zipf['empty'] = zipf['controversies/china/border']
        if entropy:
            for item in gdelt:
                gdelt[item] = gdelt[item] * zipf[item]
        
    else:
        with open(data_path + args.youtube_corr, 'r') as f:
            corr = json.loads(f.read())
            corr['empty'] = corr['controversies/china/border']
        with open(data_path + args.youtube_input,encoding="utf-8") as f:
            gdelt_file = json.loads(f.read())
            gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}

    event = [key for key in corr['benefits/development/energy']]

    print("load %d frames, and %d external events" %(len([key for key in corr]), len(event)))
                
    if condition:
        nonlinear(gdelt, 0.3)
        #decay(gdelt, 0.99)
    
    result = dict()
    result_con =  dict()
    for item in frame:
        result[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}
        result_con[item] = {'EventCount':[], 'UserCount':[], 'NewUserCount':[]}

    if debug:
        mode = ['EventCount']
    else:
        mode = ['EventCount', 'UserCount', 'NewUserCount']
    for option in mode:
    
        threshold = [6,0.0]
        print(gdelt)
        X_train, X_test, X_pred, Y_train, Y_test, sample_weight, _ = dataloader(frame, twitter, gdelt, threshold[0], threshold[1], option, split - val_len, val_len, pred_len)
        X_train_con, X_test_con, X_pred_con, Y_train_con, Y_test_con, sample_weight, Y_con = dataloader(frame, twitter, gdelt, threshold[0], threshold[1], option, split, val_len, pred_len)

        regression = Lasso(fit_intercept=False,normalize=False, positive = True)
        regression.fit(X_train, Y_train)
        Y_pred = postprocess(regression.predict(X_test))
    
        regression = LinearRegression()
        regression.fit(X_train_con, Y_train_con)
        Y_con = postprocess(regression.predict(X_pred_con))
    
        threshold = [20,0.0]
        X_train, X_test, X_pred, Y_train, Y_test, sample_weight, _ = dataloader(frame, twitter, gdelt, threshold[0], threshold[1], option, split - val_len, val_len, pred_len)
        X_train_con, X_test_con, X_pred_con, Y_train_con, Y_test_con, sample_weight, _ = dataloader(frame, twitter, gdelt, threshold[0], threshold[1], option, split, val_len, pred_len)
        #print(len(X_train), len(X_test), len(Y_train), len(Y_test))
        rmse, ape = 0, 0
        for index, name in enumerate(frame):
            val_size = split - val_len
            #regression = LinearRegression(fit_intercept=False,normalize=True)
            regression = Lasso(fit_intercept=False,normalize=False, positive = True)
            regression.fit(X_train[val_size * index: val_size * (index + 1)], Y_train[val_size * index: val_size * (index + 1)])
            pred = postprocess(regression.predict(X_test[val_len * index: val_len * (index + 1)]))
        
            regression = LinearRegression()
            #regression = Lasso(fit_intercept=False,normalize=False, positive = True)
            regression.fit(X_train_con[split * index: split * (index + 1)], Y_train_con[split * index: split * (index + 1)])
            con = postprocess(regression.predict(X_pred_con[pred_len * index: pred_len * (index + 1)]))
        
            ratio = sum(pred) / (sum(Y_pred[val_len * index: val_len * (index + 1)]) + 0.1)
            ratio_con = sum(con) / (sum(Y_con[pred_len * index: pred_len * (index + 1)]) + 0.1)
            result[name][option] = pred
            y = postprocess(ratio_con * np.array(Y_con[pred_len * index: pred_len * (index + 1)]))
            
            if sum(result_con[name][option]) == 0:
                result_con[name][option] = postprocess(con)
        
            if option == 'EventCount':
                gt = twitter[name].EventCount.tolist()[-val_len:]
            elif option == 'UserCount':
                gt = twitter[name].UserCount.tolist()[-val_len:]
            else:
                gt = twitter[name].NewUserCount.tolist()[-val_len:]
            print(name, sum(result_con[name][option]), sum(gt))

        rmse, ape, size, size_con = 0, 0, 0, 0
        for index, name in enumerate(frame):
            if True:
                if sum(result[name][option]) == 0:
                    result[name][option] = [1 for item in range(val_len)]
                if sum(result_con[name][option]) == 0:
                    result_con[name][option] = [1 for item in range(pred_len)]
                if option == 'EventCount':
                    rmse_, ape_ = evaluation(twitter[name].EventCount.tolist()[-val_len:], result[name][option])
                elif option == 'UserCount':
                    ratio_N = 1. * sum(twitter[name].UserCount.tolist())/sum(twitter[name].EventCount.tolist())
                #rmse_, ape_ = evaluation(twitter[name].UserCount.tolist()[-val_len:], ratio_N * np.array(result_con[name]['EventCount']))
                    rmse_, ape_ = evaluation(twitter[name].UserCount.tolist()[-val_len:], result[name][option])
                else:
                    ratio_N = 1. * sum(twitter[name].NewUserCount.tolist()[:28])/(sum(twitter[name].EventCount.tolist()[:28]) + 0.1)
                #rmse_, ape_ = evaluation(twitter[name].NewUserCount.tolist()[-val_len:], ratio_N * np.array(result_con[name]['EventCount']))
                    rmse_, ape_ = evaluation(twitter[name].NewUserCount.tolist()[-val_len:], result[name][option])
                #print(name, rmse_, ape_), 
                size += sum(twitter[name][option].tolist()[-28:])
                size_con += sum(result_con[name][option])
                rmse += rmse_
                ape += ape_
                # if debug:
                #   draw_m2(name, result[name][option], result_con[name][option])
            
        print("RMSE: %f, APE: %f, Size: %f, Size_con: %f" %(rmse/len(frame), ape/len(frame), size, size_con))

    if not debug:
        output(args.start_date, result_con, result_con, args.output_dir + platform + '_' + args.output_file, split, pred_len)

def main():
    args = parse_args()
    if args.challenge == "CP5":
        main_CP5(args)
    else:
        main_CP6(args)

if __name__ == '__main__':
    main()

