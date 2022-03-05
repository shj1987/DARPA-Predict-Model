from scipy import optimize, integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import time
import random
import math
from tqdm import trange
from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression,Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import copy
from math import log


def load_global_data_cp6(platform, input_source, data_path, date):
    # Get twitter (platform data)
    with open(data_path + 'timeseries/cp6_{}_timeseries_to_{}.json'.format(platform, date)) as f:
        twitter_file = json.loads(f.read())
        twitter = {k: pd.read_json(v, orient='columns') for k, v in twitter_file.items()}

    # Get gdelt (input data)
    if (input_source == 'gdelt'):
        with open(data_path + "timeseries/cp6_{}_timeseries.json".format(input_source), encoding="utf-8") as f:
            gdelt_file = json.loads(f.read())
            gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}
    else:
        with open(data_path + "timeseries/cp6_newssrc_timeseries_to_01_10.json", encoding="utf-8") as f:
            gdelt_file = json.loads(f.read())
            gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}

    # Get entropy
    with open(data_path + 'timeseries/cp6_zipf_timeseries_to_01_10.json') as f:
        ent_file = json.loads(f.read())
        ent = {k: pd.read_json(v, typ='series') for k, v in ent_file.items()}

        # Get corr
    with open(data_path + 'inputs/cp6_{}_gdelt_corr_to_01_10.json'.format(platform)) as f:
        corr = json.loads(f.read())

    # Get frames
    frames = list(twitter.keys())
    events = list(gdelt.keys())

    # Output stats
    print('number of external events (from gdelt): ', len(gdelt))
    print('number of gdelt days: ', len(gdelt[events[0]]))
    print('number of twitter days: ', len(twitter[frames[0]]))

    return twitter, frames, gdelt, ent, corr


def dataloader_cp6(
        twitter,  # global time series
        frame,  # current frame, eg 'covid'
        gdelt,  # global gdelt
        corr,  # global corr
        corr_top_n,  # integer, top n
        ent,  # global entropy
        option,  # userCount, NewUser, EventCount
        train_len,  # the last x days of training data
        test_len,  # global
        is_eval,  # False if validation. True if testing
        method,  # IND
        abnormal_use_last_X  # dict: frame to last X
):
    X_train, X_test, Y_train, Y_test = [], [], [], []
    filted_gdelt = []  # filted_gdelt is the gdelt events that has the top-gdelt_threshold occurrence
    gdelt_corr_values = {event: corr[frame][event] for event in
                         gdelt}  # the sum of occurrence of each event across whole time span
    if is_eval:
        train_len += test_len
    filted_gdelt = sorted(list(gdelt_corr_values.keys()), key=lambda x: gdelt_corr_values[x], reverse=True)[
                   :corr_top_n]  # the top-k events with most occurrence, k=gdelt_threshold

    # Load training set, X_train, Y_train
    for i in range(0, train_len):
        # get Y_train, X_train
        if option == 'EventCount':
            Y_train.append(twitter[frame].EventCount.tolist()[i])
        elif option == 'UserCount':
            Y_train.append(twitter[frame].UserCount.tolist()[i])
        elif option == 'NewUserCount':
            Y_train.append(twitter[frame].NewUserCount.tolist()[i])
        else:
            raise Exception('Unknown option! Please use option \'EventCount\', \'UserCount\', or \'NewUserCount\'')

        X = []
        for event in filted_gdelt:
            current_X_train_value = gdelt[event].tolist()[i] * ent[frame][i]
            X.append(current_X_train_value)

        X_train.append(np.array(X))

    if frame in abnormal_use_last_X:
        X_train = X_train[-abnormal_use_last_X[frame]:]
        Y_train = Y_train[-abnormal_use_last_X[frame]:]

    # Testing, X_test, Y_test, similar to training
    for i in range(train_len, train_len + test_len):
        # get Y_test
        if (not is_eval):
            if option == 'EventCount':
                Y_test.append(twitter[frame].EventCount.tolist()[i])
            elif option == 'UserCount':
                Y_test.append(twitter[frame].UserCount.tolist()[i])
            elif option == 'NewUserCount':
                Y_test.append(twitter[frame].NewUserCount.tolist()[i])

        # Load X_test
        X = []
        for event in filted_gdelt:
            current_X_test_value = gdelt[event].tolist()[i] * ent[frame][i]
            X.append(current_X_test_value)
        X_test.append(X)

    # scale
    if frame in []:
        Y_train = [y * 0.3 for y in Y_train]

    if (is_eval):
        return X_train, X_test, Y_train
    return X_train, X_test, Y_train, Y_test

def dateGenerator_cp6(span):
    now = datetime.datetime(2020, 2, 1)
    delta = datetime.timedelta(days=1)
    endnow = now+datetime.timedelta(days=span)
    endnow = str(endnow.strftime('%Y-%m-%d'))
    offset = now

    Count = []
    while str(offset.strftime('%Y-%m-%d')) != endnow:
        tmp = int(time.mktime(offset.timetuple()) * 1000.0 + offset.microsecond / 1000.0)
        Count.append(str(offset.strftime('%Y-%m-%d')))
        offset += delta
    return Count


def output_cp6(result, dev_test_len, filename, referrence_file=None):
    order = ['EventCount', 'UserCount', 'NewUserCount']
    date = dateGenerator(400)

    out = dict()
    for name in result:
        data = np.array([result[name][item] for item in order]).T
        df = pd.DataFrame(data, columns=order, index=date[366 - dev_test_len:366])
        out[name] = df.to_json(orient="columns", force_ascii=False)

    # Get previous result
    if referrence_file != None:

        def convert(d):
            return np.array([d[_] for _ in date[366 - dev_test_len:366]])

        with open(referrence_file) as f:
            previous_result = json.loads(f.read())
            previous_frames = list(previous_result.keys())
        for name in previous_result:
            previous_result[name] = json.loads(previous_result[name])

        for name in previous_result:
            if name not in result:
                print(name)
                data = np.array([convert(previous_result[name][item]) for item in order]).T
                df = pd.DataFrame(data, columns=order, index=date[366 - dev_test_len:366])
                out[name] = df.to_json(orient="columns", force_ascii=False)

    with open(filename, 'w') as wf:
        json.dump(out, wf)
    with open(filename, 'r') as f:
        output_file = json.loads(f.read())
        out = {}
        for frame in output_file:
            df = pd.read_json(output_file[frame], orient='columns')
            df = df[order]
            out[frame] = df

    for key in out:
        print(key, sum(out[key].EventCount))
        print(out[key])


def dateGenerator(span):
    now = datetime.datetime(2020, 3, 30)
    delta = datetime.timedelta(days=1)
    endnow = now+datetime.timedelta(days=span)
    endnow = str(endnow.strftime('%Y-%m-%d'))
    offset = now

    Count = []
    while str(offset.strftime('%Y-%m-%d')) != endnow:
        tmp = int(time.mktime(offset.timetuple()) * 1000.0 + offset.microsecond / 1000.0)
        Count.append(str(offset.strftime('%Y-%m-%d')))
        offset += delta
    return Count

def postprocess(pred):
    pred = np.array([round(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred

def output(start_date, result, result_con, filename, split, pred_len):
    frames = [key for key in result]
    for item in result:
        for i in range(len(result[item]['UserCount'])):
            result[item]['UserCount'][i] = max(result[item]['NewUserCount'][i],result[item]['UserCount'][i])
            result[item]['EventCount'][i] = max(result[item]['EventCount'][i],result[item]['UserCount'][i])
    
    date = dateGenerator_output(start_date, 200)
    
    output = dict()
    for name in result:
        data = np.array([result_con[name][item] for item in result_con[name]]).T
        df = pd.DataFrame(data, columns = [item for item in result_con[name]], index = date[split:split + pred_len])
        output[name] = df.to_json(orient="columns",force_ascii=False)
    with open(filename ,'w') as wf:
        json.dump(output, wf)
    with open(filename ,'r') as f:
        output_file = json.loads(f.read())
        output = {k: pd.read_json(v, orient='columns') for k, v in output_file.items()}

    for key in output:
        print(key,sum(output[key].EventCount))
        print(output[key])

def dataloader(frame, twitter, gdelt, gdelt_cut, corr_cut, option, split, val_len, pred_len):
    # X_train shape: [T, N], T: time_step, N: input dimension
    X_train, X_test, X_pred, Y_train, Y_test, sample_weight, summation = [], [], [], [], [], [], []
    #print(filted_gdelt)
    size, size_gt = 0, 0
    for item in frame:

        for i in range(0,split + 1):
            if option == 'EventCount':
                Y_train.append(twitter[item].EventCount.tolist()[i])
                size_gt += twitter[item].EventCount.tolist()[i]
            elif option == 'UserCount':
                Y_train.append(twitter[item].UserCount.tolist()[i])
                size_gt += twitter[item].UserCount.tolist()[i]
            else:
                Y_train.append(twitter[item].NewUserCount.tolist()[i])
                size_gt += twitter[item].NewUserCount.tolist()[i]
            X_train.append(np.array([gdelt[item].tolist()[i]]))
                
        for i in range(split, split + val_len):
            if i >= len(twitter[item].EventCount.tolist()):
                continue
            if option == 'EventCount':
                Y_test.append(twitter[item].EventCount.tolist()[i])
                #size_gt += twitter[item].EventCount.tolist()[i]
            elif option == 'UserCount':
                Y_test.append(twitter[item].UserCount.tolist()[i])
            else:
                Y_test.append(twitter[item].NewUserCount.tolist()[i])
            X_test.append(np.array([gdelt[item].tolist()[i] ]))
        for i in range(split, split + pred_len):
            X_pred.append(np.array([gdelt[item].tolist()[i] ]))
            #summation.append(output)
    return X_train, X_test, X_pred, Y_train, Y_test, sample_weight, summation

def evaluation(Y_test, Y_pred):
    if sum(Y_test) == 0:
        return 0, 0
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test)), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / (sum(Y_test))
    return rmse, ape

def draw_m2(name, pred, con):
    plt.plot(twitter[name].EventCount.tolist())
    plt.plot(range(split - val_len, split), pred)
    plt.plot(range(split, split + pred_len), con)
    plt.legend(['GT', 'Prediction', 'Con'], frameon=False)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.xticks(np.arange(0, len(date[:split + pred_len + 5]), 3), date[:split + pred_len + 5:3], rotation='60')
    plt.title("Prediction for EventCount of %s" %narrative[index])
    plt.tight_layout()
    #plt.savefig("fig/%d.pdf" % index)
    plt.show()

def nonlinear(gdelt, factor):
    for event in gdelt:
        date = gdelt[event].index
        for item in date:
            gdelt[event][item] = gdelt[event][item] * (1 + factor * log(1 + gdelt[event][item]))

def decay(gdelt, decay_factor):
    Y = copy.deepcopy(gdelt)
    for event in gdelt:
        date = gdelt[event].index
        for i in range(len(date)):
            if i == 0:
                continue
            gdelt[event][date[i]] = decay_factor * gdelt[event][date[i-1]] + Y[event][date[i]] - Y[event][date[i-1]]
            if gdelt[event][date[i]] < 0:
                gdelt[event][date[i]] = 0

def dateGenerator_output(start_date, span):
    date = [int(item) for item in start_date.split('-')]
    now = datetime.datetime(date[0], date[1], date[2]) + datetime.timedelta(hours=19, minutes=0, seconds=0)
    delta = datetime.timedelta(days=1)
    endnow = now+datetime.timedelta(days=span)
    endnow = str(endnow.strftime('%Y-%m-%d'))
    offset = now

    Count = []
    while str(offset.strftime('%Y-%m-%d')) != endnow:
        tmp = int(time.mktime(offset.timetuple()) * 1000.0 + offset.microsecond / 1000.0)
        #Count.append(str(offset.strftime('%Y-%m-%d')))
        Count.append(tmp)
        offset += delta
    return Count

def evaluation(Y_test, Y_pred):
    if sum(Y_test) == 0:
        return 0, 0
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test)), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / (sum(Y_test))
    return rmse, ape

