import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import time
from sklearn.metrics import mean_squared_error as mse
import copy
from math import log


def load_global_data_cp6(timeseries_path, input_source, exo_path, ent_path, corr_path, nodes_path):
    # Get twitter (platform data)
    with open(timeseries_path) as f:
        twitter_file = json.loads(f.read())
        twitter = {k: pd.read_json(v, orient='columns') for k, v in twitter_file.items()}

    # Get gdelt (input data)
    if not (input_source == 'gdelt' or input_source == 'newssrc'):
        raise Exception('Unknown input source! Please use input_source \'gdelt\' or \'newssrc\'')
    with open(exo_path) as f:
        gdelt_file = json.loads(f.read())
        gdelt = {k: pd.read_json(v, typ='series') for k, v in gdelt_file.items()}

    # Get entropy
    with open(ent_path) as f:
        ent_file = json.loads(f.read())
        ent = {k: pd.read_json(v, typ='series') for k, v in ent_file.items()}

    # Get corr
    with open(corr_path) as f:
        corr = json.loads(f.read())

    # Get frames
    with open(nodes_path, 'r') as f:
        frames = f.read().strip().split('\n')
    events = list(gdelt.keys())

    # Output stats
    print('number of external events (from gdelt): ', len(gdelt))
    print('number of gdelt days: ', len(gdelt[events[0]]))
    print('number of platform days: ', len(twitter[frames[0]]))

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

    if (is_eval):
        return X_train, X_test, Y_train
    return X_train, X_test, Y_train, Y_test

def output_cp6(result, dev_test_len, filename, start_date, end_date):
    order = ['EventCount', 'UserCount', 'NewUserCount']
    # date = dateGenerator(400)

    out = dict()
    for name in result:
        data = np.array([result[name][item] for item in order]).T
        df = pd.DataFrame(data, columns=order, index=pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1)))
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

def postprocess(pred):
    pred = np.array([round(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred

def evaluation(Y_test, Y_pred):
    if sum(Y_test) == 0:
        return 0, 0
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test)), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / (sum(Y_test))
    return rmse, ape
