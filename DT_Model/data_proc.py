import pandas as pd
import numpy as np
import json
import copy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import argparse

infoIDs_twitter = ["arrests", "arrests/opposition", "guaido/legitimate", "international/aid",
                   "international/aid_rejected",
                   "international/respect_sovereignty", "maduro/cuba_support", "maduro/dictator", "maduro/legitimate",
                   "maduro/narco", "military", "military/desertions", "other/anti_socialism", "other/censorship_outage",
                   "other/chavez", "other/chavez/anti", "protests", "violence"]
dict_infoID_twitter = dict()
ind = 0
for infoID in infoIDs_twitter:
    dict_infoID_twitter[infoID] = ind
    ind += 1


def json_to_csv(data_file, date, platform, time_series):
    '''
    from json file to csv file
    platform: twitter or youtube
    csv file format:
    InfoID 2018-12-24, 2018-12-25, ...,
    ID1        x           x
    '''
    # with open(f'./{data_file}/' + platform + f'_time_series_to_{date}.json', 'r') as f:
    #     d = json.loads(f.read())
    with open(f'./{data_file}/{time_series}', 'r') as f:
        d = json.loads(f.read())
    # print d.keys()
    t = ''.join(date.spliit('_'))
    dd = {k: pd.read_json(v, orient='columns') for k, v in d.items()}

    index = list(dd.keys())
    column = dd[index[0]].index.values[:]

    # print(index)

    arr_event = []
    arr_user = []
    arr_newuser = []

    for key in index:
        arr_event.append(dd[key]['EventCount'].values[:])
        arr_user.append(dd[key]['UserCount'].values[:])
        arr_newuser.append(dd[key]['NewUserCount'].values[:])

    arr_event = np.array(arr_event)
    arr_user = np.array(arr_user)
    arr_newuser = np.array(arr_newuser)

    df_event = pd.DataFrame(arr_event, index=index, columns=column)
    df_user = pd.DataFrame(arr_user, index=index, columns=column)
    df_newuser = pd.DataFrame(arr_newuser, index=index, columns=column)

    df_event.to_csv(f'./{data_file}/' + platform + f'_{t}_event.csv', index_label='InfoID')
    df_user.to_csv(f'./{data_file}/' + platform + f'_{t}_user.csv', index_label='InfoID')
    df_newuser.to_csv(f'./{data_file}/' + platform + f'_{t}_newuser.csv', index_label='InfoID')


def gdelt_to_csv(data_file, gdelt_file):
    # GDELT time series: form json to csv
    with open(f'./{data_file}/{gdelt_file}', 'r') as f:
        d = json.loads(f.read())
    gdelt = {k: pd.read_json(v, typ='series') for k, v in d.items()}

    # GDELT decay
    decay_factor = 1
    Y = copy.deepcopy(gdelt)
    for event in gdelt:
        date = gdelt[event].index
        for i in range(len(date)):
            if i == 0:
                continue
            gdelt[event][date[i]] = decay_factor * gdelt[event][date[i - 1]] + Y[event][date[i]] - Y[event][date[i - 1]]
            if gdelt[event][date[i]] < 0:
                gdelt[event][date[i]] = 0

    index = list(gdelt.keys())
    column = gdelt[index[0]].index.values
    arr = []
    for key in index:
        arr.append(gdelt[key].values)

    arr = np.array(arr)
    df = pd.DataFrame(arr, index=index, columns=column)
    df.to_csv(f'./{data_file}/gdelt.csv', index_label='InfoID')


def acled_to_csv(data_file, date, acled_file):
    # GDELT time series: form json to csv
    with open(f'./{data_file}/{acled_file}', 'r') as f:
        d = json.loads(f.read())
    gdelt = {k: pd.read_json(v, typ='series') for k, v in d.items()}

    # GDELT decay
    decay_factor = 1
    Y = copy.deepcopy(gdelt)
    for event in gdelt:
        d = gdelt[event].index
        for i in range(len(d)):
            if i == 0:
                continue
            gdelt[event][d[i]] = decay_factor * gdelt[event][d[i - 1]] + Y[event][d[i]] - Y[event][d[i - 1]]
            if gdelt[event][d[i]] < 0:
                gdelt[event][d[i]] = 0

    index = list(gdelt.keys())
    column = gdelt[index[0]].index.values
    arr = []
    for key in index:
        arr.append(gdelt[key].values)

    arr = np.array(arr)
    df = pd.DataFrame(arr, index=index, columns=column)
    df.to_csv('./2021_csv/acled.csv', index_label='InfoID')


def corr_to_csv(data_file, date, corr_file):
    # correlation between infoID (narrative) and eventID (GDELT event)
    # Output csv file format:
    #          eventID1, eventID2, ...
    # infoID1,    c11  ,    c12  , ...
    # infoID2,    c21  ,    c22  , ...

    # with open('./CP4_test/news_leidos_bert_time_series_to_02_14.json') as f:
    # 	data = json.load(f)
    # index = data.keys()
    # col = json.loads(data['arrests']).keys()
    # df_dict = {}
    # for i in index:
    # 	df_dict[i] = {}
    # 	for d in col:
    # 		df_dict[i][d] = json.loads(data[i])[d]
    # df = pd.DataFrame.from_dict(df_dict).T
    # df_twitter = pd.DataFrame(corr_twitter.T, index=infoID, columns=eventID)
    # df_youtube = pd.DataFrame(corr_youtube.T, index=infoID, columns=eventID)
    # corr_file = data_file/news_gdelt_leidos_bert_corr_to_314.json
    t = ''.join(date.split('_'))
    with open(f'./{data_file}/{corr_file}') as f:
        df_twitter = pd.DataFrame(json.load(f)).T
    df_twitter.to_csv(f'./{data_file}_csv/corr_{t}_twitter.csv')
    with open(f'./{data_file}/{corr_file}') as f:
        df_youtube = pd.DataFrame(json.load(f)).T
    df_youtube.to_csv(f'./{data_file}_csv/corr_{t}_youtube.csv')
    return


def generate_training_data(platform, section, corr, nodelist, training_length, prediction_length):
    '''
    Generate training data from csv file
    N: for each infoID, find top N correlated eventIDs
    '''
    path = f'./{data_file}_csv/' + platform + '_' + section + '/'
    t = ''.join(date.split('_'))
    # read narrative time series
    df_event = pd.read_csv(f'./{data_file}_csv/' + platform + f'_{t}_' + section + '.csv', header=0, index_col=0)
    # read GDELT time series
    if platform == "twitter":
        df_gdelt = pd.read_csv(f'./{data_file}_csv/gdelt.csv', header=0, dtype={'InfoID': str})
    if platform == "youtube":
        df_gdelt = pd.read_csv(f'./{data_file}_csv/gdelt.csv', header=0, dtype={'InfoID': str})
    df_gdelt.set_index('InfoID', inplace=True)

    # read correlation
    df_corr = pd.read_csv(f'./{data_file}/corr_{t}_' + platform + '.csv', header=0, index_col=0)
    infoIDs = sorted(df_event.index.values)  # narrative
    # eventIDs = df_corr.columns.values # GDELT

    df_gdelt = df_gdelt.sort_index()
    # Find the popular event ID we use as the input
    active_event_1 = df_corr[df_corr.sum(axis=1).gt(0)].columns.values
    active_event_2 = df_gdelt[df_gdelt.sum(axis=1).gt(0)].index.values
    active_event = set(active_event_1).intersection(set(active_event_2))
    # # active_gdelt = list(active_gdelt)
    df_gdelt = df_gdelt.loc[active_event, :]
    df_corr = df_corr.loc[:, active_event]
    # df_corr[df_corr.lt(0.01)] = 0

    # k+1,K+2,...,k+n
    k = 0
    n = 1
    # path = './train/'+platform+'/'+str(n)
    if not os.path.exists(path):
        os.mkdir(path)

    arr_gdelt = df_gdelt.values
    for infoID in infoIDs:
        if infoID not in nodelist:
            continue
        # if infoID == "maduro/narco":
        # 	df_corr[df_corr.lt(0.001)] = 0
        # else:
        # df_corr[df_corr.lt(0.01)] = 0
        arr_event = df_event.loc[infoID, :].values
        w = df_corr.loc[infoID, :].values
        evt_ind = w.argsort()[-30:][::-1]
        corr[platform][infoID] = {}
        for index in evt_ind:
            corr[platform][infoID][df_corr.columns[index]] = w[index]
        # W = []
        # for j in range(30):
        # W.append(evt_ind[j])
        id = infoID.replace('/', '#')
        f_train = open(os.path.join(path, id + f'_original_{t}_train.csv'), 'w')
        for i in range(0, training_length):
            I = arr_gdelt[:, i + k:i + k + n].T
            x = I * np.array(w)
            x = x.flatten()
            X = x[evt_ind]
            y = arr_event[i]
            if i == 0:
                f_train.write(','.join("x" + str(cn) for cn in range(len(X))))
                f_train.write(",y\n")
            f_train.write(','.join(str(e) for e in X))
            f_train.write(',' + str(y) + '\n')
        f_train.close()

        f_test = open(os.path.join(path, id + f'_original_{t}_test.csv'), 'w')
        for i in range(training_length, training_length + prediction_length):
            I = arr_gdelt[:, i + k:i + k + n].T
            x = I * np.array(w)
            x = x.flatten()
            x = x[evt_ind]
            y = 0  # arr_event[i]
            if i == training_length:
                f_test.write(','.join("x" + str(cn) for cn in range(len(x))))
                f_test.write(",y\n")
            f_test.write(','.join(str(e) for e in x))
            f_test.write(',' + str(y) + '\n')  # '0
        f_test.close()

# start date: 2020-03-10, end_date: 2020-08-10
def generating_new_data(platform, section, nodelist, training_length, prediction_length,
                        start_date, end_date, corr_file, ent_file):
    path = f'./{data_file}_csv/' + platform + '_' + section + '/'
    t = ''.join(date.split('_'))
    df_event = pd.read_csv(f'./{data_file}_csv/' + platform + f'_{t}_' + section + '.csv', header=0, index_col=0)
    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
    with open(f'./{data_file}/{corr_file}', 'r') as f:
        leidos_ts = json.load(f)
    leidos = {k: pd.read_json(v, typ='series').reindex(idx, fill_value=0) for k, v in leidos_ts.items()}
    with open(f'./{data_file}/{ent_file}', 'r') as f:
        zipf_ts = json.load(f)
    zipf = OrderedDict(sorted({k: pd.read_json(v, typ='series').reindex(idx, fill_value=0)
                               for k, v in zipf_ts.items()}.items(), key=lambda kv: kv[1].sum(), reverse=True))

    for infoID in nodelist:
        arr_event = df_event.loc[infoID, :].values
        id = infoID.replace('/', '#')
        f_train = open(os.path.join(path, id + f'_original_{t}_train.csv'), 'w')
        x_train = leidos[infoID] * zipf[infoID]
        # x_train = leidos[infoID]
        for i in range(training_length):
            x = x_train[i]
            y = arr_event[i]
            if i == 0:
                f_train.write(','.join("x" + str(cn) for cn in range(1)))
                f_train.write(",y\n")
            f_train.write(str(x))
            f_train.write(',' + str(y) + '\n')
        f_train.close()

        f_test = open(os.path.join(path, id + f'_original_{t}_test.csv'), 'w')
        for i in range(training_length, training_length + prediction_length):
            x = x_train[i]
            y = 0
            if i == training_length:
                f_test.write(','.join("x" + str(cn) for cn in range(1)))
                f_test.write(",y\n")
            f_test.write(str(x))
            f_test.write(',' + str(y) + '\n')
        f_test.close()
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Parameters for training DT model")
    args.add_argument('-n', '--nodelist_file', default=None, type=str, help='Topic nodes list file path (default: None)')
    args.add_argument('-d', '--date', default=None, type=str, help="Prediction date")
    args.add_argument('-df', '--data_file', default=None, type=str, help="The folder of all data files")
    args.add_argument('-p', '--platform', default="twitter", type=str, help="The platform of the data")
    args.add_argument('-corr', '--corr_file', default=None, type=str, help="The name of correlation file")
    args.add_argument('-gdelt', '--gdelt_file', default=None, type=str, help="The name of gdelt data file")
    args.add_argument('-ent', '--entropy_file', default=None, type=str, help="The name of the processed entropy data file")
    args.add_argument('-ts', '--time_series', default=None, type=str, help="The name of time series file")
    args.add_argument('-sd', '--start_date', default=None, type=str, help="The starting state of training")
    args.add_argument('-pl', '--prediction_length', default=None, type=int, help="The length of prediction")
    args.add_argument('-tl', '--train_length', default=None, type=int, help="The length of training")
    args = args.parse_args()
    nodelist_file = args.nodelist_file
    date = args.date
    data_file = args.data_file
    training_length = args.train_length
    prediction_length = args.prediction_length
    with open(nodelist_file, 'r') as f:
        nodelist = list(f.readlines())
    gdelt_to_csv(data_file, args.gdelt_file)
    corr_to_csv(data_file, date, args.corr_file)
    corr = {}
    platforms = [args.platform]
    # platforms = ['twitter']
    sec = ['event', 'user', 'newuser']
    for plf in platforms:
        json_to_csv(data_file, date, plf, args.time_series)
        corr[plf] = {}
        for s in sec:
            generate_training_data(plf, s, corr, nodelist, training_length, prediction_length)

