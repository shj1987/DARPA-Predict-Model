import os
import json
from sklearn import linear_model
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta
from math import floor

def read_main_data(main_data):
    """Get main data

    :param main_data: name of data file
    :return: Dict contain main platform data indexed by narrative
    """
    with open(os.path.join(os.getcwd(), main_data), 'r') as f:
        d = json.loads(f.read())
    dd = {k: pd.read_json(v, orient='columns') for k, v in d.items()}
    return dd

def get_highest_gdelt(num):
    """Get ids for top gdelt events

    :param num: Number of top gdelt events
    :return: Ids for top gdelt events
    """
    with open(os.path.join(os.getcwd(), 'gdelt_time_series.json'), 'r') as f:
        d = json.loads(f.read())
    dd = {k: pd.read_json(v, typ='series').sum() for k, v in d.items()}
    top_ids = sorted(dd, key=dd.get, reverse=True)[:num]  # sort by sum
    return top_ids


def get_cross_data(events, after_shift_gdelt):
    """Get required gdelt data

    :param events: gdelt event ids
    :param after_shift_gdelt: shift the gdelt to match the trend of narratives
    :return: List of required gdelt data
    """
    with open(os.path.join(os.getcwd(), 'gdelt_time_series.json'), 'r') as f:
        d = json.loads(f.read())
    dd = {k: pd.read_json(v, typ='series') for k, v in d.items()}
    return [dd[e][after_shift_gdelt:] for e in events]


def get_highest_related_gdelt(narrative, platform, corrmat_file):
    """Get the gdelt event with highest correlation with input narrative

    :param narrative: input
    :param platform: twitter or youtube
    :param corrmat_file: which correlation matrix we will use
    :return: gdelt event id with highest correlation
    """
    with open(os.path.join(os.getcwd(), corrmat_file), 'r') as f:
        d = json.loads(f.read())
    eventCodeMap = {v: k for k, v in d['eventCodeMap'].items()}  # dict: {index : eventCode}
    narrativeMap = d['narrativeMap']  # dict: {narratives : index}

    if platform == 'twitter':
        GdeltMat = np.array(d['twitterGdeltMat'])
    else:
        GdeltMat = np.array(d['youtubeGdeltMat'])
    # index of top gdelt
    top_arg = GdeltMat[:, narrativeMap[narrative]].argmax()
    # event id of top gdelt
    top_event_id = eventCodeMap[top_arg]

    return top_event_id


def get_cross_corr(narratives, platform, events, corrmat_file):
    """Get correlation between top gdelt events and each narrative

    :param narratives: List of narratives' name
    :param platform: Twitter or Youtube
    :param events: Ids for top gdelt events
    :param corrmat_file: which correlation matrix we will use
    :return: List of correlation lists for each narrative
    """
    with open(os.path.join(os.getcwd(), corrmat_file), 'r') as f:
        d = json.loads(f.read())

    # dict of list to store correlations for each narrative
    corr_matrix = defaultdict(list)
    for na in narratives:
        if na == 'empty':
            corr_matrix[na] = [d["controversies/china/border"][event] for event in events]
        else:
            corr_matrix[na] = [d[na][event] for event in events]
    return corr_matrix


def ARIMA_fun(history, gdelt_data, p, p1, corr, narratives):
    """Training regression model

    :param history: Training data for all narratives
    :param cross_data: Top gdelt data
    :param p: How many days before current gdelt date for prediction
    :param p1: How many days after current gdelt date for prediction
    :param corr: Correlation
    :param narratives: list of names for narratives
    :return: Trained coefficients for three predict target
    """
    scale_factor = {narra: [1,0,1,0,1,0] for narra in narratives}
    for n in range(3):
        data_y_event = []
        data_y_user = []
        data_y_newuser = []
        data_x_event = []
        data_x_user = []
        data_x_newuser = []
        for na in narratives:
            narra_i = history[na].to_numpy()  # ith narrative
            scale_na = scale_factor[na]
            for k in range(p, len(narra_i)):
                y = narra_i[k]  # data on kth day of narrative i
                data_y_event.append(y[0] - scale_na[1])
                data_y_user.append(y[1] - scale_na[3])
                data_y_newuser.append(y[2] - scale_na[5])
                current_x_event = []
                current_x_user = []
                current_x_newuser = []
                for j in range(len(gdelt_data)):
                    gdelt_j = gdelt_data[j].to_list()[k - p:k + p1]  # jth top gdelt event from k-p to k+p1 days
                    current_corr = corr[na][j]  # correlation between ith narrative and jth gdelt
                    current_x_event += [current_corr * g * scale_na[0] for g in gdelt_j]
                    current_x_user += [current_corr * g * scale_na[2] for g in gdelt_j]
                    current_x_newuser += [current_corr * g * scale_na[4] for g in gdelt_j]

                data_x_event.append(current_x_event)
                data_x_user.append(current_x_user)
                data_x_newuser.append(current_x_newuser)

        # linear fitting
        ar_coef = []
        regr = linear_model.LinearRegression()
        regr.fit(np.array(data_x_event), np.array(data_y_event))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)  # append intercept
        ar_coef.append(coef)
        regr.fit(np.array(data_x_user), np.array(data_y_user))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)
        ar_coef.append(coef)
        regr.fit(np.array(data_x_newuser), np.array(data_y_newuser))
        coef = list(regr.coef_)
        coef.append(regr.intercept_)
        ar_coef.append(coef)

        for na in narratives:
            if na == 'controversies/china/border':
                continue
            narra_i = history[na].to_numpy()
            current_na_event = []
            current_na_user = []
            current_na_newuser = []
            na_x_event = []
            na_x_user = []
            na_x_newuser = []
            for k in range(p, len(narra_i)):
                y = narra_i[k]
                current_na_event.append(y[0])
                current_na_user.append(y[1])
                current_na_newuser.append(y[2])
                current_x_gdelt = []

                for j in range(len(gdelt_data)):
                    gdelt_j = gdelt_data[j].to_list()[k - p:k + p1]  # jth top gdelt event from k-p to k+p1 days
                    current_corr = corr[na][j]  # correlation between ith narrative and jth gdelt
                    current_x_gdelt += [current_corr * g for g in gdelt_j]

                na_x_event.append([np.dot(ar_coef[0][:-1], current_x_gdelt) + ar_coef[0][-1]])
                na_x_user.append([np.dot(ar_coef[1][:-1], current_x_gdelt) + ar_coef[1][-1]])
                na_x_newuser.append([np.dot(ar_coef[2][:-1], current_x_gdelt) + ar_coef[2][-1]])

            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(np.array(na_x_event), np.array(current_na_event))
            if len(list(regr.coef_)) != 1:
                print('error')
            scales = list(regr.coef_)
            scales.append(regr.intercept_)

            regr.fit(np.array(na_x_user), np.array(current_na_user))
            scales.append(list(regr.coef_)[0])
            scales.append(regr.intercept_)

            regr.fit(np.array(na_x_newuser), np.array(current_na_newuser))
            scales.append(list(regr.coef_)[0])
            scales.append(regr.intercept_)

            scale_factor[na] = scales

    return [ar_coef, scale_factor]


def autoRegressive(platform, top_num, train_num, p, p1, narratives, corrmat_file, after_shift_gdelt):
    """Prepare for regression

    :param platform: Twitter or Youtube
    :param top_num: Number of top gdelt
    :param train_num: how many days used for train
    :param p: How many days before current gdelt date for prediction
    :param p1: How many days after current gdelt date for prediction
    :param narratives: Narratives used for training
    :param corrmat_file: which correlation matrix we will use
    :param after_shift_gdelt: shift the gdelt to match the trend of narratives
    :return: Trained coefficients
    """

    main_data = read_main_data(platform)

    # split training data
    history = {narrative: main_data[narrative][:train_num] for narrative in narratives}
    gdelt_id = get_highest_gdelt(top_num)
    gdelt_data = get_cross_data(gdelt_id, after_shift_gdelt)
    cross_corr = get_cross_corr(narratives, platform, gdelt_id, corrmat_file)

    coef = ARIMA_fun(history, gdelt_data, p, p1, cross_corr, narratives)

    return coef


# prediction
prevs = [0] # prev dates
platform = 'twitter'
after_shift_gdelt = 1 if platform == 'twitter' else 0 # shift window to match the time lag between different data 
top_num = 15  # top 15 gdelt
test_num = 14   
corrmat_file = 'news_gdelt_leidos_bert_corr_to_8_03.json'
main_file = 'twitter_time_series_to_8_03.json'

# get main data and training narrative names
main_data = read_main_data(platform)
narratives = list(main_data.keys())
print(narratives)
train_num = len(main_data['benefits/covid'])  # number of training dates for main platform

# get gdelt data for prediction
gdelt_id = get_highest_gdelt(top_num)
gdelt_data = get_cross_data(gdelt_id, after_shift_gdelt)

# dict to store gdelt data
with open(os.path.join(os.getcwd(), 'gdelt_time_series.json'), 'r') as f:
    d = json.loads(f.read())
gdelt_dict = {k: pd.read_json(v, typ='series') for k, v in d.items()}

# prediction date
idx = gdelt_dict['010'][train_num: train_num + test_num].index    # dates for prediction
print(idx)

# column tag
columns_name = main_data['benefits/covid'].columns.values

for p in prevs:
    print('Processing: ' + str(p))
    rst_dict = {}  # result dict indexed by narrative name
    train_rst = autoRegressive(platform, top_num, train_num, p, 1, narratives, corrmat_file, after_shift_gdelt)
    coef = train_rst[0] # coefficients
    scale = train_rst[1]
    if len(coef[0]) != (p + 1)*top_num + 1:
        print("error")
    cross_corr = get_cross_corr(narratives, platform, gdelt_id, corrmat_file)  # gdelt correlation for training narratives

    # predict for current training narrative
    for i, na in enumerate(narratives):
        main_data_truth = main_data[na]
        re = []
        current_scale = scale[na]
        for d in range(test_num):  # dth day of prediction
            gdelt_data_predict_d = []
            for j in range(len(gdelt_data)):
                gdelt_j = gdelt_data[j][train_num + d - p:train_num + d + 1].to_list()  # jth gdelt data for prediction on dth day
                corr = cross_corr[na][j]
                gdelt_data_predict_d += [corr * g for g in gdelt_j]

            y_hat_event = np.dot(coef[0][:-1], gdelt_data_predict_d) + coef[0][-1]
            y_hat_user = np.dot(coef[1][:-1], gdelt_data_predict_d) + coef[1][-1]
            y_hat_newuser = np.dot(coef[2][:-1], gdelt_data_predict_d) + coef[2][-1]
            # scale
            y_hat_event = y_hat_event * current_scale[0] + current_scale[1]
            y_hat_user = y_hat_user * current_scale[2] + current_scale[3]
            y_hat_newuser = y_hat_newuser * current_scale[4] + current_scale[5]
            # fix negative value
            if platform == 'twitter':
                y_hat_newuser = max(y_hat_newuser, 1)
                y_hat_user = max(y_hat_user, y_hat_newuser, 1)
                y_hat_event = max(y_hat_event, y_hat_user, y_hat_newuser, 1)
            elif platform == 'youtube':
                y_hat_newuser = max(y_hat_newuser, 0)
                y_hat_user = max(y_hat_user, y_hat_newuser, 0)
                y_hat_event = max(y_hat_event, y_hat_user, y_hat_newuser, 0)
            re.append([int(round(y_hat_event)), int(round(y_hat_user)), int(round(y_hat_newuser))])
        re = np.array(re)
        rst_dict[na] = pd.DataFrame(re, columns=columns_name, index=idx).to_json()

with open('test_cp4/res/' + platform + '_UIUC_ARIMA_CORR_TUNED_BERT.json', 'w') as out:
    json.dump(rst_dict, out)