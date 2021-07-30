import os, csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_csv_data
from src.ModelTree import ModelTree
import json
from sklearn.metrics import mean_squared_error as mse
from random import randrange, uniform

def main():
    nodelist = ["arrests","arrests/opposition","guaido/legitimate","international/aid","international/aid_rejected",
            "international/respect_sovereignty","maduro/cuba_support","maduro/dictator","maduro/legitimate",
            "maduro/narco","military","military/desertions","other/anti_socialism","other/censorship_outage",
            "other/chavez","other/chavez/anti","protests","violence"]
    # targets = ["twitter_event","twitter_user","twitter_newuser"]
    targets = ["twitter_event", "twitter_user", "twitter_newuser",
    "youtube_event","youtube_user","youtube_newuser"]
    yt = {}
    tt = {}
    yt10 = {}
    tt10 = {}
    yt5 = {}
    tt5 = {}
    error = np.zeros(2)
    for key in nodelist:
        print(key)
        tt[key] = {}
        yt[key] = {}
        tt10[key] = {}
        yt10[key] = {}
        tt5[key] = {}
        yt5[key] = {}

        for target in targets:
            name = key.replace('/','#')
            fname = name + "_original_314_train.csv"
            data_csv_data_filename = './CP4_csv/'+os.path.join(target,fname)
            X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            # Train different depth model tree fits and plot results
            #from models.mean_regr import mean_regr
            #plot_model_tree_fit(mean_regr(), X, y, name, mapes, rmses, target)
            from models.lasso import lasso
            #from models.linear_regr import linear_regr
            #top10, top5, avg = plot_model_tree_fit(linear_regr(), X, y, name, target, error)
            top10, top5, avg = plot_model_tree_fit(lasso(), X, y, name, target, error)
            sdate = 1552003200000# 3-22 1552608000000 #3-15 1552003200000 # 3-8 1551398400000 #3-1 #1550188800000# 2-14 #1549584000000
            if target == "twitter_event":
                tt10[key]["EventCount"] = {}
                tt5[key]["EventCount"] = {}
                tt[key]["EventCount"] = {}
                for i in range(len(avg)):
                    tt[key]["EventCount"][str(sdate + i * 86400000)] = int(avg[i])
                    tt10[key]["EventCount"][str(sdate + i * 86400000)] = int(top10[i])
                    tt5[key]["EventCount"][str(sdate + i * 86400000)] = int(top5[i])

            if target == "twitter_user":
                tt10[key]["UserCount"] = {}
                tt5[key]["UserCount"] = {}
                tt[key]["UserCount"] = {}
                for i in range(len(avg)):
                    tt10[key]["UserCount"][str(sdate + i * 86400000)] = int(top10[i])
                    tt5[key]["UserCount"][str(sdate + i * 86400000)] = int(top5[i])
                    tt[key]["UserCount"][str(sdate + i * 86400000)] = int(avg[i])
            if target == "twitter_newuser":
                tt10[key]["NewUserCount"] = {}
                tt5[key]["NewUserCount"] = {}
                tt[key]["NewUserCount"] = {}
                for i in range(len(avg)):
                    tt10[key]["NewUserCount"][str(sdate + i * 86400000)] = int(top10[i])
                    tt5[key]["NewUserCount"][str(sdate + i * 86400000)] = int(top5[i])
                    tt[key]["NewUserCount"][str(sdate + i * 86400000)] = int(avg[i])
                    
            if target == "youtube_user":
                yt10[key]["UserCount"] = {}
                yt5[key]["UserCount"] = {}
                yt[key]["UserCount"] = {}
                for i in range(len(avg)):
                    yt10[key]["UserCount"][str(sdate + i * 86400000)] = int(top10[i])
                    yt5[key]["UserCount"][str(sdate + i * 86400000)] = int(top5[i])
                    yt[key]["UserCount"][str(sdate + i * 86400000)] = int(avg[i])
                    
            if target == "youtube_newuser":
                yt10[key]["NewUserCount"] = {}
                yt5[key]["NewUserCount"] = {}
                yt[key]["NewUserCount"] = {}
                for i in range(len(avg)):
                    yt10[key]["NewUserCount"][str(sdate + i * 86400000)] = int(top10[i])
                    yt5[key]["NewUserCount"][str(sdate + i * 86400000)] = int(top5[i])
                    yt[key]["NewUserCount"][str(sdate + i * 86400000)] = int(avg[i])
                
            if target == "youtube_event":
                yt10[key]["EventCount"] = {}
                yt5[key]["EventCount"] = {}
                yt[key]["EventCount"] = {}
                for i in range(len(avg)):
                    yt[key]["EventCount"][str(sdate + i * 86400000)] = int(avg[i])
                    yt10[key]["EventCount"][str(sdate + i * 86400000)] = int(top10[i])
                    yt5[key]["EventCount"][str(sdate + i * 86400000)] = int(top5[i])
        tt[key] = pd.DataFrame(tt[key]).to_json()
        yt[key] = pd.DataFrame(yt[key]).to_json()
        tt5[key] = pd.DataFrame(tt5[key]).to_json()
        yt5[key] = pd.DataFrame(yt5[key]).to_json()
        tt10[key] = pd.DataFrame(tt10[key]).to_json()
        yt10[key] = pd.DataFrame(yt10[key]).to_json()
    #error = error/108
    #print("rmse:",  error[0])
    #print("ape:", error[1])
    with open('./CP4_test/youtube_UIUC_ML_LASSO_LEIDOS_314.json', 'w') as outfile:
        json.dump(yt10, outfile)
    with open('./CP4_test/twitter_UIUC_ML_LASSO_LEIDOS_314.json', 'w') as outfile:
        json.dump(tt10, outfile)
    with open('youtube_UIUC_lasso_top5_dryrun_314.json', 'w') as outfile:
        json.dump(yt5, outfile)
    with open('twitter_UIUC_lasso_top5_dryrun_314.json', 'w') as outfile:
        json.dump(tt5, outfile)
    # with open('youtube_UIUC_lasso_avg_dryrun.json', 'w') as outfile:
    #     json.dump(yt, outfile)
    # with open('twitter_UIUC_lasso_avg_dryrun.json', 'w') as outfile:
    #     json.dump(tt, outfile)
# ********************************
#
# Side functions
#
# ********************************
def evaluation(Y_test, Y_pred):
    rmse = np.sqrt(mse(np.array(Y_test).cumsum()/(sum(Y_test) + 0.1), np.array(Y_pred).cumsum()/(sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / sum(Y_test)
    return rmse, ape
    
def postprocess(pred):
    pred = np.array([int(item) for item in pred])
    pred[np.where(pred < 0)] = 0
    return pred

def plot_model_tree_fit(model, X, y, name,  target, error):
        #output_filename = os.path.join("output_"+target, "west_1day_5000_linear_{}_greedy_leaf_5_{}_fit.png".format(model.__class__.__name__, name))
        #print("Saving model tree predictions plot y vs x to '{}'...".format(output_filename))

        #mape_ls = np.zeros(12)
        # random forest
        bag = 20
        placeholder = []
        for i in range(bag):
            X_real = copy.deepcopy(X)
            Y_real = copy.deepcopy(y)
            depth = randrange(1, 4)
            leaf = randrange(5,10)
            mask_num = randrange(1,4)
            for j in range(mask_num):
                mask = randrange(len(X_real))
                X_real = np.delete(X_real, mask, 0)
                Y_real = np.delete(Y_real, mask, 0)
        #depth = 1
        #if depth == 1:
            # Form model tree
            # print(" -> training model tree depth={}...".format(depth))
            model_tree = ModelTree(model, max_depth=depth, min_samples_leaf=leaf,
                                   search_type="greedy", n_search_grid=10)
            # Train model tree
            model_tree.fit(X_real, Y_real, verbose=False)
            
            data_csv_data_filename = './CP4_csv/'+os.path.join(target, name+"_original_314_test.csv")
            X_test, y_test, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=False)
            y_train_pred = model_tree.predict(X)
            y_pred = model_tree.predict(X_test)
            y_pred = postprocess(y_pred)
            placeholder.append(y_pred)
        placeholder = np.array(placeholder)
        from scipy import stats
        avg_pred = stats.hmean(placeholder, axis = 0)
        sum_ls = []
        for j in range(20):
            sum_this = sum(placeholder[j])
            if sum_this == 0:
                sum_ls.append(1e10)
                continue
            sum_ls.append(sum_this)
        smallest = (np.array(sum_ls)).argsort()[:10]
        top10 = placeholder[smallest]
        top10_pred = stats.hmean(top10, axis = 0)
        smallest = (np.array(sum_ls)).argsort()[:5]
        top5 = placeholder[smallest]
        top5_pred = stats.hmean(top5, axis = 0)
        #rmse, ape = evaluation(y_test, final_pred)
        #error[0] += rmse
        #error[1] += ape
        return top10_pred, top5_pred, avg_pred


def generate_csv_data(func, output_csv_filename, x_range=(0, 1), N=500):
    x_vec = np.linspace(x_range[0], x_range[1], N)
    y_vec = np.vectorize(func)(x_vec)
    with open(output_csv_filename, "w") as f:
        writer = csv.writer(f)
        field_names = ["x1", "y"]
        writer.writerow(field_names)
        for (x, y) in zip(x_vec, y_vec):
            field_values = [x, y]
            writer.writerow(field_values)

# Driver
if __name__ == "__main__":
    main()

