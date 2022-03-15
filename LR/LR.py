import numpy as np
from tqdm import trange

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from utils import *
from arg_parser import parse_args

def run_model_cp6(X_train, X_test, Y_train, Y_test, platform):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    rmse, ape, size, size_con = 0, 0, 0, 0

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

# Used to compare the prediction and the ground truth by calculating RMSE and APE
def evaluation_cp6(Y_test, Y_pred):
    if sum(Y_test) == 0:
        return 0, 0
    rmse = np.sqrt(mse(np.array(Y_test).cumsum() / (sum(Y_test)), np.array(Y_pred).cumsum() / (sum(Y_pred) + 0.1)))
    ape = 1. * abs(sum(Y_test) - sum(Y_pred)) / (sum(Y_test))
    return rmse, ape

def main_CP6(args):
    platform = args.platform
    data = load_global_data_cp6(args.timeseries_path, args.input_source, \
        args.exo_path, args.ent_path, args.corr_path, args.nodes_path)
    twitter, frames, gdelt, ent, corr = data
    dev_test_len = args.dev_test_len
    # train_len = 366 - dev_test_len
    train_len = len(twitter[frames[0]]) - dev_test_len
    print(train_len)

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
                                                          train_len - dev_test_len, dev_test_len, False,
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
                                                          'NewUserCount', train_len - dev_test_len,
                                                          dev_test_len, False, 'method_1', {})
            rmse, ape, size, size_con, pred = run_model_cp6(X_train[-length:], X_test, Y_train[-length:], Y_test, platform)
            ape_result.append(ape)
            total_result.append(rmse + ape)
        min_total = min(total_result)
        npArray = np.array(total_result)
        occurance = np.where(npArray == min_total)
        min_length = x_axis[occurance[0][0]]
        abnormal_use_last_X[frame] = min_length

    file_name = args.file_name

    result = {name: {'EventCount': [], 'UserCount': [], 'NewUserCount': []} for name in frames}
    for option in ['EventCount', 'UserCount', 'NewUserCount']:
        for frame in frames:
            X_train, X_test, Y_train = dataloader_cp6(twitter, frame, gdelt, corr, most_corr_N[frame], ent, option,
                                                  train_len - dev_test_len, dev_test_len, True, 'method_1',
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

    output_cp6(result, dev_test_len, file_name, args.start_date, args.end_date)

def main():
    args = parse_args()
    main_CP6(args)

if __name__ == '__main__':
    main()
