import itertools
import logging
import numpy as np
import time
import os
import json

import warnings
warnings.filterwarnings("ignore")

from tqdm.contrib.itertools import product
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from learning_problems.utils.utils import get_preprocessed_data, save_features, plot_pred_scores, plot_predictions, \
    plot_data_distributions, plot_features, plot_most_important_features

# Define it as a multi-objective optimization problem and observe(F1, time and memory at the same time).
# One can perform discretization and treat the problem as multi-label or directly as multi-target regression

img_dir = "figs"
percentage_train_set = 0.7
val_folds = 5
target_dir = "results"


def make_multi_regression(metric_pair, method, data, plot):
    metric1, metric2 = metric_pair
    tmp1 = data[data.metric_type == metric1]
    tmp2 = data[data.metric_type == metric2]
    # merge two dataframes and preserve left's index
    merged = tmp1[["index", method]].reset_index().merge(tmp2[["index", method]], how="left", on='index',
                                                         suffixes=(f'_{metric1}', f'_{metric2}')).set_index(tmp1.index)
    filters = [f'{method}_{metric1}', f'{method}_{metric2}']

    target = merged[filters]
    scaler = MinMaxScaler()

    # now data is only the first occurence
    data = data[data.metric_type == metric1]

    val_scores_mean = []
    val_scores_std = []
    test_predictions = []
    predictions = []
    all_indices = np.array(data.index)

    index_method_mapping = {}
    for idx, x in enumerate(data.columns):
        index_method_mapping[x] = idx

    dataset_dictionary = defaultdict(list)
    feature_importances = []

    for _ in range(100):
        # X = data[data.metric_type == metric1].iloc[:, 1: -16].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=percentage_train_set, random_state=42)

        train_data = data.sample(int(data.shape[0] * percentage_train_set))
        train_indices = np.array(train_data.index)
        test_indices = np.array(list(set(all_indices).difference(set(train_indices))))
        test_data = data.loc[test_indices, :]

        X_train = train_data.iloc[:, 1: -16].values
        X_test = test_data.iloc[:, 1: -16].values

        y_train = scaler.fit_transform(target.loc[train_indices, :].values)
        y_test = scaler.fit_transform(target.loc[test_indices, :].values)

        model = RandomForestRegressor(n_estimators=20, max_depth=None, min_samples_leaf=2, criterion="mse")

        try:
            scores = (-1) * cross_val_score(model, X_train, y_train, cv=val_folds,
                                            scoring="neg_mean_absolute_error")
        except Exception as e:
            logging.error(f"{e} occurred in {metric} and {method}")
            continue

        # print("Scores mean {} std {}".format(np.mean(scores), np.std(scores)))
        val_scores_mean.append(np.mean(scores))
        val_scores_std.append(np.std(scores))
        # predict 2 metrics, given the data
        model.fit(X_train, y_train)
        feature_importances.append(model.feature_importances_)
        predictions = model.predict(X_test)
        # test mae values appended
        test_predictions.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        # for each dataset, store the abs. difference of predicted metrics versus actual scores
        for idx, x in enumerate(test_indices):
            dataset_dictionary[x].append(np.abs(predictions[idx] - y_test[idx]))

    # save the errors for algo-selection
    result = []
    for key in dataset_dictionary.keys():
        row = {
           "dataset_id": int(key),
            "metric": f"{metric1}_{metric2}",
            "method": method,
            "mae": float(np.mean(dataset_dictionary[key])),
            "std": float(np.std(dataset_dictionary[key])),
            }
        result.append(row)

    metric = f"{metric1}_{metric2}"
    filename = f"{method}_{metric}.json"
    with open(f"./{target_dir}/{filename}", 'w') as json_file:
        json.dump(result, json_file)

    if plot:
        plot_pred_scores(test_predictions, val_scores_mean, val_scores_std, method, metric)
        plot_predictions(predictions, y_test, method, metric)
        plot_data_distributions(data, dataset_dictionary, method, metric)
        plot_features(model, method, metric)
        plot_most_important_features(data, feature_importances, model, method, metric)

    q = np.mean(feature_importances, axis=0)
    qi = np.where(q > 0.025, True, False)
    save_features(data, qi, method, metric)


if __name__ == '__main__':
    #  F1 and TrainTime
    METRIC_TYPES = ['TrainTime', 'NLL', 'BALACC', 'Prec', 'AUROC', 'ACC', 'Recall', 'F1', 'MCC']
    METHODS = ['TS-CHIEF', 'HIVE-COTE v1.0', 'ROCKET', 'InceptionTime', 'STC', 'ResNet', 'ProximityForest',
               'WEASEL', 'S-BOSS', 'cBOSS', 'BOSS', 'RISE', 'TSF', 'Catch22']

    TEST_METRICS = ['TrainTime', 'F1']
    TEST_METHODS = ['BOSS', 'RISE', 'TSF']

    start_time = time.time()

    if not os.path.exists('figs'):
        os.makedirs('figs')
    if not os.path.exists('results'):
        os.makedirs('results')
    # flush features json if restarting the process
    with open("./important_features.json", "w") as file:
        json.dump({}, file)

    data = get_preprocessed_data(meta_data_file="time_series_merged_perf_metrics_2nd_lvl_rep.csv")

    # todo: 504 iterations in 21540 secs
    for metric_pair, method in product(itertools.combinations(TEST_METRICS, 2), TEST_METHODS):
    # for metric_pair, method in product(itertools.combinations(METRIC_TYPES, 2), METHODS):
        make_multi_regression(metric_pair, method, data, plot=False)

    print(f"{time.time() - start_time} seconds to complete the task")

