import time
import os
import itertools
import json
import logging
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import random
from learning_problems.utils.utils import get_preprocessed_data, save_features, plot_pred_scores, plot_predictions, \
    plot_data_distributions, plot_features, plot_most_important_features
from tqdm.contrib.itertools import product
import warnings
warnings.filterwarnings("ignore")


meta_data_path = "time_series_merged_perf_metrics_2nd_lvl_rep.csv"
img_dir = "figs"
percentage_train_set = 0.7
val_folds = 5
target_dir = "results"


class RunCase:
    def __init__(self, metric, target_method, data, plot):
        self.metric = metric
        self.target_method = target_method
        self.data = data
        self.result = None
        self.plot = plot

    def run(self):
        metric = self.metric
        target_method = self.target_method
        data = self.data

        data = data[data.metric_type == metric]
        all_indices = np.array(data.index)

        val_scores_mean = []
        val_scores_std = []
        test_predictions = []
        predictions = []

        index_method_mapping = {}
        for idx, x in enumerate(data.columns):
            index_method_mapping[x] = idx

        dataset_dictionary = defaultdict(list)
        feature_importances1 = []
        for _ in range(100):
            random_state = random.randint(10, 10000)

            test_data_numpy_X, test_data_numpy_Y, test_indices, train_data_numpy_X, train_data_numpy_Y = self.test_train_split(
                all_indices, data, index_method_mapping, target_method)

            model = RandomForestRegressor(n_estimators=20, max_depth=None, min_samples_leaf=2, criterion="mse")

            try:
                scores = (-1) * cross_val_score(model, train_data_numpy_X, train_data_numpy_Y, cv=val_folds,
                                            scoring="neg_mean_absolute_error")
            except Exception as e:
                logging.error(f"{e} occurred in {metric} and {target_method}")
                continue
            val_scores_mean.append(np.mean(scores))
            val_scores_std.append(np.std(scores))
            # predict the F1-score, given the features
            model.fit(train_data_numpy_X, train_data_numpy_Y)
            feature_importances1.append(model.feature_importances_)
            predictions = model.predict(test_data_numpy_X)
            # test mae values appended
            test_predictions.append(mean_absolute_error(y_true=test_data_numpy_Y, y_pred=predictions))
            # for each dataset, store the abs. difference of predicted F1-score versus actual scores
            for idx, x in enumerate(test_indices):
                dataset_dictionary[x].append(np.abs(predictions[idx] - test_data_numpy_Y[idx]))

        # save the errors for algo-selection
        result = []
        for key in dataset_dictionary.keys():
            row = {
                "dataset_id": int(key),
                "metric": self.metric,
                "method": self.target_method,
                "mae": float(np.mean(dataset_dictionary[key])),
                "std": float(np.std(dataset_dictionary[key])),
            }
            result.append(row)
        self.result = result

        # filename = f"{target_method}_{metric}.json"
        # with open(f"./{target_dir}/{filename}", 'w') as json_file:
        #     json.dump(result, json_file)

        if self.plot:
            plot_pred_scores(test_predictions, val_scores_mean, val_scores_std, target_method, metric)

            plot_predictions(predictions, test_data_numpy_Y, target_method, metric)

            plot_data_distributions(data, dataset_dictionary, target_method, metric)

            plot_features(model, target_method, metric)

            plot_most_important_features(data, feature_importances1, model, target_method, metric)

        q = np.mean(feature_importances1, axis=0)
        qi = np.where(q > 0.025, True, False)
        print(data.columns[1:-16][qi].tolist())
        save_features(data, qi, target_method, metric)

    def test_train_split(self, all_indices, data, index_method_mapping, target_method):
        train_data = data.sample(int(data.shape[0] * percentage_train_set))
        train_indices = np.array(train_data.index)
        test_indices = np.array(list(set(all_indices).difference(set(train_indices))))
        test_data = data.loc[test_indices, :]
        train_data_numpy_X = train_data.iloc[:, 1: -16].values
        train_data_numpy_Y = train_data.iloc[:, index_method_mapping[target_method]].values
        test_data_numpy_X = test_data.iloc[:, 1: -16].values
        test_data_numpy_Y = test_data.iloc[:, index_method_mapping[target_method]].values
        return test_data_numpy_X, test_data_numpy_Y, test_indices, train_data_numpy_X, train_data_numpy_Y


if __name__ == '__main__':

    METRIC_TYPES = ['TrainTime', 'NLL', 'BALACC', 'Prec', 'AUROC', 'ACC', 'Recall', 'F1', 'MCC']
    METHODS = ['TS-CHIEF', 'HIVE-COTE v1.0', 'ROCKET', 'InceptionTime', 'STC', 'ResNet', 'ProximityForest',
               'WEASEL', 'S-BOSS', 'cBOSS', 'BOSS', 'RISE', 'TSF', 'Catch22']

    TEST_TYPES = ["F1"]
    TEST_METHODS = ["BOSS"]
        # , "InceptionTime", "ROCKET"]

    test_combinations = list(itertools.product(TEST_METHODS, TEST_TYPES))
    start_time = time.time()

    if not os.path.exists('figs'):
        os.makedirs('figs')
    if not os.path.exists('results'):
        os.makedirs('results')
    # # flush features json if restarting the process
    # with open("./important_features.json", "w") as file:
    #     json.dump({}, file)

    data = get_preprocessed_data(meta_data_file="time_series_merged_perf_metrics_2nd_lvl_rep.csv")

    # for target_method, metric in product(METHODS, METRIC_TYPES):
    for target_method, metric in test_combinations:
        run_case = RunCase(metric, target_method, data, plot=True)
        run_case.run()

    print(f"{time.time() - start_time} seconds to complete the task")
