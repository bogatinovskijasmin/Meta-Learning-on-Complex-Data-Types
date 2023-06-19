import pandas as pd
import numpy as np
import time
import os
import json
from collections import defaultdict
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm.contrib.itertools import product
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer

from learning_problems.learning_problem_2.train_model_as_classification import _get_preprocessed_data

from learning_problems.utils.utils import plot_pred_scores, plot_predictions, plot_data_distributions, plot_features, \
    save_features, plot_most_important_features
import warnings
warnings.filterwarnings("ignore")


img_dir = "figs"
target_dir = "results"

# How to organize the methods? By family of methods and treat the problem as multi-class classification?


def classify_family(metric, target_method, data, method2family, plot=False):
    val_folds = 5
    percentage_train_set = 0.7
    data = data.fillna(method="ffill", axis=1)
    data = data[data.metric_type == metric]

    le = preprocessing.LabelEncoder()
    all_indices = np.array(data.index)

    val_scores_mean = []
    val_scores_std = []
    feature_importances = []
    test_predictions = []
    dataset_dictionary = defaultdict(list)

    for _ in range(100):

        train_data = data.sample(int(data.shape[0] * percentage_train_set))
        train_indices = np.array(train_data.index)
        test_indices = np.array(list(set(all_indices).difference(set(train_indices))))
        test_data = data.loc[test_indices, :]
        # two columns were attached - label, family
        X_train = train_data.iloc[:, 1: -16].values
        X_test = test_data.iloc[:, 1: -16].values

        target = data['label'].map(method2family)

        y_train = le.fit_transform(target.loc[train_indices].values)
        y_test = le.fit_transform(target.loc[test_indices].values)

        model = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_leaf=2, max_features="sqrt")
        f_one_scorer = make_scorer(f1_score)
        scores = cross_val_score(model, X_train, y_train, cv=val_folds, scoring=f_one_scorer)

        val_scores_mean.append(np.mean(scores))
        val_scores_std.append(np.std(scores))

        model.fit(X_train, y_train)
        feature_importances.append(model.feature_importances_)

        predictions = model.predict(X_test)

        test_predictions.append(f1_score(y_true=y_test, y_pred=predictions, average="weighted"))

        # for each dataset, store the difference of predicted versus actual labels
        for idx, x in enumerate(test_indices):
            if y_test[idx] == predictions[idx]:
                dataset_dictionary[x].append(1)
            else:
                dataset_dictionary[x].append(0)

    # save the errors for algo-selection
    result = []
    for key in dataset_dictionary.keys():
        row = {
            "dataset_id": int(key),
            "metric": metric,
            "method": target_method,
            "truths": float(np.sum(dataset_dictionary[key])),
            "size": float(np.size(dataset_dictionary[key])),
        }
        result.append(row)

    filename = f"{target_method}_{metric}.json"
    with open(f"./{target_dir}/{filename}", 'w') as json_file:
        json.dump(result, json_file)

    if plot:
        plot_pred_scores(test_predictions, val_scores_mean, val_scores_std, target_method, metric)
        plot_predictions(predictions, y_test, target_method, metric)
        plot_data_distributions(data, dataset_dictionary, target_method, metric)
        plot_features(model, target_method, metric)
        plot_most_important_features(data, feature_importances, model, target_method, metric)

    q = np.mean(feature_importances, axis=0)
    qi = np.where(q > 0.025, True, False)
    save_features(data, qi, target_method, metric)


if __name__ == '__main__':

    METRIC_TYPES = ['TrainTime', 'NLL', 'BALACC', 'Prec', 'AUROC', 'ACC', 'Recall', 'F1', 'MCC']
    METHODS = ['TS-CHIEF', 'HIVE-COTE v1.0', 'ROCKET', 'InceptionTime', 'STC', 'ResNet', 'ProximityForest',
               'WEASEL', 'S-BOSS', 'cBOSS', 'BOSS', 'RISE', 'TSF', 'Catch22']

    start_time = time.time()

    if not os.path.exists('figs'):
        os.makedirs('figs')
    if not os.path.exists('results'):
        os.makedirs('results')
    # flush features json if restarting the process
    with open("./important_features.json", "w") as file:
        json.dump({}, file)

    data = _get_preprocessed_data(meta_data_file="time_series_merged_perf_metrics_2nd_lvl_rep.csv")

    methods_data = data.iloc[:, -15:-1]
    # find methods that give max.metric value
    labels = methods_data.idxmax(axis=1).to_dict()

    data["label"] = pd.Series(labels)

    # todo: very imbalanced target classes
    with open("family_methods.json", "r") as fh:
        family_method_map = json.load(fh)

    TEST_TYPES = ["F1", "TrainTime"]
    TEST_METHODS = ["BOSS", "HIVE-COTE v1.0"]

    FAMILIES = []
    for method in METHODS:
        FAMILIES.append(family_method_map[method])
    FAMILIES = set(FAMILIES)

    res = []
    for method in TEST_METHODS:
        res.append(family_method_map[method])
    TEST_FAMILIES = set(res)

    # for target_method, metric in product(TEST_FAMILIES, TEST_TYPES):
    for target_method, metric in product(FAMILIES, METRIC_TYPES):
        classify_family(metric, target_method, data, family_method_map)

    print(f"{time.time() - start_time} seconds to complete the task")