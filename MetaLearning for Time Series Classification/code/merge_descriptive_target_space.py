import numpy as np
import pandas as pd
import os

data_path_descriptive = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/meta_time_series/meta_data/train"
data_path_performnace = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/MegaComparison"
metrics = ["ACC", "AUROC", "BALACC", "F1", "MaxMemory/TEST/TESTMaxMemory_MEDIANS.csv", "MCC", "NLL", "Prec", "Recall", "TimingsRAW/TRAIN/TRAINTrainTimes_MEDIANS.csv"]

def merge_input(data_path):
    files = os.listdir(data_path)
    lista = []
    for file in files:
        data = pd.DataFrame(pd.read_csv(data_path + "/" + file).iloc[:, 1])
        data.columns = [file.split("_train.csv")[0].lower()]
        lista.append(data)
    to_return = pd.concat(lista, axis=1)
    to_return.index =  pd.read_csv(data_path + "/" + file).iloc[:, 0]
    return to_return.T

def merge_predictive_descriptive(descriptive_data, performance):
    return pd.merge(descriptive_data, performance, on=["index"], how='inner')

def performance_input(data_path_performnace, metrics, data_path_descriptive):
    list_perf_metrics = []
    descriptive_data = merge_input(data_path_descriptive)
    descriptive_data = descriptive_data.reset_index()
    dataset_ = set(descriptive_data.iloc[:, 0])
    for metric in metrics:
        if ("Times" not in metric) and ("Memory" not in metric):
            performance = pd.read_csv(data_path_performnace + "/" + metric + "/TEST/TEST" + metric + "_MEANS.csv")
            performance.iloc[:, 0] = performance.iloc[:, 0].apply(lambda x: str.lower(x))
            performance["metric_type"] = [metric for _ in range(performance.shape[0])]
        elif "memory" not in metric:
            performance = pd.read_csv(data_path_performnace + "/" + metric)
            performance.iloc[:, 0] = performance.iloc[:, 0].apply(lambda x: str.lower(x))
            performance["metric_type"] = ["TrainTime" for _ in range(performance.shape[0])]
        else:
            performance = pd.read_csv(data_path_performnace + "/" + metric)
            performance.iloc[:, 0] = performance.iloc[:, 0].apply(lambda x: str.lower(x))
            performance["metric_type"] = ["MaxMemory" for _ in range(performance.shape[0])]
        per_data_col_names = []
        per_data_col_names.append("index")
        names = [per_data_col_names.append(x) for x in performance.columns[1:]]
        performance.columns = per_data_col_names
        perf_set = set(performance.iloc[:, 0])
        print("missing datasets {}".format(dataset_.difference(perf_set)))
        concat_descriptive_predictive_space = merge_predictive_descriptive(descriptive_data, performance)
        list_perf_metrics.append(concat_descriptive_predictive_space)
    return list_perf_metrics

performance_data = performance_input(data_path_performnace, metrics, data_path_descriptive)
final_frame = pd.DataFrame(np.vstack(performance_data))
final_frame.columns = list(performance_data[0].columns)
final_frame.to_csv("/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/meta_time_series/meta_data/time_series_merged_perf_metrics_2nd_lvl_rep.csv")