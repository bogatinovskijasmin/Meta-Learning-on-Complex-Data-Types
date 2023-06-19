import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import os
import glob

from sktime.utils.load_data import load_from_tsfile_to_dataframe

path = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/meta_time_series/Univariate2018_ts/Univariate_ts/"
path1 = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/original_data/Univariate2018_ts/Univariate_ts/"

folder_to_store_train = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/calculated_meta_features_split_train_test/train/"
folder_to_store_test = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/calculated_meta_features_split_train_test/test/"

folders = [f for f in glob.glob(path + "**/", recursive=True)]

for folder in folders[1:]:
    dataset_name = folder.rsplit("/")[-2]
    original_data = path1+dataset_name+"/"
    for dataset in os.listdir(original_data):
        if "TRAIN" in dataset:
            train_shape = load_from_tsfile_to_dataframe(original_data + dataset)[0].shape[0]
            break
    files = os.listdir(folder)
    if len(files) == 0:
        continue
    meta_data_full = pd.read_csv(folder + files[0])
    try:
        os.makedirs(folder_to_store_train + dataset_name)
    except:
        print("cannot make folder")
        continue

    try:
        os.makedirs(folder_to_store_test + dataset_name)
    except:
        print("cannot make folder")
        continue

    meta_data_full.iloc[:train_shape, :].to_csv(folder_to_store_train + dataset_name + "/" + dataset_name + "_train.csv")
    meta_data_full.iloc[train_shape:, :].to_csv(folder_to_store_test + dataset_name + "/" + dataset_name + "_test.csv")
    # break

