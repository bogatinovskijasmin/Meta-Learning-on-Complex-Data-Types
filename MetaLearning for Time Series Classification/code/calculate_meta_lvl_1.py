import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import os
import glob

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from tsfresh import extract_features
from tsfresh import extract_relevant_features



path = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/original_data/Univariate2018_ts/Univariate_ts/"

folders = [f for f in glob.glob(path + "**/", recursive=True)]

for folder in folders[1:]:
    files = os.listdir(folder)
    new_folder = folder.replace("original_data", "meta_time_series")
    try:
        os.makedirs(new_folder)
    except:
        print("cannot make folder")
        continue

    dataset_name = new_folder.rsplit("/")[-2]
    print("Processing dataset {}".format(dataset_name))

    first = True
    control_size = []
    try:
        data1 = []
        for element in files:
            print(element)

            if "TEST" in element and first==True:
                test_first = True
                path_original = folder + element
                data = load_from_tsfile_to_dataframe(folder+element)
                control_size.append(data[0].shape[0])
                data1.append(data)
            else:
                test_first = False
                path_original = folder + element
                data = load_from_tsfile_to_dataframe(folder + element)
                control_size.append(data[0].shape[0])
                data1.append(data)
            first = False

        data_pom = []
        if test_first == True:
            data_pom = [data1[1]]
            data_pom.append(data1[0])
            data1 = data_pom


        data_des = pd.DataFrame(np.vstack([data1[0][0], data1[1][0]]))
        data_tar = np.vstack([np.array(data1[0][1]).reshape(-1, 1), np.array(data1[1][1]).reshape(-1, 1)])


        print("dataset name {}, dataset1_size {}, dataset2_size {}".format(element, control_size[0], control_size[1]))
        assert data_des.shape[0] == np.sum(control_size), "Missing instnace"

        data = (data_des, data_tar)

        ts_fresh_frame = data[0]
        ts_fresh_store_data = []

        for x in range(ts_fresh_frame.shape[0]):
            ts_fresh_store_data.append(ts_fresh_frame.iloc[x].values[0].values)
        pom = np.vstack(ts_fresh_store_data).T

        dataset = []

        for x in range(pom.shape[1]):
            nf = pd.DataFrame(pom[:, x])
            index = pd.DataFrame(np.full((nf.shape[0],), fill_value=x))
            target = pd.DataFrame(np.full((nf.shape[0],), fill_value=data[1][x]))
            q = pd.concat([index, nf, target], axis=1)
            q = q.reset_index()
            dataset.append(q.values)

        q = pd.DataFrame(np.vstack(dataset))
        q.columns = ["time", "id", "values", "target"]
        q.time = q.time.apply(lambda x: np.int64(x))
        q.id = q.id.apply(lambda x: np.int64(x))
        q.values = q.iloc[:, 2].apply(lambda x: np.float64(x))

        X = q.iloc[:, :-1]
        y = pd.Series([int(x) for x in data[1]])


        values = extract_relevant_features(X, y=y, column_id="id", column_sort="time",)
            # print("There are total of {} features for the dataset {}".format(values.shape, dataset_name))

            # values = extract_features(X, column_id="id", column_sort="time")
        y.columns=["target"]
        pd.concat([values, y], axis=1).to_csv(new_folder + element, index=False)

        assert values.shape[0] == np.sum(control_size), "Wrong calculation!!!"
    except:
        print("Dataset with problems {}".format(dataset_name))

    print("##########")


