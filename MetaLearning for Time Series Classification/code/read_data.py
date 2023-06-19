import pandas as pd
import numpy as np
import matplotlib
import sktime
import scipy


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

store_path = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/results/correlation/"


read_metric = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/MegaComparison/Spec/TEST/TESTSpec_MEANS.csv"



def plot_score(data, name):
    sns.set(font_scale=0.6)
    sns.clustermap(data, metric="correlation", method="complete")
    plt.savefig(store_path+ name)


data = pd.read_csv(read_metric)
data.index = data.iloc[:, 0]
data.drop(["TESTSpec"], axis=1, inplace=True)
plot_score(data.iloc[:, 1:], name="Spec")
