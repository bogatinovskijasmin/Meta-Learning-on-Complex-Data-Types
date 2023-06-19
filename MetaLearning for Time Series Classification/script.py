import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def fcn(x, uniques):
    return uniques[x]



path = "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/meta_time_series/Univariate2018_ts/Univariate_ts/TwoPatterns/TwoPatterns_TRAIN.ts"

data = pd.read_csv(path)
tsne = TSNE(2, perplexity=30)
componnts = tsne.fit_transform(data.iloc[:, :-1].values)


uniques = np.unique(data.iloc[:, -1].values)
colors = ["red", "blue", "green", "orange"]
d = {}
for idx, x in enumerate(uniques):
    d[x] = colors[idx]
names = data.iloc[:, -1].apply(lambda x: fcn(x, d))

plt.scatter(componnts[:, 0], componnts[:, 1], c=names)