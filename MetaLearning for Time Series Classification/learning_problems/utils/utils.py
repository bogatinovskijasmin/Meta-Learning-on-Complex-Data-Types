import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore")


def create_df_from_results(path_to_json):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    dfs = []
    for file in json_files:
        df = pd.read_json(f'./{path_to_json}{file}')
        dfs.append(df)
    temp = pd.concat(dfs)
    return temp


def get_preprocessed_data(meta_data_file):
    data = pd.read_csv(f"../{meta_data_file}")
    data.drop_duplicates(subset=['index', 'metric_type'], keep='last', inplace=True)
    data = data.iloc[:, 1:]
    data = data.fillna(method="ffill", axis=1)
    return data

# todo: data.columns[1:-16] might not always match
def save_features(data, qi, target_method, metric):
    important_features = {}
    important_features[f"{target_method}_{metric}"] = data.columns[1:-16][qi].tolist()
    with open("./important_features.json", "r+") as file:
        features = json.load(file)
        features.update(important_features)
        file.seek(0)
        json.dump(features, file)


def plot_pred_scores(test_predictions, val_scores_mean, val_scores_std, target_method, metric, img_dir='figs'):
    # figure 1
    sns.distplot(val_scores_mean, label="val_mae")
    sns.distplot(val_scores_std, label="val_mae_std")
    sns.distplot(test_predictions, label="test_mae")
    plt.legend()
    plt.savefig(f'./{img_dir}/{target_method}_{metric}_fig1.png', bbox_inches='tight', dpi=300)


def plot_predictions(predictions, test_data_numpy_Y, target_method, metric, img_dir='figs'):
    plt.figure(2)
    try:
        plt.scatter(test_data_numpy_Y, np.abs(predictions - test_data_numpy_Y))
    except ValueError as e:
        logging.error(f"{e} occured when {metric} and {target_method} with {test_data_numpy_Y.shape}")
        raise ValueError
    plt.xlabel("TRUE VALUES OF SCORE {}".format(metric[0]))
    plt.ylabel("Mean Absolute ERROR for each point {}".format(metric[0]))
    plt.savefig(f'./{img_dir}/{target_method}_{metric}_fig2.png', bbox_inches='tight', dpi=300)


def plot_data_distributions(data, dataset_dictionary, target_method, metric, img_dir='figs'):
    plt.figure(3)
    nrows = 10
    ncols = 11
    # fig, ax = plt.subplots(nrows, ncols )
    datasets = list(dataset_dictionary.keys())
    cnt = 0
    for cnti in range(ncols):
        for cntj in range(nrows):
            if cnt >= len(datasets):
                break

            ax = plt.subplot(nrows, ncols, cnt + 1)
            x = datasets[cnt]
            sns.distplot(dataset_dictionary[x])
            # ax.set_xlabel("")
            ax.set_ylabel(" ", size=4)
            ax.set_xlim([0, 1.0])
            ax.tick_params(axis="both", labelsize=2)
            ax.set_title(data.loc[x, "index"], fontsize=2)
            plt.subplots_adjust(hspace=0.8)
            cnt += 1
    plt.savefig(f'./{img_dir}/{target_method}_{metric}_fig3.png', bbox_inches='tight', dpi=300)


def plot_features(model, target_method, metric, img_dir='figs'):
    plt.figure(4)
    plt.scatter(np.arange(0, len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel("")
    plt.ylabel("Overall features with importance")
    plt.savefig(f'./{img_dir}/{target_method}_{metric}_fig4.png', bbox_inches='tight', dpi=300)


def plot_most_important_features(data, feature_importances1, model, target_method, metric, img_dir='figs'):
    plt.figure(5)
    q = np.mean(feature_importances1, axis=0)
    plt.scatter(np.arange(0, len(model.feature_importances_)), q)
    plt.xlabel("")
    plt.ylabel("Most important features")
    plt.savefig(f'./{img_dir}/{target_method}_{metric}_fig5.png', bbox_inches='tight', dpi=300)