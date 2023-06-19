import pandas as pd
import itertools
import logging
from learning_problems.utils.utils import create_df_from_results


def map_to_dataset_name(dataset_id, dataset_names_map):
    try:
        return dataset_names_map[dataset_id]
    except KeyError as e:
        logging.error(f"{dataset_id} does not exist")


def get_best_method(dataset_name, metric, temp):
    filtered_df = temp[(temp.dataset_name == dataset_name) & (temp.metric == metric)].reset_index()
    if filtered_df.empty:
        return 'NO METHOD'
    if 'mae' in temp:
        idx = filtered_df.mae.idxmin()
    else:
        raise Exception
    return filtered_df.loc[idx, :].method


def main(dataset_map):
    temp = create_df_from_results(path_to_json)
    temp.reset_index()
    if 'dataset_id' in temp:
        temp["dataset_name"] = temp.dataset_id.apply(lambda x: map_to_dataset_name(x, dataset_map))
    combinations = list(itertools.product(DATASET_NAMES, METRICS))
    labels = []
    for dataset_name, metric in combinations:
        best = get_best_method(dataset_name, metric, temp)
        label = {
            "dataset_name": dataset_name,
            "metric": metric,
            "best_method": best,

        }
        labels.append(label)
    df = pd.DataFrame(labels)
    df.to_csv("best_methods.csv", index=False)


if __name__ == '__main__':

    METRICS = ['TrainTime', 'NLL', 'BALACC', 'Prec', 'AUROC', 'ACC', 'Recall', 'F1', 'MCC']

    meta_data_path = "time_series_merged_perf_metrics_2nd_lvl_rep.csv"

    path_to_json = "results/"
    data = pd.read_csv(f"../{meta_data_path}")
    data.drop_duplicates(subset=['index', 'metric_type'], keep='last', inplace=True)

    DATASET_NAMES = data["index"].unique()
    dataset_names_map = data["index"].to_dict()
    main(dataset_map=dataset_names_map)


