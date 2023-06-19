import matplotlib.pyplot as plt
import numpy as np
from util import *
import argparse
from pathlib import Path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    link = args.input

    clusterings_all = {}
    silhouette_scores_all = {}
    per_cluester_silhouette_scores_all = {}

    typ = Path(link).stem
    df = pd.read_csv(link, sep=' ').to_numpy()

    clusteringsi, _, silhouette_scoresi, per_cluster_silhi = compute_clusters(df, prefix=typ)
    clusterings_all = {**clusterings_all, **clusteringsi}
    silhouette_scores_all = {**silhouette_scores_all, **silhouette_scoresi}
    per_cluester_silhouette_scores_all = {**per_cluester_silhouette_scores_all, **per_cluster_silhi}

    # Save them to csv files
    create_directory_if_not_exist(args.output_dir)
    for k, v in clusterings_all.items():
        np.savetxt(f"{args.output_dir}/{k}.csv", v.astype(int), fmt='%i', delimiter=",")

