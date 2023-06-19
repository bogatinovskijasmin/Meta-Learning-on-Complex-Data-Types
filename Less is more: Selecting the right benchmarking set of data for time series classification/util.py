from sklearn.cluster import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import *
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

def create_directory_if_not_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def compute_clusters(df, prefix=''):
    clusterings = {}
    clusterings_proba = {}
    silhouette_scores = {}
    per_cluester_silhouette_scores = {}

    """
    embedding_algorithm = TSNE(n_components=2, random_state=0, perplexity=5)
    X_2d = embedding_algorithm.fit_transform(df)
    clusterings[f'{prefix}_TSNE_2_KMeans'] = KMeans(random_state=0, n_clusters=10).fit(X_2d).labels_
    """
  
    #clusterings[f'{prefix}_KMeans'] = KMeans(random_state=0, n_clusters=11).fit(df).labels_
    #clusterings[f'{prefix}_AgglomerativeClustering'] = AgglomerativeClustering(n_clusters=9).fit(df).labels_
    #clusterings[f'{prefix}_Birch'] = Birch(n_clusters=10).fit(df).labels_
    #clusterings[f'{prefix}_DBSCAN'] = DBSCAN(eps=6.3).fit(df).labels_
  
    #gm = GaussianMixture(n_components=10, random_state=0).fit(df)
    #clusterings[f'{prefix}_GaussianMixture'] = gm.predict(df)
    #clusterings_proba[f'{prefix}_GaussianMixture'] = gm.predict_proba(df)
  
    #Clustering on reduced dimentions
    '''
    clusterings[f'{prefix}_PCA_2_KMeans'] = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', KMeans(random_state=0, n_clusters=10))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_10_KMeans'] = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', KMeans(random_state=0, n_clusters=10))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_30_KMeans'] = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', KMeans(random_state=0, n_clusters=10))]).fit(df)['cluster_algo'].labels_
  
    clusterings[f'{prefix}_PCA_2_AgglomerativeClustering'] = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', AgglomerativeClustering(n_clusters=9))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_10_AgglomerativeClustering'] = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', AgglomerativeClustering(n_clusters=9))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_30_AgglomerativeClustering'] = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', AgglomerativeClustering(n_clusters=9))]).fit(df)['cluster_algo'].labels_
  
    clusterings[f'{prefix}_PCA_2_AgglomerativeClustering_cos'] = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', AgglomerativeClustering(n_clusters=9, affinity='cosine', linkage='average'))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_10_AgglomerativeClustering_cos'] = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', AgglomerativeClustering(n_clusters=9, affinity='cosine', linkage='average'))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_30_AgglomerativeClustering_cos'] = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', AgglomerativeClustering(n_clusters=9, affinity='cosine', linkage='average'))]).fit(df)['cluster_algo'].labels_
  
    clusterings[f'{prefix}_PCA_2_Birch'] = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', Birch(n_clusters=10))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_10_Birch'] = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', Birch(n_clusters=10))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_30_Birch'] = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', Birch(n_clusters=10))]).fit(df)['cluster_algo'].labels_
  
    clusterings[f'{prefix}_PCA_2_DBSCAN'] = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', DBSCAN(eps=4))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_10_DBSCAN'] = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', DBSCAN(eps=3))]).fit(df)['cluster_algo'].labels_
    clusterings[f'{prefix}_PCA_30_DBSCAN'] = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', DBSCAN(eps=3))]).fit(df)['cluster_algo'].labels_
    '''
  
    # GaussianMixture clusters
    '''
    pca_gauss_pipeline = Pipeline([('pca', PCA(n_components=2)), ('cluster_algo', GaussianMixture(n_components=10, random_state=0))]).fit(df)
    clusterings[f'{prefix}_PCA_2_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict(pca_gauss_pipeline['pca'].transform(df))
    clusterings_proba[f'{prefix}_PCA_2_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict_proba(pca_gauss_pipeline['pca'].transform(df))
  
    pca_gauss_pipeline = Pipeline([('pca', PCA(n_components=10)), ('cluster_algo', GaussianMixture(n_components=10, random_state=0))]).fit(df)
    clusterings[f'{prefix}_PCA_10_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict(pca_gauss_pipeline['pca'].transform(df))
    clusterings_proba[f'{prefix}_PCA_10_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict_proba(pca_gauss_pipeline['pca'].transform(df))
  
    pca_gauss_pipeline = Pipeline([('pca', PCA(n_components=30)), ('cluster_algo', GaussianMixture(n_components=10, random_state=0))]).fit(df)
    clusterings[f'{prefix}_PCA_30_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict(pca_gauss_pipeline['pca'].transform(df))
    clusterings_proba[f'{prefix}_PCA_30_GaussianMixture'] = pca_gauss_pipeline['cluster_algo'].predict_proba(pca_gauss_pipeline['pca'].transform(df))
    '''
  
    # Compute hierarhical custering with cos similarity for different sized clusters
    #length = np.sqrt((df**2).sum(axis=1))[:,None]
    #X = df / length
    #clusterings[f'{prefix}_K_medians_cos'] = KMeans(n_clusters=10, random_state=0).fit(X).labels_
  
    #for i in range(3, 30):
    #    algo_name = f'{prefix}_K_medians_cos_clusters_{i}'
    #    res = KMeans(n_clusters=i, random_state=0).fit(X).labels_
    #    clusterings[algo_name] = res
    #    silhouette_scores[algo_name] = metrics.silhouette_score(df, res, metric='cosine')
  
  
    # Compute hierarhical custering with cos similarity for different sized clusters
    for i in range(3, 30):
        algo_name = f'{prefix}_AgglomerativeClustering_cos_clusters_{i}'
        res = AgglomerativeClustering(n_clusters=i, affinity='cosine', linkage='average').fit(df).labels_
        clusterings[algo_name] = res
        #silhouette_scores[algo_name] = metrics.silhouette_score(df, res, metric='cosine')
        
        #sample_silhouette_values = metrics.silhouette_samples(df, res, metric='cosine')

        #means_lst = []
        #for label in range(i):
        #    means_lst.append(sample_silhouette_values[res == label].mean())
        #per_cluester_silhouette_scores[algo_name] = means_lst
  
    return clusterings, clusterings_proba, silhouette_scores, per_cluester_silhouette_scores
