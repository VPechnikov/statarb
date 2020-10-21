import numpy as np
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn.cluster import DBSCAN
from src.DataRepository import Universes
from src.util.Features import Features
from datetime import date


class Clusterer:

    def __init__(self, clusters=None):
        # To store the clusters of tickers we found the previous day. On day 1 this will be None.
        # Same as return type of DBscan Method
        # Type:
        #    return a dict {int: list of ticker couples} like the following:
        # {
        # 1: [('AAPL', 'GOOG'),('MSFT', 'GOOG'),('MSFT', 'AAPL')],
        # 2: [('AMGN', 'MMM')]
        # }
        #    key: cluster number Cx, x=1,2,...,n

        self.cluster_history = []

    def dbscan(self, today: date, min_samples, eps=None, window=None)->dict:
        """
        It takes as parameters:
        today: is the today date
        min_samples: is an int. The minimum number of points to define a cluster
        eps: is an int. The maximum distance for two points to be considered as close points
        window: is an object of the class Window. Here the get_data function of window is used for collecting the data
        for SnP and ETFs. This data is in Dataframe type.

        The function:
        1)assigns the intraday volatility and the volume for SNP's all stocks to snp_to_cluster_on dataframe. It does
        the same for ETFs.
        2)Concatenates the two dataframes into one with one exception: in case of different lengths, only the common
        dates are used.
        3)Calculates the mean of each SNP's stock volatility over time and the mean of each SNP's stock volume over time.
        It does the same for ETFs.
        4)Ranks the volatilities in ascending order and assigns the result in the ranked_mean_intraday_vols.
        5)Ranks the volumes in ascending order and assigns the result in the volumes.
        6)Normalises the results both for volatilities and for volumes.
        7)Reconcatenates the results into one dataframe X.
        8)Performs the dbscan algorithm in order to create the clusters.
        9)Takes the tickers in each cluster(label) and puts them as values to the dict clusters. Takes the labels(0,1,2)
        as the keys.
        10)Returns the dict clusters.

        """

        self.window = window

        clustering_features = [Features.INTRADAY_VOL, Features.VOLUME]
        snp_to_cluster_on = window.get_data(Universes.SNP, None, clustering_features)
        etf_to_cluster_on = window.get_data(Universes.ETFs, None, clustering_features)
        try:
            data_to_cluster_on = pd.concat([snp_to_cluster_on, etf_to_cluster_on], axis=1)
        except ValueError as _:
            print(f"Got different lengths for etfs and snp for clustering data, taking intersection...")

            common_dates = sorted(set(snp_to_cluster_on.index).intersection(set(etf_to_cluster_on.index)))

            data_to_cluster_on = pd.concat([
                snp_to_cluster_on.loc[snp_to_cluster_on.index.intersection(common_dates)].drop_duplicates(keep='first'),
                etf_to_cluster_on.loc[etf_to_cluster_on.index.intersection(common_dates)].drop_duplicates(keep='first')
            ], axis=1)

        # to now we have a single number per column,
        # (averaging over time dim) so can now compare cross-sectionally, rank-wise

        mean_of_features_over_time = data_to_cluster_on.mean(axis=0)

        ranked_mean_intraday_vols = mean_of_features_over_time.loc[:, Features.INTRADAY_VOL].rank(ascending=True)
        ranked_mean_volumes = mean_of_features_over_time.loc[:, Features.VOLUME].rank(ascending=True)

        rank_normaliser = lambda val, series: ((val - 0.5 * (max(series) - min(series)))) / (
                0.5 * (max(series) - min(series)))

        normed_volume_ranks = ranked_mean_volumes.apply(lambda val: rank_normaliser(val, ranked_mean_volumes))
        normed_intraday_vol_ranks = ranked_mean_intraday_vols.apply(
            lambda val: rank_normaliser(val, ranked_mean_intraday_vols))

        X = pd.concat([normed_volume_ranks, normed_intraday_vol_ranks], axis=1)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        self.tickers = normed_volume_ranks.index
        labels = dbscan.labels_
        self.unique_labels = set(labels)
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = list(labels).count(-1)
        self.noise = np.where(labels == -1)[0]

        clusters = {}

        for cluster_num in self.unique_labels:
            tickers_in_this_cluster = X.index[np.where(labels == cluster_num)[0]]

            clusters[cluster_num] = tickers_in_this_cluster.values

        return clusters

# if __name__ == '__main__':
#
#     X = pd.concat([normed_volume_ranks, normed_intraday_vol_ranks], axis=1)
#
#     plt.figure()
#     plt.scatter(x=X.loc[:, 0], y=X.loc[:, 1])
#     plt.xlabel(str(X.columns[0]))
#     plt.ylabel(str(X.columns[1]))
#     plt.tight_layout()
#     plt.show()
