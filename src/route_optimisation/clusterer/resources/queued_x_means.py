"""
Adapted from yasaichi/x_means.py on Github Gist:
https://gist.github.com/yasaichi/254a060eff56a3b3b858

Based on the 2015-04-05 revision, copied on 2024-09-02. Modified into a greedy
dequeue solution to accept a strict k_max, albeit most likely with some loss
in model optimality. Other edits were to document or fix errors.

The linked paper (currently down) points to the Ishioka (2000) extension.
The Gist uses the BIC approx that can cache and build on old BICs, plus uses a
covariance-based probability density function. However, it does not implement
the post-clustering merge, iterative approach, etc.

Original description:
以下の論文で提案された改良x-means法の実装

クラスター数を自動決定するk-meansアルゴリズムの拡張について
http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

import heapq  # Min queue only, so always invert key for "worst BIC"


class Cluster:
    """
    Represents a k-means cluster with BIC-related info.
    """

    # indices are all indices of the cluster data, relative to X's shape
    def __init__(self, X: np.ndarray, indices: np.ndarray, k_means: KMeans, label: int):
        # NOTE: Do not mutate. Treat this as an immutable struct.
        # TODO: Enforce with properties

        # Cluster info
        self.data = X[k_means.labels_ == label]
        self.indices = indices[k_means.labels_ == label]
        self.size = self.data.shape[0]
        self.center = k_means.cluster_centers_[label]

        # BIC-related info
        self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2  # BIC's k
        self.cov = np.cov(self.data.T)
        self.log_likelihood = sum(
            stats.multivariate_normal.logpdf(x, self.center, self.cov, allow_singular=True)
            for x in self.data
        )
        self.bic = self.df * np.log(self.size) - 2 * self.log_likelihood
        # Larger likelihood (smaller Ishioka BIC) is better

        # NOTE: Probability density function is prone to failure when
        # covariance matrix has low eigenvalues. This can occur due to:
        # - Data appearing in lower-like dimensions (most of our use case)
        # - Sporadic covariance from farless samples (data) vs vars (dims)

        # Short of performing fancy dimensionality reduction from original
        # cartesians, the best temporary solution is to allow singular.
        # This may give suboptimal results as it resorts to backup tricks.


class QueuedXMeans:
    """
    Class that performs x-means clustering.
    """

    # NOTE: Since the static nested class builder is hard to typehint, just
    # document the public methods for now

    def __init__(self, k_max=np.inf, k_init=2, **k_means_args):
        """
        Inits an x-means implementation. Similar interface to sklearn's
        k-means, but without the prediction methods.

        While the original and derivatives (that accept a max k and correctly
        enforce it) typically store several models or traverse in BFS to reach
        the optimal model, possibly with pruning or backtracked merging, this
        approach will instead attempt to split only greedily by selecting the
        worst performing cluster. With BIC, higher is worse.

        Parameters
        ----------
        k_max : int or float, default=np.inf
            Max number of clusters to make.
        k_init: int, default=2
            Initial number of clusters. It is possible to make less clusters
            than this, so actual minimum is 1 cluster.
        **k_means_args:
            Args compatible with k-means.
        """
        self.__k_max = k_max
        self.__k_init = k_init
        self.__k_means_args = k_means_args

        # Runtime vars (can be converted to pure functions later)
        self.__k_curr = 0
        self.__pending = []
        self.__completed = []
        self.__tie_breaker = 0  # Keeps cluster queue deterministic

    def __build_clusters(
        self, X: np.ndarray, k_means: KMeans, index: np.ndarray | None = None
    ) -> list[Cluster]:
        """
        From a fitted k-means, init a tuple of Cluster instances.
        """
        if index is None:
            index = np.array(range(0, X.shape[0]))
        labels = range(0, k_means.get_params()["n_clusters"])

        return [Cluster(X, index, k_means, label) for label in labels]

    def __attempt_split(self, cluster: Cluster) -> None:
        """
        Attempts to perform 2-means clustering to split.

        If successful, push new clusters to queue. Else push old to completed.
        """
        # Don't let subclusters get too small
        if cluster.size <= 3:
            self.__completed.append(cluster)
            return

        # Attempt to split into 2
        k_means = KMeans(2, **self.__k_means_args).fit(cluster.data)

        # FIX: Prevent alpha divide by zero if clusters don't diverge
        if len(np.unique(k_means.labels_)) == 1:
            self.__completed.append(cluster)
            return

        c1, c2 = self.__build_clusters(cluster.data, k_means, cluster.indices)

        # Compute the combined BIC of the 2 new clusters
        # FIX: Prevent beta divide by zero with np.divide
        # Alpha handles the inf correctly, albeit with console warning
        beta = np.divide(
            np.linalg.norm(c1.center - c2.center),
            np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov)),
        )
        alpha = 0.5 / stats.norm.cdf(beta)
        bic = 2 * cluster.df * np.log(cluster.size) - 2 * (
            cluster.size * np.log(alpha) + c1.log_likelihood + c2.log_likelihood
        )

        # If BIC improved (got smaller), keep attempt and requeue
        if bic < cluster.bic:
            heapq.heappush(self.__pending, (-c1.bic, self.__tie_breaker, c1))
            heapq.heappush(self.__pending, (-c2.bic, self.__tie_breaker + 1, c2))
            self.__tie_breaker += 2
            self.__k_curr += 1
        else:
            self.__completed.append(cluster)

    def fit(self, X: np.ndarray):
        """
        Cluster data points via x-means.

        Parameters
        ----------
        X : array-like
            Data points, shaped (n_samples, n_features).

        Returns
        -------
        self : XMeans
            Fitted estimator.
        """
        self.__completed = []

        # Initial clustering, producing k_init clusters
        initial_clusters = self.__build_clusters(
            X, KMeans(self.__k_init, **self.__k_means_args).fit(X)
        )
        self.__k_curr = len(initial_clusters)

        # Set up queuing stuff, making sure to invert for max-BIC
        self.__pending = [(-c.bic, i, c) for i, c in enumerate(initial_clusters)]
        heapq.heapify(self.__pending)
        self.__completed = []
        self.__tie_breaker = len(self.__pending)

        # 2-means cluster on the worst cluster until BIC stops improving
        while self.__k_curr < self.__k_max and len(self.__pending) != 0:
            self.__attempt_split(heapq.heappop(self.__pending))

        # Generate sklearn-style properties
        self.labels_ = np.empty(X.shape[0], dtype=np.intp)

        final_clusters = [t[2] for t in self.__pending] + self.__completed
        for i, c in enumerate(final_clusters):
            self.labels_[c.indices] = i
        self.cluster_centers_ = np.array([c.center for c in final_clusters])
        self.cluster_log_likelihoods_ = np.array(
            [c.log_likelihood for c in final_clusters]
        )
        self.cluster_sizes_ = np.array([c.size for c in final_clusters])

        return self
