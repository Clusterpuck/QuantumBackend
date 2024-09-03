import numpy as np

from pydantic_models import Order
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.clusterer.resources.queued_x_means import QueuedXMeans


# Pyclustering is deprecated and Clustpy currently requires downgrading to numpy ~1.2
# Clustpy seems to be broken in several ways (small k create too many clusters, global state has zero division, etc)
# As a temp measure, use a custom x-means


class XMeansClusterer(Clusterer):

    def __init__(self, k_max: int, k_init: int = 1):
        """
        Inits an strategy to cluster orders via deterministic x-means. This
        strategy inherently uses 3D, global Cartesian as its clustering metric.

        X-means is a heuristic application of k-means that accepts an initial
        and max cluster count. Not currently set up to be compatible with
        recursive subclustering.

        Parameters
        ----------
        k_max : int
            Max number of clusters to make.
        k_init: int, default=1
            Initial number of clusters.

        Raises
        ------
        ValueError
            k_init less than 1 or more than k_max.
        ValueError
            Invalid value of duplicate_clusters.
        """
        # NOTE: Won't make this recursive subcluster compatible until we get a
        # stable version of x-means
        if not 1 <= k_init <= k_max:
            raise ValueError("k_init must be between 1 and k_max, inclusive.")

        self.__k_max = k_max
        self.__k_init = k_init

    def cluster(self, orders: list[Order]) -> np.ndarray:
        """
        Clusters orders by their Cartesian distances, via x-means with BIC
        metric and k-means++.

        Parameters
        ----------
        orders : list of Order
            Contains all x, y, z points.

        Returns
        -------
        ndarray
            1D, input-aligned array labelled by their cluster
            (eg. [0, 0, 1, 2, 0]).

        Raises
        ------
        ValueError
            Given less points than k_max.
        RuntimeError
            Duplicate clusters found and was configured to fail explicitly.
        """
        # Fast fail trivial
        if len(orders) == 0:
            raise ValueError("Must provide clustering data.")
        elif len(orders) < self.__k_init:
            raise ValueError("X-means does not accept fewer data points than k_init.")

        # Attempt to cluster
        points = np.array([[o.x, o.y, o.z] for o in orders])
        x_means = QueuedXMeans(
            k_init=self.__k_init,
            k_max=self.__k_max,
            init="k-means++",
            random_state=0,
            n_init=10,
        ).fit(points)
        # Not quite ideal as is. Would recommend fixing up x-means subclusters
        # with PCA... if we can confirm that the scoring remains compatible.

        # Check enough clusters were found
        labels = x_means.labels_
        if len(np.unique(labels)) < self.__k_init:
            raise RuntimeError("Too few clusters found (possibly duplicate points).")

        return labels
