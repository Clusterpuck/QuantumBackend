from typing import Literal
import numpy as np
from sklearn.cluster import KMeans

from route_optimisation.clusterer.clusterer import Clusterer
from pydantic_models import Order


class KMeansClusterer(Clusterer):

    def __init__(
        self,
        k: int,
        allow_less_data: bool = True,
        duplicate_clusters: Literal["allow", "raise", "split"] = "raise",
    ):
        """
        Inits an strategy to cluster orders via deterministic k-means++. This
        strategy inherently uses 3D, global Cartesian as its clustering metric.

        Parameters
        ----------
        k : int
            Number of clusters to make
        allow_less_data : bool, default=True
            If True, less data points than k will just return as many clusters
            as data points (temporarily lower k to data length). If False, fail
            with ValueError instead.
        duplicate_clusters : {'allow', 'raise', 'split'}, default='raise'
            - If allow, return as many clusters as found.
            - If raise, duplicates will raise a ValueError.
            - If split, subdivide large clusters until k (using the lowered k
            from allow_less_data).

        Raises
        ------
        ValueError
            k less than 1.
        ValueError
            Invalid value of duplicate_clusters.
        """
        if k < 1:
            raise ValueError("k must be 1 or greater.")
        elif duplicate_clusters not in ("allow", "raise", "split"):
            raise ValueError("duplicate_clusters must be 'allow', 'raise', or 'split'.")

        self.__k = k
        self.__allow_less_data = allow_less_data
        self.__duplicate_clusters = duplicate_clusters

    def __fallback_split(self, labels: np.ndarray, k: int) -> np.ndarray:
        """
        For when there is no choice left but to force a split.

        Typically, good clustering would either use reduce k, split on sparse
        clusters, or just attempt a reseed. This might hold for vehicle
        clustering, but recursive cluster prefers less recursions. Therefore,
        we try splitting on the largest clusters, often being the problem.

        Parameters
        ----------
        labels: ndarray
            Cluster labels after k-means, ranging 0 to k-1 but missing values.
        k : int
            Number of clusters k-means was supposed to (but failed) to find.

        Returns
        -------
        ndarray
            Complete cluster labelling, fixed with blind subdivisions.
        """
        # Iteratively split largest clusters to balance recursion tree better
        missing_labels = [x for x in range(k) if x not in np.unique(labels)]
        for m_l in missing_labels:
            # Get the current largest cluster
            counts = np.bincount(labels)
            largest_cluster = np.argmax(counts)  # Aka most frequent label
            # Deterministically picks lowest cluster label when equal

            # Get indices to the first half
            half_size = counts[largest_cluster] // 2
            half_indices = np.nonzero(labels == largest_cluster)[0][:half_size]

            # Fill half the cluster with the missing label
            labels[half_indices] = m_l

        return labels

    def cluster(self, orders: list[Order]) -> np.ndarray:
        """
        Clusters orders by their Cartesian distances, via K-means++.

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
            Given less points than k and was configured to fail explicitly.
        RuntimeError
            Duplicate clusters found and was configured to fail explicitly.
        """
        # Fast fail trivial
        if len(orders) == 0:
            raise ValueError("Must provide clustering data.")

        # Enforce settings if insufficent data
        temp_k = self.__k
        if len(orders) < self.__k:
            if self.__allow_less_data:
                temp_k = len(orders)  # Lower k to continue safely
            else:
                raise ValueError("Less than k data points given. Raised due to config.")

        # Attempt to cluster
        points = [[o.x, o.y, o.z] for o in orders]
        k_means = KMeans(
            n_clusters=temp_k, init="k-means++", random_state=0, n_init=10
        ).fit(points)

        # Enforce settings if stacked clusters
        labels = k_means.labels_
        if len(np.unique(labels)) < temp_k:
            if self.__duplicate_clusters == "raise":
                raise RuntimeError("Too few clusters found. Raised due to config.")
            elif self.__duplicate_clusters == "split":
                labels = self.__fallback_split(k_means.labels_, temp_k)

        return labels
