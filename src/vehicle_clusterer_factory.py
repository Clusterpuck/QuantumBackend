"""Validate and decide what vehicle clustering strategy to build for the API"""

from pydantic_models import ClusterConfig
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer
from route_optimisation.clusterer.x_means_clusterer import XMeansClusterer


class VehicleClustererFactory:
    """Validate and decide what vehicle clustering strategy to build for the API"""

    def create(self, clusterer_config: ClusterConfig) -> Clusterer:
        """
        Creates clusterer based on provided cluster configuration

        Parameters
        ----------
        clusterer_config: ClusterConfig
            Configuration of clusterer

        Returns
        -------
        Clusterer
            Specific clusterer object

        Raises
        ------
        ValueError
            If solver type is unknown
            If kmeans is missing k parameter
            If xmeans is missing maximum k parameter
        """
        if clusterer_config.type == "kmeans":
            try:
                # Can be made stricter for vehicle-level, but keeping for now
                return KMeansClusterer(
                    clusterer_config.k,
                    allow_less_data=True,
                    duplicate_clusters="split",
                )
            except AttributeError as e:
                raise ValueError("K-means missing params. Requires 'k'.") from e
        elif clusterer_config.type == "xmeans":
            try:
                # Max k is required, but initial can default to 1
                params = {"k_max": clusterer_config.k_max}
                if "k_init" in clusterer_config.model_extra:
                    params["k_init"] = clusterer_config.k_init

                # Currently not supported for route subclustering
                # (...and not exactly the most recommended even if feasible)
                return XMeansClusterer(**params)
            except AttributeError as e:
                raise ValueError("X-means missing params. Requires 'k_max'.") from e
        else:
            raise ValueError("Unsupported clusterer type.")

        # NOTE: May add others at a later date to demo configurability
