"""Factory for vehicle clusterer method"""

from pydantic_models import ClusterConfig
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer


class VehicleClustererFactory:
    """
    Factory class for creating a particular Clusterer instance
    """

    def create(self, clusterer_config: ClusterConfig) -> Clusterer:
        """
        Validate and decide what vehicle clustering strategy to build for the API

        Parameters
        ----------
        clusterer_config : ClusterConfig
            Configuration of the clusterer

        Returns
        -------
        Clusterer
            Clusterer strategy depending on clusterer_config

        Raises
        ------
        ValueError
            If provided clusterer type doesn't exist.
            If clusterer config is missing 'k' value
        """
        if clusterer_config.type == "kmeans":
            try:
                # Recursion must not fail, so always try to split on dupes
                return KMeansClusterer(
                    clusterer_config.k,
                    allow_less_data=True,
                    duplicate_clusters="split",
                )
            except AttributeError as e:
                raise ValueError("K-means missing params. Requires 'k'.") from e
        else:
            raise ValueError("Unsupported clusterer type.")

        # NOTE: May add others at a later date to demo configurability
