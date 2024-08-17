from pydantic_models import ClusterConfig
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer


class VehicleClustererFactory:
    # Validate and decide what vehicle clustering strategy to build for the API
    def create(self, clusterer_config: ClusterConfig) -> Clusterer:
        if clusterer_config.type == "kmeans":
            try:
                # Recursion must not fail, so always try to split on dupes
                return KMeansClusterer(
                    clusterer_config.k,
                    allow_less_data=True,
                    duplicate_clusters="split",
                )
            except AttributeError:
                raise ValueError("K-means missing params. Requires 'k'.")
        else:
            raise ValueError("Unsupported clusterer type.")

        # NOTE: May add others at a later date to demo configurability
