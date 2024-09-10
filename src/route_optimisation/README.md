# Route Optimisation

Aims to provide a solution to the asymmetric vehicle routing problem.
After receiving a list of delivery orders and the total number of vehicles, the program returns the ordered allocation of deliveries for all routes.

## Recursive Cluster-First, Route-Second

Partition into vehicle routes, then solve for heuristically optimal visit order via recursive clustering. Routes from recursive clusters are merged together to form a larger route. Recursive clustering is necessary with current hardware limitations with D-Wave Ocean.  

The [vehicle clusterer](#clusterer) allocates orders to clusters that represent delivery vehicles. Each cluster is recursively sub-divided into smaller clusters.

- If a cluster's size is less than or equal to threshold, a [distance matrix](#distance-matrix) between orders is created. A [route solver](#route-solver) is used to find the best route from the distance matrix and returns the visit order to the cluster.

- If a cluster's size (the number of orders within the cluster), exceeds the sub-cluster threshold, the cluster is partitioned into sub-cluster using [k-means++](#k-means). Recursion continues for each child cluster

- When solving parent clusters, each child cluster will have a best route allocated via recursion. To solve a parent cluster, each child cluster must be combined to represent a larger route. This has been implemented by creating a [distance matrix](#distance-matrix) where the end of each child route is connected to the start of each other child route. A [route solver](#route-solver) will then find the best route between all child clusters. The parent cluster will then contain the full route between all the child routes.

After recursion, this will result in the ordered allocation of deliveries for all routes.

## Clusterer

Contains the methods to define clustering strategies.

### K-Means++

K-means++ is an unsupervised machine learning algorithm that partitions data points into k clusters.
This program uses the greedy k-means++ implementation. 
More information can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Requires a 'k' value for the number of clusters to be made

Will result in exactly 'k' different routes

### X-Means

Clusters delivery locations via X-means.

Requires a k-max value for the maximum number of possible clusters

Will result 1 to k-max different clusters

## Distance Matrix

Contains the methods to define the distance matrix between orders for a cluster.

### Cartesian Distance Finder

Converts latitude and longitude to 3D cartesian coordinates.

The distance matrix contains the 3D distance between points.

Accuracy greatly suffers as the distances between points on the globe increases. 
This is due to the 3D distance cutting through the globe rather than conforming to the curvature of the Earth.

### Mapbox Distance Finder

Uses latitude and longitude of deliveries to query Mapbox for a distance matrix.

The distance matrix values represent driving duration between points.

## Route Solvers

Contains the methods to define the solver to find optimal route in a cluster

### Brute Force Solver

Generates every permutation of possible routes within a distance matrix and iterates through them to find the optimal route.

Guaranteed to find optimal route.

Takes factorial time to solve, not recommended for large routes.

### D-Wave Solver

Reformulates the distance matrix into a Travelling Salesman Problem QUBO.

The QUBO is used by the D-Wave solver to find a good route from the distance matrix.

More information on QUBO Formulation can be viewed here: \<QUBO LINK>
