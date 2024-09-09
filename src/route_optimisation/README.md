# Route Optimisation

Aims to provide a solution to the asymmetric vehicle routing problem.
After receiving a list of delivery orders and the total number of vehicles, the program returns the ordered allocation of deliveries for all routes.
Aimed to interface with https://github.com/Clusterpuck/QuantumDelivery

## Recursive Cluster-First, Route-Second

Partition into vehicle routes, then solve for heuristically optimal visit order via recursive clustering.

All orders are partitioned into routes per vehicle, representing a cluster.

If a cluster's size (the number of orders within the cluster), exceeds the sub-cluster threshold, the cluster is partitioned into sub-clusters.

If a cluster's size is within the threshold, a [distance matrix](#distance-matrix) between orders is created. A [route solver](#route-solver) is used to find the optimal route from the distance matrix and returns the visit order to the cluster.

When solving parent clusters, the end of each child route is connected to the start of each other child route. The respective orders are part of the [distance matrix](#distance-matrix).

## Clusterer

Contains the methods to define clustering strategies.

### K-Means++

Clusters delivery locations via K-means++.

Requires a 'k' value, a number of clusters to be made

Will result in k different routes

### X-Means

Clusters delivery locations via X-means.

Requires a k-max value, the maximum number of possible clusters

Will result 1 to k-max different routes

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

Generates every permutation of possible routes and iterates through them to find the optimal route.

Guaranteed to find optimal route.

Takes factorial time to solve, not recommended for large routes.

### D-Wave Solver

Reformulates the distance matrix into a Travelling Salesman Problem QUBO.

The QUBO is used by the D-Wave solver to find a good route from the distance matrix.

More information on QUBO Formulation can be viewed here: \<QUBO LINK>
