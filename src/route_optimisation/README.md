# Route Optimisation

Aims to provide a solution to the asymmetric vehicle routing problem.
After receiving a list of delivery orders and the total number of vehicles, the program returns the ordered allocation of deliveries for all routes.
Aimed to interface with https://github.com/Clusterpuck/QuantumDelivery

## Program Structure

### Recursive Cluster-First, Route-Second

Partition into vehicle routes, then solve for heuristically optimal visit order via recursive clustering.

All orders are partitioned into routes per vehicle, representing a cluster.

If a cluster's size (the number of orders within the cluster), exceeds the sub-cluster threshold, the cluster is partitioned into sub-clusters.

If a cluster's size is within the threshold, a [distance matrix](#distance-matrix) between orders is created. A [route solver](#route-solver) is used to find the optimal route from the distance matrix and returns the visit order to the cluster.

When solving parent clusters, the end of each child route is connected to the start of each other child route. The respective orders are part of the [distance matrix](#distance-matrix).

### Distance Matrix

Contains the methods to define the distance metric between orders for a cluster.
View more information [Here](distance_matrix/README.md)

### Route Solver

Contains the methods to find the optimal route for a cluster.
View more information [Here](route_solver/README.md)
