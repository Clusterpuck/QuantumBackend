# Route Optimisation

Aims to provide a solution to the vehicle routing problem.
After receiving a list of customers that needs to be delivered and the number of routes, the program creates routes so vehicles can be allocated to those routes later.
***
## How to run
You can run this using: (Not done yet)

## Program Structure
**Subject to change**
1. Receives list of customers and number of routes (**k**).
2. Clustering algorithm splits customers in **k** groups and returns the allocation. This refers to which point is part of which cluster
2.1 Inputs are checked to ensure they are valid
2.2 Query database for longitude and latitude
2.3 Geographic position (longitude, latitude) are converted to cartesian position (x,y,z)
2.4 Use K-means++ to cluster points and returns allocation. 
3. A dictionary is created to... (keep track of customers currently, will need modify dictionary so it also store lat, long)
4. Each cluster's size is checked to see if it is small enough for the quantum annealer
4.1 Intialise the cluster hierarchy tree (Explain this later)
4.2 If a cluster is within the split threshold, create a leaf node
4.3 Else if a cluster is beyond the split threshold,
4.3.1 Split the customers in the cluster by (some var) using k-means++
4.3.2 Create a parent node
4.3.3 Recursively split the child clusters until every leaf node is within threshold
4.3.4 Return the final tree
5. Select a distance matrix builder
6. Select a solver strategy
7. Solve the tree (explain this)
___