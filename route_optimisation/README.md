# Route Optimisation

***

Aims to provide a solution to the vehicle routing problem.
After receiving a list of customers that needs to be delivered and the number of routes, the program creates routes so vehicles can be allocated to those routes later.
 
## Program Structure
***
1. Receives list of customers and number of routes (**k**)
2. Clustering algorithm splits customers in **k** groups and returns the allocation. This refers to which point is part of which cluster
2.1 Inputs are checked to ensure they are valid
2.2 Query database for longitude and latitude
2.3 Geographic position (longitude, latitude) are converted to cartesian position (x,y,z)
2.4 Use K-means++ to cluster points and returns allocation. 
[0 1 0] means: 
* first customer belongs to cluster 0 
* second customer belongs to cluster 1
* third customer belongs to cluster 0
* clusters range from 0 to **k-1** (since zero-based)
3. A dictionary is created to... (keep track of customers currently, will need modify dictionary so it also store lat, long)
4. Each cluster's size is checked to see if it is small enough for the quantum annealer
4.1