# Distance Matrix

This folder contains strategies that can be used to define the distance matrix. 

distance_matrix.py contains the abstract class
distance_matrix_context.py controls which distance matrix strategy is being used. Utilises strategy pattern

This idea is that you can easily change between distance matrix builders without much trouble.
___

## Strategies

### Spatial Matrix
Converts latitude and longitude to cartesian coordinates.

[Latitude, Longitude] &rarr; [x,y,z]

The distance matrix contains the 3D distance from each point to another.

Accuracy greatly suffers as the distances between points on the globe increases. This is due to the 3D distance cutting through the globe rather than conforming to the curvature of the Earth.