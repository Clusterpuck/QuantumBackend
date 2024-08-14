# Distance Matrix

This folder contains strategies that can be used to define the distance matrix between delivery locations.

distance_finder.py contains the abstract class that new distance finders must inherit from

___

## Strategies

### Cartesian Distance Finder
Converts latitude and longitude to cartesian coordinates.

[Latitude, Longitude] &rarr; [x,y,z]

The distance matrix contains the 3D distance between points.

Accuracy greatly suffers as the distances between points on the globe increases. 
This is due to the 3D distance cutting through the globe rather than conforming to the curvature of the Earth.

### Mapbox Distance Finder

To be added later