import numpy as np

def new_geographic_array(runsheet):
    lat_long_array = runsheet[['Latitude', 'Longitude']].to_numpy()
    return lat_long_array


# Convert geographic to cartesian
# Simple math conversion
def geographic_to_cartesian(geo_array):
    cartesian_array = np.zeros((geo_array.shape[0],3))
    for index, x in enumerate(geo_array):
        cartesian_coord = __get_cartesian(x[0],x[1])
        cartesian_array[index] = cartesian_coord
    return cartesian_array

def __get_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return np.array((x,y,z))

def runsheet_to_cartesianV2(runsheet):
    geo_array = new_geographic_array(runsheet)
    cartesian_array = geographic_to_cartesian(geo_array)
    return cartesian_array