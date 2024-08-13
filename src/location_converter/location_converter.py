import numpy as np
from pydantic_models import OrderInput

def locations_to_2D(locations: list[OrderInput]):
    # This must convert geographic to 2D

    latitudes = [location.lat for location in locations]
    longitudes = [location.long for location in locations]

    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    R = 6371  # Radius of Earth
    centre_point_deg = -31.952258602714696

    # Convert to radians
    center_latitude_radians = np.radians(centre_point_deg)
    longitudes_radians = np.radians(longitudes)
    latitudes_radians = np.radians(latitudes)
    
    # Apply equirectangular projection
    longitudes = R * longitudes_radians * np.cos(center_latitude_radians)
    latitudes = R * latitudes_radians
    
    return latitudes, longitudes

def geographic_to_2D(latitudes, longitudes):
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    R = 6371  # Radius of Earth
    centre_point_deg = -31.952258602714696

    # Convert to radians
    center_latitude_radians = np.radians(centre_point_deg)
    longitudes_radians = np.radians(longitudes)
    latitudes_radians = np.radians(latitudes)
    
    # Apply equirectangular projection
    longitudes = R * longitudes_radians * np.cos(center_latitude_radians)
    latitudes = R * latitudes_radians
    
    return latitudes, longitudes
