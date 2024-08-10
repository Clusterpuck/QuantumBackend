"""Validation for customer_allocation.py for the runsheet and k value"""

import pandas as pd

def validate_inputs(delivery_list, k, split_threshold):
    """
    Primary method that verifies runsheet and k value. Invalid parameters raise an exception

    Parameters
    ----------
    runsheet: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    k: int
        Number of routes to be made
    split_threshold: int
        Number of sub-routes to be made per partition

    Raises
    ------
    IOError
        If runsheet is not a Pandas dataframe
        If runsheet does not contain atleast 1 row or exactly 3 columns
        If runsheet contains null, incorrect labels, non-unique IDs or customers
        or invalid latitude, longitude values
        If k is not an int
        If k is not between (and including) 1 and total_customers
    """
    try:
        __validate_format(delivery_list)
        __validate_entries(delivery_list)
        total_customers = delivery_list.shape[0] # total_customer = no. of rows
        __validate_k(k,total_customers, split_threshold)
    except (TypeError, ValueError) as ex:
        raise IOError(ex) from ex

def __validate_format(delivery_list):
    """
    Verify the runsheet is in the correct format.
    The format must be a Pandas DataFrame with atleast 1 row and exactly 3 columns.

    Parameters
    ----------
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    
    Raises
    ------
    TypeError
        If runsheet is not a Pandas dataframe
    ValueError
        If runsheet does not contain atleast 1 row or exactly 3 columns
    """
    if not isinstance(delivery_list, pd.DataFrame):
        raise TypeError(f'runsheet must be in a dataframe. '
                        f'Runsheet = {type(delivery_list)}')
    if not delivery_list.shape[0] > 0:
        raise ValueError(f'runsheet must contain atleast one customer. '
                         f'Runsheet has {delivery_list.shape[0]} rows')
    if delivery_list.shape[1] != 3:
        raise ValueError(f'runsheet must contain exactly three columns. '
                         f'Runsheet has {delivery_list.shape[1]} columns')

def __validate_entries(delivery_list):
    """
    Verify the runsheet has correct values.
    runsheet cannot contain null values, have correct labels, unique IDs,
    unique customers and valid latitude and longitude values

    Parameters
    ----------
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    
    Raises
    ------
    ValueError
        If runsheet contains null, incorrect labels, non-unique IDs or customers
        or invalid latitude, longitude values
    """
    if delivery_list.isnull().values.any():
        raise ValueError('Delivery list cannot have null values.')
    labels = delivery_list.columns.values
    if not (labels[0] == "ID" and labels[1] == "Latitude" and labels[2] == "Longitude"):
        raise ValueError(f'Delivery list titles must be "ID", "Latitude" and "Longitude".'
                         f'Currently "{labels[0]}", "{labels[1]}" and "{labels[2]}"')
    if not delivery_list['ID'].is_unique:
        raise ValueError('Delivery list contains non-unique IDs. ')
    if delivery_list['Latitude'].min() < -90 or delivery_list['Latitude'].max() > 90:
        raise ValueError(f'Delivery list contains invalid latitudes.'
                         f'Lowest entry is {delivery_list['Latitude'].min()}'
                         f'Highest entry is {delivery_list['Latitude'].max()}')
    if delivery_list['Longitude'].min() < -180 or delivery_list['Longitude'].max() > 180:
        raise ValueError(f'Delivery list contains invalid Longitude.'
                         f'Lowest entry is {delivery_list['Longitude'].min()}'
                         f'Highest entry is {delivery_list['Longitude'].max()}')

def __validate_k(k, total_customers,split_threshold):
    """
    Verify that k is a valid value

    Parameters
    ----------
    k: int
        Number of routes to be made
    total_customers: int
        Number of customers in delivery list
    split_threshold: int
        Number of routes to be made per partition
    
    Raises
    ------
    Typerror
        If k is not an int
    ValueError
        If k is not between (and including) 1 and total_customers
    """
    if not isinstance(k, int):
        raise TypeError(f'k must be in an int. k = {type(k)}')
    if not 1 <= k <= total_customers:
        raise ValueError(f'k must be greater than 1 and less than total customers. '
                         f'k = {k}, total_customers = {total_customers}')
    if split_threshold <= 1:
        raise ValueError(f'split_threshold must be greater than 1 '
                         f'split_threshold = {k}')
