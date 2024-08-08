"""Validation for customer_allocation.py for the runsheet and k value"""

import pandas as pd

#TODO Comments need to be changed to numpy format
def validate_inputs(runsheet, k):
    """Primary method that verifies runsheet and k value

    :param pd.DataFrame runsheet: A runsheet containing IDs and customers
    :param int k: The person sending the message
    :param str connection_string: for the database
    """
    try:
        __validate_runsheet_format(runsheet)
        __validate_runsheet_entries(runsheet)
        total_customers = runsheet.shape[0] # total_customer = no. of rows
        __validate_k(k,total_customers)
    except (TypeError, ValueError) as ex:
        raise IOError(ex) from ex
    return True

def __validate_runsheet_format(runsheet):
    """Verify the runsheet is in the correct format.
    The format must be a Pandas DataFrame with atleast 1 row and exactly 2 columns.

    :param pd.DataFrame runsheet: A runsheet containing IDs and customers

    :raises TypeError: If runsheet is not a Pandas dataframe
    :raises ValueError: If runsheet does not contain atleast 1 row or exactly 2 columns
    """
    if not isinstance(runsheet, pd.DataFrame):
        raise TypeError(f'runsheet must be in a dataframe. '
                        f'Runsheet = {type(runsheet)}')
    if not runsheet.shape[0] > 0:
        raise ValueError(f'runsheet must contain atleast one customer. '
                         f'Runsheet has {runsheet.shape[0]} rows')
    if runsheet.shape[1] != 3:
        raise ValueError(f'runsheet must contain exactly three columns. '
                         f'Runsheet has {runsheet.shape[1]} columns')

def __validate_runsheet_entries(runsheet):
    """Verify the runsheet has correct values.
    runsheet cannot contain null values, have correct labels, unique IDs,
    unique customers and valid latitude and longitude values

    :param pd.DataFrame runsheet: A runsheet containing IDs and customers
    :param str connection_string: for the database

    :raises ValueError: If runsheet contains null, incorrect labels, non-unique IDs or customers
                       or invalid latitude, longitude values
    """
    if runsheet.isnull().values.any():
        raise ValueError('runsheet cannot have null values.')
    labels = runsheet.columns.values
    if not (labels[0] == "ID" and labels[1] == "Latitude" and labels[2] == "Longitude"):
        raise ValueError(f'Runsheet titles must be "ID", "Latitude" and "Longitude".'
                         f'Currently "{labels[0]}", "{labels[1]}" and "{labels[2]}"')
    if not runsheet['ID'].is_unique:
        raise ValueError('runsheet contains non-unique IDs. ')
    if runsheet['Latitude'].min() < -90 or runsheet['Latitude'].max() > 90:
        raise ValueError(f'runsheet contains invalid latitudes.'
                         f'Lowest entry is {runsheet['Latitude'].min()}'
                         f'Highest entry is {runsheet['Latitude'].max()}')
    if runsheet['Longitude'].min() < -180 or runsheet['Longitude'].max() > 180:
        raise ValueError(f'runsheet contains invalid Longitude.'
                         f'Lowest entry is {runsheet['Longitude'].min()}'
                         f'Highest entry is {runsheet['Longitude'].max()}')

def __validate_k(k, total_customers):
    """Verify that k is a valid value

    :param int k: The person sending the message
    :param int total_customers: The recipient of the message

    :raises TypeError: If k is not an int
    :raises ValueError: If k is not between (and including) 1 and total_customers
    """
    if not isinstance(k, int):
        raise TypeError(f'k must be in an int. k = {type(k)}')
    if not 1 <= k <= total_customers:
        raise ValueError(f'k must be greater than 1 and less than total customers. '
                         f'k = {k}, total_customers = {total_customers}')
