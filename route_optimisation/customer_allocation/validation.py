"""Validation for customer_allocation.py for the runsheet and k value"""

import pandas as pd
import pyodbc
import database_connector as dc
import os

#TODO: Review validation checks

def validate_inputs(runsheet, k, connection_string):
    """Primary method that verifies runsheet and k value

    :param pd.DataFrame runsheet: A runsheet containing IDs and customers
    :param int k: The person sending the message
    :param str connection_string: for the database
    """
    __validate_runsheet_format(runsheet)
    __validate_runsheet_entries(runsheet, connection_string)
    total_customers = runsheet.shape[0] # total_customer = no. of rows
    __validate_k(k,total_customers)

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
    if runsheet.shape[1] != 2:
        raise ValueError(f'runsheet must contain exactly two columns. '
                         f'Runsheet has {runsheet.shape[1]} columns')

#TODO Validate that the customer corresponds to ID
def __validate_runsheet_entries(runsheet, connection_string):
    """Verify the runsheet has correct values.
    runsheet cannot contain null values, have correct labels, unique IDs,
    unique customers and each entry exists in database.

    :param pd.DataFrame runsheet: A runsheet containing IDs and customers
    :param str connection_string: for the database

    :raises TypeError: If runsheet contains null, incorrect labels, non-unique IDs or customers
    :raises pyodbc.DatabaseError: If row doesn't exist in database
    """
    if runsheet.isnull().values.any():
        raise ValueError('runsheet cannot have null values.')
    labels = runsheet.columns.values
    if not (labels[0] == "ID" and labels[1] == "Customer"):
        raise ValueError(f'Runsheet titles must be "ID" and "Customer". '
                         f'Currently "{labels[0]}" and "{labels[1]}"')
    if not runsheet['ID'].is_unique:
        raise ValueError('runsheet contains non-unique IDs. ')
    if not runsheet['Customer'].is_unique:
        raise ValueError('runsheet contains non-unique Customers. ')
    conn = dc.DatabaseConnector(connection_string)
    for row in runsheet.itertuples(index=False): # This should be a seperate function, not technically validation
        cursor = conn.create_cursor()
        string = f'SELECT Top 1 customerName from Customer WHERE ID={row[0]}' # Select exactly one matching row
        cursor.execute(string)
        x = cursor.fetchone()
        if x is None:
            raise pyodbc.DatabaseError(f'Entry does not exist. {row}')
        else:
            if row[1] != x[0]:
                raise ValueError(f'Entry exists but does not match runsheet. {row[1]} and {x[0]}') #TODO Update function string
        conn.close_cursor(cursor)

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
