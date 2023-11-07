import pandas as pd
import numpy as np
from sqlite3 import connect



def read_csv(path, **kwargs):
    """Reads a CSV file and returns a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.
        **kwargs: Keyword arguments to pass to pandas.read_csv.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(path, **kwargs)

def read_excel(path, **kwargs):
    """Reads an Excel file and returns a pandas DataFrame.

    Args:
        path (str): Path to the Excel file.
        **kwargs: Keyword arguments to pass to pandas.read_excel.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(path, **kwargs)

