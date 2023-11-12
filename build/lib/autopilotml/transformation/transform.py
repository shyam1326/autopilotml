
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def onehot_transform(df, categorical_columns):
    """
    This function is used to transform categorical data into numerical data.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        categorical_columns (list): The list of categorical columns.
        
    Returns:
        pandas.DataFrame: The DataFrame containing the transformed data.
        encoder (sklearn.preprocessing.OneHotEncoder): The OneHotEncoder object used for transformation.
    """

    # OneHotEncoder for binary categorical data
    encoder = OneHotEncoder(handle_unknown ='ignore')
    feature_array = encoder.fit_transform(df[categorical_columns]).toarray()
    feature_labels = encoder.get_feature_names_out(categorical_columns)
    df[feature_labels] = feature_array
    df.drop(categorical_columns, axis=1, inplace=True)

    return df, encoder

def ordinal_transform(df, categorical_columns):
    """
    This function is used to transform categorical data into numerical data.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        categorical_columns (list): The list of categorical columns.

    Returns:
        pandas.DataFrame: The DataFrame containing the transformed data.
        encoder (sklearn.preprocessing.OrdinalEncoder): The OrdinalEncoder object used for transformation.
    """

    # OrdinalEncoder for binary categorical data
    encoder = OrdinalEncoder()
    feature_array = encoder.fit_transform(df[categorical_columns])
    df[categorical_columns] = feature_array

    return df, encoder

def label_transform(df, target_column):
    """
    This function is used to transform target feature into numerical feature.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column.

    Returns:
        pandas.DataFrame: The DataFrame containing the transformed data.
        encoder (sklearn.preprocessing.LabelEncoder): The LabelEncoder object used for transformation.
    """

    # OrdinalEncoder for binary categorical data
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])

    return df, encoder