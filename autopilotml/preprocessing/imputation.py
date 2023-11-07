
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer





def drop_missing_rows(df,**kwargs):
        
        """ This function is used to drop rows with missing values greater than the threshold.
        Args:
            df (pandas.DataFrame): The DataFrame containing the missing values.
            threshold (float): The threshold to use for dropping rows with missing values. Must be between 0 and 1.
        Returns:
            pandas.DataFrame: The DataFrame containing the dropped rows.
        """

        # Drop rows with missing values greater than the threshold
        df = df.dropna(**kwargs, axis=0)

        return df

def drop_missing_columns(df, threshold=0.25):

    """ This function is used to drop columns with missing values greater than the threshold.
    Args:
        df (pandas.DataFrame): The DataFrame containing the missing values.
        threshold (float): The threshold to use for dropping columns with missing values. Must be between 0 and 1.
    Returns:
        pandas.DataFrame: The DataFrame containing the dropped columns.
    """

    # Drop columns with missing values greater than the threshold
    df = df.dropna(thresh=threshold*len(df), axis=1)

    return df  

def imputation(df, strategy_numerical, strategy_categorical, numerical_columns, categorical_columns, fill_value = None, **kwargs):

    """ This function is used to impute the missing values in the dataset.
    Args:
        df (pandas.DataFrame): The DataFrame containing the missing values.
        strategy_numerical (str): The imputation strategy to use for numerical columns. Can be one of "mean", "median", "most_frequent", "knn" or "constant".
        strategy_categorical (str): The imputation strategy to use for categorical columns. Can be one of "most_frequent", or "constant".
        fill_value (str/int/float): The (str) value to use for imputing missing values when strategy is "constant for object data types, For numerical it should be a int/float".
    Returns:
        pandas.DataFrame: The DataFrame containing the imputed data.
    """

    if numerical_columns.any():
        
        if strategy_numerical == 'knn':
            # Impute missing values for numerical columns using KNN
            # imputer = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5), weights=kwargs.get('weights', 'uniform'))
            imputer = KNNImputer(**kwargs)
            df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

        else:
            # Impute missing values for numerical columns using SimpleImputer
            imputer = SimpleImputer(strategy=strategy_numerical, fill_value= [None if strategy_numerical != 'constant' else fill_value])
            df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    if categorical_columns.any():
        # Impute missing values for categorical columns using SimpleImputer
        imputer = SimpleImputer(strategy=strategy_categorical, fill_value=[None if strategy_numerical != 'constant' else fill_value])
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

    return df




