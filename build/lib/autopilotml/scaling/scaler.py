from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer


def standard_scale(df, feature_columns):
    """
    This function is used to scale the numerical data using StandardScaler.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_columns (list): The list of feature columns exclude the target column.

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.StandardScaler): The StandardScaler object used for scaling.
    """

    # StandardScaler for numerical data
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, scaler

def target_scale(df, target_column):
    """
    This function is used to scale the numerical data using StandardScaler.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column.

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.MinMaxScaler): The MinMaxScaler object used for scaling.
    """

    # StandardScaler for numerical data
    scaler = MinMaxScaler()
    df[target_column] = scaler.fit_transform(df[target_column])

    return df, scaler

def robust_scale(df, feature_columns):
    """
    This function is used to scale the numerical data using RobustScaler.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_columns (list): The list of feature columns exclude the target column.


    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.RobustScaler): The RobustScaler object used for scaling.
    """

    # RobustScaler for numerical data
    scaler = RobustScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, scaler

def maxabs_scale(df, feature_columns):
    """
    This function is used to scale the numerical data using MaxAbsScaler.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_columns (list): The list of feature columns exclude the target column.

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.MaxAbsScaler): The MaxAbsScaler object used for scaling.
    """

    # MaxAbsScaler for numerical data
    scaler = MaxAbsScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, scaler

def power_transform(df, feature_columns, method='yeo-johnson'):
    """
    This function is used to scale the numerical data using PowerTransformer.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_columns (list): The list of feature columns exclude the target column.
        method (str): The method to use for transformation. Can be one of "yeo-johnson", "box-cox".

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.PowerTransformer): The PowerTransformer object used for scaling.
    """

    # PowerTransformer for numerical data
    scaler = PowerTransformer(method=method)
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, scaler

def quantile_transform(df, feature_columns, method='uniform'):
    """
    This function is used to scale the numerical data using QuantileTransformer.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_columns (list): The list of feature columns exclude the target column.
        method (str): The method to use for transformation. Can be one of "uniform", "normal".

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
        scaler (sklearn.preprocessing.QuantileTransformer): The QuantileTransformer object used for scaling.
    """

    # QuantileTransformer for numerical data
    scaler = QuantileTransformer(method=method)
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, scaler