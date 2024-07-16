import numpy as np

def inter_quartile_range(df, iqr_threshold, Lc, Uc, cap, numerical_columns, **kwargs):

    """ This function is used to remove outliers from the dataset using the inter-quartile range method.
    Args:
        df (pandas.DataFrame): The DataFrame containing the outliers.
        iqr_threshold (float): The threshold to use for removing outliers. Must be greater than 0.
        Lc (float): The lower cut-off value to use for removing outliers. Must be between 0 and 1.
        Uc (float): The upper cut-off value to use for removing outliers. Must be between 0 and 1.
        cap (bool): If True, the outliers will be capped to the upper and lower bounds. If False, the outliers will be removed.
    Returns:
        pandas.DataFrame: The DataFrame containing the outliers.
    """

    for col in numerical_columns:
        Q1 = df[col].quantile(Lc)
        Q3 = df[col].quantile(Uc)

        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_threshold*IQR
        upper_bound = Q3 + iqr_threshold*IQR

        if cap:
            df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else lower_bound if x < lower_bound else x)
        else:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def remove_outliers_zscore(df, numerical_columns, zscore_threshold=3, **kwargs):

    """ This function is used to remove outliers from the dataset using the z-score method.
    Args:
        df (pandas.DataFrame): The DataFrame containing the outliers.
        remove_outliers_zscore_threshold (float): The threshold to use for removing outliers. Must be greater than 0.
    Returns:
        pandas.DataFrame: The DataFrame containing the outliers.
    """

    outlier_mask = np.abs((df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()) > zscore_threshold
    df = df[~outlier_mask.any(axis=1)]

    return df



















































































































