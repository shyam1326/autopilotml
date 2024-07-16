import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier

from sklearn.feature_selection import RFE, RFECV

def estimator_model(model_name):
    """
    This function is used to select the estimator model.
    Args:
        model_name (str): The name of the model.
    Returns:
        model : The selected Regression model.
    """

    model_list = {
        # Regression models
        'LinearRegression': LinearRegression,
        'BayesianRidge': BayesianRidge,
        'RandomForestRegressor': RandomForestRegressor,
        'XGBRegressor': XGBRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor,

        # Classifier models
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'XGBClassifier': XGBClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
    }

    if model_name not in model_list:
        raise ValueError("Model not supported")

    # Create the selected model
    model_class = model_list[model_name]
    
    model = model_class()

    return model

def rfe(dataframe, label_column, estimator, n_features_to_select,**kwargs):

    """ This function is used to perform Recursive Feature Elimination.
    Args:
        estimator (str): The estimator to use for feature selection. Can be one of SVC(),SVR(), RandomForestClassifier(), 
        RandomForestRegressor(), GradientBoostingClassifier(), GradientBoostingRegressor(), LogisticRegression(), LinearRegression(), etc..
        n_features_to_select (int): The number of features to select.
        step (int): The number of features to remove at each iteration.
        verbose (int): Controls verbosity of output.
    Returns:
        pandas.DataFrame: The DataFrame containing the selected features.
    """
    
    # Perform Recursive Feature Elimination
    estimator = estimator_model(estimator)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, **kwargs)
    feature_array = selector.fit_transform(dataframe.drop([label_column], axis =1), dataframe[label_column])
    best_columns= list(selector.get_feature_names_out())
    best_columns.append(label_column)
    df = dataframe[best_columns]

    return df, selector

def rfecv(dataframe, label_column, estimator,scoring, cv, n_jobs = -1, **kwargs):

    """ This function is used to perform Recursive Feature Elimination with Cross-Validation.
    Args:
        estimator (str): The estimator to use for feature selection. Import the estimator from sklearn and Can be one of SVC(),SVR(), RandomForestClassifier(), 
        RandomForestRegressor(), GradientBoostingClassifier(), GradientBoostingRegressor(), LogisticRegression(), LinearRegression(), etc..
        step (int): The number of features to remove at each iteration.
        cv (int): Determines the cross-validation splitting strategy.
    Returns:
        pandas.DataFrame: The DataFrame containing the selected features.
    """

    # Perform Recursive Feature Elimination with Cross-Validation
    estimator = estimator_model(estimator)
    selector = RFECV(estimator= estimator, scoring= scoring, cv= cv, n_jobs= n_jobs, **kwargs)
    feature_array = selector.fit_transform(dataframe.drop([label_column], axis =1), dataframe[label_column])
    best_columns= list(selector.get_feature_names_out())
    best_columns.append(label_column)
    df = dataframe[best_columns]
    return df, selector