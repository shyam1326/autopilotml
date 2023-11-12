

from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier


# Regression Models

def model_fitting(x_train, x_test, y_train, y_test, model_name, params):
    """
    This function is used to create a Regression model.

    Args:
        x_train (pandas.DataFrame): The DataFrame containing the training data.
        x_test (pandas.DataFrame): The DataFrame containing the testing data.
        y_train (pandas.Series): The Series containing the training labels.
        y_test (pandas.Series): The Series containing the testing labels.
        model_name (str): The name of the model.
        **kwargs: The hyperparameters for the selected Regression model.

    Returns:
        model : The selected Regression model.
        y_pred (numpy.ndarray): The predicted labels.
    """
    model_list = {
        # Regression models
        'LinearRegression': LinearRegression,
        'BayesianRidge': BayesianRidge,
        'RandomForestRegressor': RandomForestRegressor,
        'XGBRegressor': XGBRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'SVR': SVR,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'KNeighborsRegressor': KNeighborsRegressor,

        # Classifier models
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'XGBClassifier': XGBClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'SVC': SVC,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'GaussianNB': GaussianNB
    }

    if model_name not in model_list:
        raise ValueError("Model not supported")

    # Create the selected model
    model_class = model_list[model_name]
    
    model = model_class(**params)

    # Fit the model
    model.fit(x_train, y_train)

    # Predict the labels
    y_pred = model.predict(x_test)

    return model, y_pred

