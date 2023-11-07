import optuna

def tuner(model_name, trial: optuna.Trial):
    """
    This function is used to tune the hyperparameters of the selected model.

    Args:
        model_name (str): The name of the model.
        trial (optuna.trial): The trial object that stores the hyperparameters.

    Returns:
        tuner_params (dict): The hyperparameters for the selected model.
    """

    tuner_params = {
        'LinearRegression': lambda:{
            'fit_intercept': trial.suggest_categorical("fit_intercept", ["True", "False"]), 
            'positive': trial.suggest_categorical("positive", ["True", "False"])
                            },

        'BayesianRidge': lambda:{
            'max_iter': trial.suggest_int("max_iter", 50, 1000), 
            'tol': trial.suggest_float("tol", 1e-9, 1.0, log=True),
            'lambda_1': trial.suggest_float("lambda_1", 1e-9, 1.0, log=True),
            'lambda_2': trial.suggest_float("lambda_2", 1e-9, 1.0, log=True),
            'alpha_1': trial.suggest_float('alpha_1', 1e-8, 1.0, log=True), 
            'alpha_2': trial.suggest_float('alpha_2', 1e-8, 1.0, log=True),
            'compute_score': trial.suggest_categorical("compute_score", ["True", "False"]),
            'fit_intercept': trial.suggest_categorical("fit_intercept", ["True", "False"])
                        },

        'RandomForestRegressor': lambda: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10, step=1),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', ['True', 'False'])
                                },

        'XGBRegressor': lambda: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10, step=1),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        },

        'GradientBoostingRegressor': lambda: {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            },

        'SVR': lambda: {
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e3, log=True) if ['kernel'] in ['rbf', 'poly'] else 'scale',
            'degree': trial.suggest_int('degree', 2, 5, step=1) if ['kernel'] == 'poly' else 3
            },

        'DecisionTreeRegressor': lambda: {
            'max_depth': trial.suggest_int('max_depth', 2, 32, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            },

        'KNeighborsRegressor': lambda: {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20, step=1),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),  # Minkowski distance power parameter
            'leaf_size': trial.suggest_int('leaf_size', 1, 50, step=1),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])

            },
        'LogisticRegression': lambda: {
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'None']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
            },

        'RandomForestClassifier': lambda: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10, step=1),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'bootstrap': trial.suggest_categorical('bootstrap', ['True', 'False'])
            },

        'XGBClassifier':  lambda: {
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0, step=0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10, step=1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.1),
            },

        'GradientBoostingClassifier':  lambda: {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, step=0.05),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            },

        'SVC': lambda: {
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),  # Regularization parameter
            'gamma': trial.suggest_float('gamma', 1e-5, 1e2, log=True),  # Kernel coefficient
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),  # Kernel type
            'degree': trial.suggest_int('degree', 2, 5, step=1) if ['kernel'] == 'poly' else 3  # Degree of the polynomial kernel function
            },

        'DecisionTreeClassifier': lambda: {
            'max_depth': trial.suggest_int('max_depth', 2, 32, step=2),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            },

        'KNeighborsClassifier': lambda: {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 10, step=1),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),  # For Minkowski distance (p=1 for Manhattan, p=2 for Euclidean)
            },

        'GaussianNB': lambda: {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-3, log=True)
        }
    }

    if model_name not in tuner_params:
        raise ValueError(f"Model {model_name} is not supported for hyperparameter tuning.")


    # param = tuner_params[model_name]
    
    return tuner_params.get(model_name, lambda: {})()



