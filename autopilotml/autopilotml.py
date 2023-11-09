
from autopilotml.data_loading import read_csv, read_excel, read_sqlite, read_postgres, read_mysql, read_mongo
from autopilotml.preprocessing import drop_missing_rows, drop_missing_columns, imputation, inter_quartile_range, remove_outliers_zscore
from autopilotml.transformation import ordinal_transform, onehot_transform, label_transform
from autopilotml.scaling import standard_scale, target_scale, robust_scale, maxabs_scale, power_transform, quantile_transform
from autopilotml.feature_selection import rfe, rfecv
from autopilotml.model_building import model_fitting
from autopilotml.model_tuning import tuner

from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import optuna
import mlflow
from mlflow import MlflowClient
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



#Stage 1: Data Loading

def load_data(path, csv = True, excel = False, **kwargs):
    """ This function is used to load the data from the source.

    Args:
        path (str): The path to the source file.
        csv (bool): If True, the source file is a CSV file.
        excel (bool): If True, the source file is an Excel file.
        **kwargs: Keyword arguments to pass to pandas.read_csv or pandas.read_excel.

    Returns:
        pandas.DataFrame: The DataFrame containing the data.
    """
    # Load the data from the csv source
    if csv:
        df = read_csv(path, **kwargs)
    elif excel:
        df = read_excel(path, **kwargs)

    return df

def load_database(database_type, query, sqlite_db_path = None, host= None, port= None, database_name= None, 
                  username= None, password=None, collection_name= None, **kwargs):
    """ This function is used to load the data from the database.

    Args:
        database (str): The type of database. For instance 'sqlite', 'mysql', 'postgres', 'mongo'.
        query (str): The query to execute.
        sqlite_db_path (str): The path to the SQLite database.

        The following arguments are used for MySQL, PostgreSQL and MongoDB databases:
        host (str): The host name.
        port (int): The port number.
        database_name (str): The database name.
        username (str): The username if exists. by default None.
        password (str): The password if exists. by default None.
        collection_name (str): The collection name for mongodb database. by default None.
        **kwargs: Keyword arguments to pass to sqlite3.connect, psycopg2.connect, mysql.connector.connect, pymongo.MongoClient.

    Returns:
        pandas.DataFrame: The DataFrame containing the data.
    """
    try:
        if database_type == 'sqlite':
            df = read_sqlite(query, sqlite_db_path)
        elif database_type == 'postgres':
            df = read_postgres(query, host, port, database_name, username, password)
        elif database_type == 'mysql':
            df = read_mysql(query, host, port, database_name, username, password)
        elif database_type == 'mongo':
            df = read_mongo(query, host, port, database_name, username, password, collection_name)
        else:
            raise ValueError('The database type is not supported.')
        
        return df
        
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)


#Stage 2: Data Preprocessing

def preprocessing(dataframe, label_column,
                missing= {'type': 'drop',
                        'drop_columns': True, 
                        'threshold': 0.25, 
                        'strategy_numerical': 'knn',
                        'strategy_categorical': 'most_frequent',
                        'fill_value': None}, 
                outlier= {'method': 'None',
                        'zscore_threshold': 3,
                        'iqr_threshold': 1.5,
                        'Lc': 0.25, 
                        'Uc': 0.75,
                        'cap': False},
                **kwargs):
    
    """ This function is used to preprocess the data.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column.

        missing (dict): The dictionary containing the imputation parameters. 
        The following parameters are supported:
        
        type (str): The type of imputation. Can be one of "drop", "impute".
        drop_columns (bool): If True, the columns with missing values greater than the threshold will be dropped.
        threshold (float): The threshold to use for dropping columns with missing values. Must be between 0 and 1.
        strategy_numerical (str): The imputation strategy to use for numerical columns. Can be one of "mean", "median", "most_frequent", "knn" or "constant".
        strategy_categorical (str): The imputation strategy to use for categorical columns. Can be one of "most_frequent", or "constant".
        fill_value (str/int/float): The (str) value to use for imputing missing values when strategy is "constant for object data types, For numerical it should be a int/float".

        outlier (dict): The dictionary containing the outlier parameters.
        The following parameters are supported:
        method (str): The method to use for removing outliers. Can be one of "zscore", "iqr", "None".
        iqr_threshold (float): The threshold to use for removing outliers. Must be greater than 0.
        Lc (float): The lower cut-off value to use for removing outliers. Must be between 0 and 1.
        Uc (float): The upper cut-off value to use for removing outliers. Must be between 0 and 1.
        cap (bool): If True, the outliers will be capped to the upper and lower bounds. If False, the outliers will be removed.
        **kwargs: Keyword arguments to pass to sklearn KNNImputer.

    Returns:
        pandas.DataFrame: The DataFrame containing the preprocessed data.
    """

    data = dataframe.copy()

    if missing['drop_columns']:
        data = drop_missing_columns(data, threshold=missing['threshold'])

    if data[label_column].isnull().sum() > 0:
        print('The Label column has missing values. so dropping the rows with missing values in label column')
        data = drop_missing_rows(data,subset=[label_column])


    # Separate columns into numerical and categorical
    numerical_columns = data.select_dtypes(exclude='object').columns
    categorical_columns = data.select_dtypes(include='object').columns

    # Remove label column from numerical and categorical columns
    if data[label_column].dtype == 'object':
        categorical_columns = categorical_columns.drop(label_column)
    else:
        numerical_columns = numerical_columns.drop(label_column)

    if missing['type'] == 'drop':
        data = drop_missing_rows(data)

    elif missing['type'] == 'impute':
        data =  imputation(data, strategy_numerical=missing['strategy_numerical'], 
                            strategy_categorical=missing['strategy_categorical'],
                            fill_value=missing['fill_value'],
                            numerical_columns=numerical_columns,
                            categorical_columns=categorical_columns, 
                            **kwargs)
    if outlier['method'] == 'None':
        pass
    elif outlier['method'] == 'zscore':
        data = remove_outliers_zscore(data, numerical_columns=numerical_columns, zscore_threshold=outlier['zscore_threshold'])
    elif outlier['method'] == 'iqr':
        data = inter_quartile_range(data, numerical_columns=numerical_columns, iqr_threshold=outlier['iqr_threshold'], Lc=outlier['Lc'], Uc=outlier['Uc'], cap=outlier['cap'])
        
    return data

#Stage 3: Data Transformation

def transformation(dataframe, label_column, type = 'ordinal',target_transform = False, cardinality = True, Cardinality_threshold = 0.3):
    """ This function is used to transform the feature data.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column.
        type (str): The type of transformation. Can be one of "ordinal", "onehot".
        target_transform (bool): If True, the label column will be transformed.
        cardinality (bool): If True, the columns with cardinality greater than the threshold will be dropped.
        Cardinality_threshold (float): The threshold to use for dropping columns with cardinality. Must be between 0 and 1.

    Returns:
        pandas.DataFrame: The DataFrame containing the transformed data.
    """
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop([label_column]) if label_column in categorical_columns else categorical_columns


    if cardinality:
        drop_columns = [x for x in categorical_columns if dataframe[x].nunique() > Cardinality_threshold*100]
        # drop_columns = [x for x in drop_columns if x not in label_column or x in categorical_columns]
        print('List of columns dropped due to high cardinality: {}'.format(drop_columns))
        dataframe.drop(drop_columns, axis=1, inplace=True)
        categorical_columns = categorical_columns.drop(drop_columns)

    if target_transform:
        dataframe, target_encoder = label_transform(dataframe, label_column)

    
    if type == 'ordinal':
        dataframe, encoder = ordinal_transform(dataframe, categorical_columns)
    elif type == 'onehot':
        dataframe, encoder = onehot_transform(dataframe, categorical_columns)
    elif type == 'None':
        print('No transformation applied on features')
    else:
        raise ValueError('The transformation type is not supported.')
    
    return dataframe, encoder

#Stage 4: Data Scaling

def scaling(dataframe, label_column, type = 'standard', target_scaling = False):
    """ This function is used to scale the feature data.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column.
        type (str): The type of scaling. Can be one of "standard", "robust", "maxabs", "power", "quantile".
        target_scaling (bool): If True, the label column will be scaled using Sklearn MinMax Scaler. Applicable only for Regression usecase.

    Returns:
        pandas.DataFrame: The DataFrame containing the scaled data.
    """
    numerical_columns = dataframe.select_dtypes(exclude='object').columns
    numerical_columns = numerical_columns.drop([label_column]) if label_column in numerical_columns else numerical_columns

    if target_scaling:
        dataframe, target_scaler = target_scale(dataframe, label_column)

    if type == 'standard':
        dataframe, scaler = standard_scale(dataframe, numerical_columns)
    elif type == 'robust':
        dataframe, scaler = robust_scale(dataframe, numerical_columns)
    elif type == 'maxabs':
        dataframe, scaler = maxabs_scale(dataframe, numerical_columns)
    elif type == 'power':
        dataframe, scaler = power_transform(dataframe, numerical_columns)
    elif type == 'quantile':
        dataframe, scaler = quantile_transform(dataframe, numerical_columns)
    elif type == 'None':
        print('No scaling applied on features')
    else:
        raise ValueError('The scaling type is not supported.')
    
    if target_scaling:
        return dataframe, scaler, target_scaler
    else:
        return dataframe, scaler

#Stage 5: Feature Engineering


#Stage 6: Feature Selection
def feature_selection(dataframe, label_column, estimator, type='rfecv', max_features=10, min_features=2, scoring= 'accuracy', 
                    cv=5, n_jobs= -1, **kwargs):
    """ This function is used to select the features.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column.
        estimator (str): The estimator to use for feature selection. Can be one of SVC(),SVR(), RandomForestClassifier(),
        RandomForestRegressor(), GradientBoostingClassifier(), GradientBoostingRegressor(), LogisticRegression(), LinearRegression(), etc..
        type (str): The type of feature selection. Can be one of "rfe", "rfecv".
        max_features (int): The maximum number of features to select.
        min_features (int): The minimum number of features to select.
        scoring (str): The scoring function to use for feature selection. Can be one of "accuracy", "f1", "precision", "recall", "roc_auc", "neg_mean_squared_error", "neg_mean_absolute_error", "neg_median_absolute_error", "r2".
        cv (int): The number of folds to use for cross-validation.
        n_jobs (int): The number of jobs to run in parallel.
        **kwargs: Keyword arguments to pass to sklearn.feature_selection.RFE or sklearn.feature_selection.RFECV.

    Returns:
        pandas.DataFrame: The DataFrame containing the selected features.
    """
    if type == 'rfe':
        df, selector = rfe(dataframe, label_column, estimator, n_features_to_select=max_features, step=1, verbose=0, **kwargs)
    elif type == 'rfecv':
        df, selector = rfecv(dataframe, label_column, estimator, scoring, cv, n_jobs, **kwargs)
    elif type == 'None':
        print('No feature selection applied')
        return None
    else:
        raise ValueError('The feature selection type is not supported.')
    
    return df, selector

#Stage 7: Model Training

def training(dataframe, label_column, model_name, problem_type, target_scaler=None, test_split =0.15, hypertune=True, n_epochs = 100):
    """ This function is used to train the model.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the label column.
        model_name (str): The name of the model. Can be one of "LinearRegression", "BayesianRidge", "RandomForestRegressor", "XGBRegressor",
                            "GradientBoostingRegressor", "SVR", "DecisionTreeRegressor", "KNeighborsRegressor", "LogisticRegression", 
                            "RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier", "SVC", "DecisionTreeClassifier", 
                            "KNeighborsClassifier", "GaussianNB".

        problem_type (str): The type of problem. Can be one of "Regression", "Classification".
        target_scaler : The scaler object name used for scaling the label column dduring scaling phase. Applicable only for Regression usecase. 
        test_split (float): The percentage of data to use for testing. it should be 0 to 1.
        hypertune (bool): If True, the model will be hypertuned using Optuna.
        n_epochs (int): The number of epochs to use for hypertuning.

    Returns:
        sklearn.pipeline.Pipeline: The pipeline containing the trained model.
    """
    # Split the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(label_column, axis=1), dataframe[label_column], 
                                                        test_size=test_split, random_state=42)

    # with mlflow.start_run(run_name=datetime.now().strftime('%Y-%m-%d_%H:%M:%S')) as run:
    #     run_id = run.info.run_id
        # print('MLflow Run ID: {}'.format(run_id))
        # client = MlflowClient()
        # client.set_tag(run_id, "mlflow.note.content", f"ML model: {model_name}")

        # tags = {"Application": "AutoPilotML"}
        # mlflow.set_tags(tags)

    # Log the model
    if "Xgboost" in model_name:
        mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True, log_models=True, 
        disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False,log_post_training_metrics=True
        )
    else:
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True, 
        disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False,log_post_training_metrics=True
        )


    if hypertune:

        def objective(trial):
            # best_accuracy = float('inf') if problem_type == 'regression' else 0.0
            # best_model = None

            tuner_params = tuner(model_name, trial)

            model, y_pred = model_fitting(x_train, x_test, y_train, y_test, model_name= model_name, params= tuner_params)
            
            if problem_type == 'Regression':
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1,1))
                accuracy = round(mean_squared_error(y_test, y_pred, squared=False),4)
            else:
                accuracy = round(accuracy_score(y_test, y_pred),4)

            return accuracy

        study = optuna.create_study(direction='minimize' if problem_type == 'Regression' else 'maximize')
        study.optimize(objective, n_trials=n_epochs)

        best_params = study.best_params

        print('Best Score: {}'.format(study.best_value))
        print('Best Parameters: {}'.format(study.best_params))

        model, y_pred = model_fitting(x_train, x_test, y_train, y_test, model_name= model_name, params= best_params)

    else:
        model, y_pred = model_fitting(x_train, x_test, y_train, y_test, model_name= model_name)

        if problem_type == 'Regression':
            if target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred.reshape(-1,1))
            accuracy = round(mean_squared_error(y_test, y_pred, squared=False),4)
        else:
            accuracy = round(accuracy_score(y_test, y_pred),4)
        
        print('Accuracy: {}'.format(accuracy))

    return model
























