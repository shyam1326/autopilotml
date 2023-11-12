<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="1280"
        src="images/autopilotml.png"
    </a>
  </p>


[![version](https://badge.fury.io/py/autopilotml.svg)](https://badge.fury.io/py/autopilotml)
<a href="https://pepy.tech/project/autopilotml"><img src="https://pepy.tech/badge/autopilotml" alt="total autopilotml downloads"></a>
[![license](https://img.shields.io/pypi/l/autopilotml)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shyam1326/autopilotml/blob/main/autopilotml/research/autopilotml_examples.ipynb)


</div>


# Autopilotml
> Automated machine learning library for analytics

## Installation

- `pip install autopilotml`

## Usage

### Load data

```python
from autopilotml import load_data, load_database

# For csv files
df = load_data(path = "dataset/titanic_train.csv", csv=True, **kwargs)

# For excel notebook
df = load_data(path = "dataset/titanic_train.xlsx", excel=True, **kwargs)

# To Load data from Database

# This framework supports sqlite, 'mysql', 'postgres', 'MongoDB'
df = load_database(database_type='sqlite', sqlite_db_path = 'database.db', query='select * from employee_table')
```

### Data Preprocessing

```python
from autopilotml import preprocessing

# If changing any values in the dictionary, whole dictionary has to be provided.

df = preprocessing(dataframe=df, label_column='Survived',
                                missing={
                                    'type':'impute',
                                    'drop_columns': False, 
                                    'threshold': 0.25, 
                                    'strategy_numerical': 'knn',
                                    'strategy_categorical': 'most_frequent',
                                    'fill_value': None},
                                outlier={
                                    'method': 'None',
                                    'zscore_threshold': 3,
                                    'iqr_threshold': 1.5,
                                    'Lc': 0.05, 
                                    'Uc': 0.95,
                                    'cap': False})
```

### Data Transformation

```python
from autopilotml import transformation

# If the target_transform is true, then the function  return 3 objects, (e.g) dataframe, feature encoder and target encoder
# else it will return 2 objects dataframe and feature encoder
df, encoder = transformation(dataframe=df,
                                label_column='Survived', 
                                type = 'ordinal',
                                target_transform = False, 
                                cardinality = True, 
                                Cardinality_threshold = 0.3)
```

### Scaling

```python
# Here if target_scaling = True only applicable for regression then it will return 3 objects dataframe, feature scaler and target scaler

from autopilotml import scaling

df, scaler = scaling(df, label_column= 'Survived', type = 'standard', target_scaling = False)
```

### Feature Selecction

```python
from autopilotml import feature_selection

df, selector = feature_selection(dataframe=df, label_column='Survived', 
                                estimator='RandomForestClassifier',           
                                type='rfe', max_features=10, 
                                min_features=2, scoring= 'accuracy', 
                                cv=5)
```

### Model Training

```python
from autopilotml import training

model = training(dataframe=df, label_column='Survived', model_name='SVC', problem_type='Classification', 
                target_scaler=None, test_split =0.15, hypertune=True, n_epochs=100)
```

### MLFlow - Track the Model Training and model Parameters

```python
!mlflow ui
```