
import pandas as pd
import sqlite3
import pymongo
import psycopg2
from mysql import connector



def read_sqlite(query, path_to_db, **kwargs):
    """Reads data from a SQL query and returns a pandas DataFrame.

    Args:
        query (str): The SQL query to execute.
        path_to_db (sqlalchemy.engine.Connection): Path to database "database.db".
        **kwargs: Keyword arguments to pass to pandas.read_sql.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the SQL query.
    """
    try:
        conn = sqlite3.connect(path_to_db, **kwargs)
        df = pd.read_sql(query, conn, **kwargs)
        conn.close()

    except ConnectionError as conn:
        print("Could not connect to SQLite database.")
        print(conn)

    except Exception as e:
        print("An error occurred.")
        print(e)

    return df

def read_postgres(query, host, port, database_name, username=None, password=None, **kwargs):
    """Reads data from a PostgreSQL database and returns a pandas DataFrame.

    Args:
        query (str): The SQL query to execute.
        host (str): The host of the PostgreSQL database.
        port (int): The port of the PostgreSQL database.
        database_name (str): The name of the PostgreSQL database.
        username (str): The username of the PostgreSQL database. 
        password (str): The password of the PostgreSQL database. 
        **kwargs: Keyword arguments to pass to pandas.read_sql.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(host=host, port=port, database=database_name, user=username, password=password, **kwargs)
        df = pd.read_sql(query, conn)
        conn.close()

    except ConnectionError as conn:
        print("Could not connect to PostgreSQL database.")
        print(conn)

    except Exception as e:
        print("An error occurred.")
        print(e)

    return df

def read_mysql(query, host, port, database_name, username=None, password=None, **kwargs):
    """Reads data from a MySQL database and returns a pandas DataFrame.

    Args:
        query (str): The SQL query to execute.
        host (str): The host of the MySQL database.
        port (int): The port of the MySQL database.
        database_name (str): The name of the MySQL database.
        username (str): The username of the MySQL database. 
        password (str): The password of the MySQL database. 
        **kwargs: Keyword arguments to pass to pandas.read_sql.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the MySQL database.
    """
    try:
        conn = connector.connect(host=host, port=port, database=database_name, user=username, password=password, **kwargs)
        df = pd.read_sql(query, conn)
        conn.close()

    except ConnectionError as conn:
        print("Could not connect to MySQL database.")
        print(conn)

    except Exception as e:
        print("An error occurred.")
        print(e)

    return df

def read_mongo(host, port, database_name, collection_name, username=None, password=None, **kwargs):
    """Reads data from a MongoDB database and returns a pandas DataFrame.

    Args:
        host (str): The host of the MongoDB database.
        port (int): The port of the MongoDB database.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.
        username (str): The username of the MongoDB database.
        password (str): The password of the MongoDB database.
        **kwargs: Keyword arguments to pass to pandas.read_mongo.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the MongoDB database.
    """
    try:
        if username and password:
            client = pymongo.MongoClient(host, port, username=username, password=password, **kwargs)
        else:
            client = pymongo.MongoClient(host, port)

        db = client[database_name]
        collection = db[collection_name]

        records = collection.find()

    except ConnectionError as conn:
        print("Could not connect to MongoDB database.")
        print(conn)

    except Exception as e:
        print("An error occurred.")
        print(e)

    return pd.DataFrame(list(records), **kwargs)


