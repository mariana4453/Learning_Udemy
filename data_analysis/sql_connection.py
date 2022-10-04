import sys
import pandas as pd
from sqlalchemy import create_engine

# DEFINE THE DATABASE CREDENTIALS
user = 'student'
password = 'Lamerere2.'
host = '127.0.0.1'
port = 3306
database = 'Learning'


def datastore_connection():
    # todo add json configuration
    db = "mysql+pymysql://{user}:{pwd}@{host}:{port}/{db_name}".format(user=user,
                                                                       pwd=password, host=host, port=port, db_name=database)
    try:
        engine = create_engine(db)
    except Exception:
        print("Connection could not be made due to the following error: \n", Exception)
        sys.exit(1)
    finally:
        print('Preparing local datastore connection ... done.\n')
        return engine


def datastore_query(sql_engine, query):
    try:
        df = pd.read_sql_query(sql=query, con=sql_engine)
    except Exception:
        print("Connection could not be made due to the following error: \n", Exception)
        sys.exit(1)
    finally:
        print("Datastore query done...\n")
        return df


def datastore_store(dataframe, table_name, sql_engine, index_column):
    try:
        dataframe.to_sql(name=table_name, con=sql_engine, if_exists='replace', index_label=index_column, index=False)
    except Exception:
        print("Table couldn't be stored due to error:", Exception)
        sys.exit(1)
    finally:
        print('Table saved to datastore...\n')
        return 0




