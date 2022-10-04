import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# local imports
from data_analysis import sql_connection

# sql_engine
sql_engine = sql_connection.datastore_connection()

######################################################
# loading csv to datastore
df_netflix = pd.read_csv('./data_analysis/netflix_titles.csv')
# sql_connection.datastore_store(df_netflix, 'netflix_series', sql_engine, df_netflix.columns[0])

# null values
nulls = df_netflix.isnull().sum()
# print(nulls)

df_netflix['date_added'] = pd.to_datetime(df_netflix['date_added'])
df_netflix['year_added'] = pd.DatetimeIndex(df_netflix['date_added']).year
df_netflix['month_added'] = df_netflix['date_added'].dt.month

# print(df_netflix['date_added'].head())
# print(df_netflix['year_added'].head())
# print(df_netflix['month_added'].head())

# dropping columns where is no year_added
df_netflix = df_netflix.dropna(subset=['year_added'])
df_netflix['year_added'] = df_netflix['year_added'].astype(int)
# print(df_netflix['year_added'].head())


# plotting
# Type vs Year added
# sns.countplot(data=df_netflix, y=df_netflix['type'], palette='Set2')
# sns.countplot(data=df_netflix, x=df_netflix['type'], palette='Set2')
# plt.show()


# order by category - year
# plt.figure(figsize=(12, 6))
sns.set_style(style="darkgrid")
#
# sns.countplot(data=df_netflix, x=df_netflix['year_added'],
#               order=df_netflix['year_added'].value_counts().index[:10])
# plt.title('Which year has more movies (top 10)')
# plt.show()



# x = df_netflix['release_year']
# y = df_netflix['year_added']
#
# plt.figure(figsize=(12, 7))
# plt.scatter(x, y)
#
# plt.title("Time between movie/show release and netflix release")
# plt.xlabel("Release Year")
# plt.ylabel("Year added")
# plt.show()


# grouping by year added
df_years = df_netflix[['show_id', 'year_added']]
df_years = df_years.groupby(df_years['year_added']).count()

plt.figure(figsize=(12, 7))
x = df_years.index
y = df_years['show_id']

plt.scatter(x, y)

plt.xlabel('-- added --')
plt.ylabel('-- number of contents added --')
plt.title('How of Increasing in number of contents added by netflix each year')

plt.show()




