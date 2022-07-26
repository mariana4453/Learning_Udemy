import pandas as pd
import numpy as np

# missing data
# print(np.nan)

df = pd.read_csv("./additional_data/movie_scores.csv")
# print(df.head())

# print(df.isnull())
# print(df.notnull())

# print(df['pre_movie_score'].notnull())
df_with_score = df[df['pre_movie_score'].notnull()]
# print(df_with_score)

# print(df[(df['pre_movie_score'].isnull()) & (df['first_name'].notnull())])

# DROP nan data
# print(df)
# print(df.dropna())
# at least one non-nan value
# print(df.dropna(thresh=1))

# based on columns
# print(df.dropna(axis=1))
#
# print(df.dropna(subset=['last_name']))

# FILL IN DATA
# print(df.fillna('NEW_VALUE'))

# fill in concrete column
# print(df['pre_movie_score'].fillna(0))
# print(df['pre_movie_score'].fillna(df['pre_movie_score'].mean()))

airline_tix = {'first': 100, 'business': np.nan,
               'economy-plus': 50, 'economy': 30}
ser = pd.Series(airline_tix)
# try guessing the value of missing the field
# print(ser.interpolate())

#################################################
# GROUP BY operations
df = pd.read_csv('./additional_data/mpg.csv')

# print(df.groupby('model_year').mean()['mpg'])

# multi-Index
# print(df.groupby(['model_year', 'cylinders']).mean())
# print(df.groupby(['model_year', 'cylinders']).mean().index)

# print(df.groupby('model_year').describe())

year_cyl = df.groupby(['model_year', 'cylinders']).mean()
# print(year_cyl)
# print(year_cyl.index.names)
# print(year_cyl.index.levels)

# print(year_cyl.loc[70])
# print(year_cyl.loc[[70, 82]])
# print(year_cyl.loc[(70, 4)])

#.xs
# print(year_cyl.xs(key=70, level='model_year'))    # can't take multi-keys
# print(year_cyl.xs(key=4, level='cylinders'))

# df_6_8 = df[df['cylinders'].isin([6, 8])].groupby(['model_year', 'cylinders']).mean()
# print(df_6_8)

# year_cyl_swap = year_cyl.swaplevel()
# print(year_cyl_swap)

# print(year_cyl.sort_index(level='model_year', ascending=False))

# df_agg = df.agg(['std', 'mean'])
# print(df_agg)

# df_agg_2 = df.agg({'mpg': ['max', 'mean'],
#                    'weight': ['mean', 'std']})
# print(df_agg_2)

#################################################
# combining Dataframes
# concat --> pasting 2 dataframes together
data_one = {'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']}
data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}

one = pd.DataFrame(data_one)
two = pd.DataFrame(data_two)

# axis 1 - upon columns
one_two = pd.concat([one, two], axis=1)
# print(one_two)

# repeats the index
one_two_rows = pd.concat([one, two], axis=0)
# print(one_two_rows)

# renaming the colummns
two.columns = one.columns
one_two_rows = pd.concat([one, two], axis=0)
# print(one_two_rows)


# MERGE - analoge to JOIN in SQL
registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})

df_inner_merge = pd.merge(registrations, logins, how='inner', on='name')
# print(df_inner_merge)

df_left_merge = pd.merge(left=registrations, right=logins, how='left', on='name')
# print(df_left_merge)

df_outer_merge = pd.merge(registrations, logins, how='outer', on='name')
# print(df_outer_merge)

# join on index
registrations = registrations.set_index('name')
# print(registrations)
df_index_merge = pd.merge(registrations, logins, left_index=True, right_on='name')
# print(df_index_merge)

registrations = registrations.reset_index()
registrations.columns = ['reg_name', 'reg_id']
# print(registrations)

#################################################
# TEXT METHODS
email = 'mariana@mail.com'
email = email.split('@')
# print(email)

names = pd.Series(['andrex', 'bobo', 'claire', 'david', '5'])
# print(names)

names = names.str.upper()
# print(names)
# print('4'.isdigit())
# print(names.str.isdigit())

tech_financle = ['GOOG,APPLE,AMZN', 'JPM,BAC,CG']
tickers = pd.Series(tech_financle)

tickers_1 = tickers.str.split(',').str[0]
# print(tickers_1)
tickers_2 = tickers.str.split(',', expand=True) # expands through columns
# print(tickers_2)

messy_names = pd.Series(['andrew  ', 'bo:bo', '   claire   '])
messy_names = messy_names.str.replace(":", '').str.strip().str.capitalize()
# print(messy_names)

# TIME METHODS
from datetime import datetime
my_year = 2015
my_month = 1
my_day = 1
my_hour = 2
my_min = 30
my_sec = 15

my_date = datetime(my_year, my_month, my_day)
# print(my_date)

my_datetime = datetime(my_year, my_month, my_day, my_hour, my_min, my_sec)
# print(my_datetime)
# print(my_datetime.year)

myseries = pd.Series(['Nov 3, 1990', '2000-01-01', None])
# print(myseries)

timeser = pd.to_datetime(myseries)
# print(timeser[0].year)

euro_dat = '31-12-2000'
euro_dat_con = pd.to_datetime(euro_dat)
# print(euro_dat_con)

euro_date_2 = '10-12-2000'
euro_date_2_conv = pd.to_datetime(euro_date_2, dayfirst=True)
# print(euro_date_2_conv)

style_date = '12--Dec--2000'
style_date_con = pd.to_datetime(style_date, format='%d--%b--%Y')
# print(style_date_con)

custom_date = '12th of Dec 2000'
custom_date_con = pd.to_datetime(custom_date)
# print(custom_date_con)

sales = pd.read_csv('./additional_data/RetailSales_BeerWineLiquor.csv')
# print(sales['DATE'].head())
sales['DATE'] = pd.to_datetime(sales['DATE'])
# print(sales['DATE'].head())

# PARSING DATES
sales_2 = pd.read_csv('./additional_data/RetailSales_BeerWineLiquor.csv', parse_dates=[0])
# print(sales_2.dtypes)

sales_2 = sales_2.set_index('DATE')
# print(sales_2.resample(rule='A').mean())
# print(sales_2['DATE'].dt.year)


# CSV FILES
import os
# alternative for cdw in Jupiter
# print(os.getcwd())

# df = pd.read_csv('./additional_data/example.csv', index_col=0)
# print(df)

# df.to_csv('./additional_data/example_export.csv', index=True)
# index - True --> when you want to save index (is default)


# HTML
url = "https://en.wikipedia.org/wiki/World_population"
# tables = pd.read_html(url)
# print(len(tables))

# world_pop = tables[2]
# world_pop = world_pop['World population milestones in billions [3](Worldometers estimates)']
# print(world_pop)

# url = 'https://worldpopulationreview.com/'
# tables = pd.read_html(url)

# world_population = tables[1]
# print(world_population.columns)

# EXCEL
# openpyxl - new excel formats, xlrd - older excel formats
import openpyxl
excel_file_path = './additional_data/my_excel_file.xlsx'
df = pd.read_excel(excel_file_path,  sheet_name='First_Sheet', engine='openpyxl')
# print(df)

# list sheet names in excel
wb = pd.ExcelFile(excel_file_path, engine='openpyxl')
# print(wb.sheet_names)

# read everything - all sheet
excel_sheet_dic = pd.read_excel(excel_file_path, sheet_name=None, engine='openpyxl')
# print(excel_sheet_dic)
# print(excel_sheet_dic['First_Sheet'])

our_df = excel_sheet_dic['First_Sheet']
# our_df.to_excel('example.xlsx', sheet_name='File', index=False)

# SQL
from sqlalchemy import create_engine
# create empty temporary database
# temp_db = create_engine('sqlite:///:memory:')

simple_df = pd.DataFrame(data=np.random.randint(low=0, high=100, size=(4, 4)), columns=['a', 'b', 'c', 'd'])
# print(simple_df)

# save db to database
# simple_df.to_sql(name='new_table', con=temp_db)
# simple_df.to_sql(name='new_table', con=temp_db)

# reading table
# new_df = pd.read_sql(sql='new_table', con=temp_db)
# print(new_df)

# result = pd.read_sql_query(sql='SELECT a, c FROM new_table', con=temp_db)
# print(result)

# PIVOT TABLES
df_crm = pd.read_csv('./additional_data/Sales_Funnel_CRM.csv')
# print(df_crm.head())

licenses = df_crm[['Company', 'Product', 'Licenses']]
# print(licenses)
pivot = pd.pivot(data=licenses, index='Company', columns='Product', values='Licenses')
# print(pivot)

pivot_table = pd.pivot_table(df_crm, index='Company', aggfunc='sum', values=['Licenses', 'Sale Price'])
# print(pivot_table)

# margins - Totals
df_contact = pd.pivot_table(df_crm, index=['Account Manager', 'Contact'],columns=['Product'], values=['Sale Price'], aggfunc='sum',
                            fill_value=0, margins=True)
print(df_contact)
