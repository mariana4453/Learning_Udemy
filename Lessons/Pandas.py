import numpy as np
import pandas as pd

# help(pd.Series)

myindex = ['USA', 'Canada', 'Mexico']
mydata = [1776, 1867, 1821]

myser = pd.Series(data=mydata, index=myindex)
# print(myser)
# print(myser['USA'])

# from dictionary
ages = {'Sam': 5, 'Frank': 10, 'Spike': 7}
dic_ser = pd.Series(ages)
# print(dic_ser)

q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}

sales_q1 = pd.Series(q1)
sales_q2 = pd.Series(q2)

# print(sales_q1.keys()) - # returns keys
# print(sales_q1 + sales_q2)
# print(sales_q1.add(sales_q2, fill_value=0))

#################################################
# Dataframes

np.random.seed(101)
mydata = np.random.randint(0, 101, (4, 3))

myindex = ['CA', 'NY', 'AZ', 'TX']
mycolumns = ['Jan', 'Feb', 'Mar']

df_1 = pd.DataFrame(mydata, index=myindex, columns=mycolumns)
# info about dataframe
# print(df_1.info())

path = "./additional_data/tips.csv"
df = pd.read_csv(path)

# print(df.index)
# last rows
# print(df.tail())

# basic statistical info for numerical values, transpose - optional
# print(df.describe().transpose())

#######################
# Dataframe - Columns
# saves as series
total_bill = df['total_bill']

# saves as datarame
my_cols = ['total_bill', 'tip']
df_cut = df[my_cols]
df_cut_2 = df[['total_bill', 'tip']]
# print(df_cut)
# print(df_cut_2)

# new column
df['tip_percentage'] = np.round(100 * df['tip'] / df['total_bill'], 2)
df['price_per_person'] = np.round(df['total_bill'] / df['size'], 2)
# print(df.head())

# dropping column
# axis = 1 --> looks in columns
df.drop('tip_percentage', axis=1, inplace=True)
# print(df.columns)
# print(df.shape)


#######################
# Dataframe - Rows
df = df.set_index('Payment ID')

# reseting index
# df = df.reset_index()

# grab row
# iloc - based on index number, loc - based on index label
# print(df.iloc[0])
# print(df.loc['Sun2959'])

# print(df.iloc[0:4])
# print(df.loc[['Sun2959', 'Sun4608', 'Sun4458', 'Sun5260']])

# remove row
df = df.drop('Sun2959')

# insert new row
one_row = df.iloc[0]
df = df.append(one_row)
