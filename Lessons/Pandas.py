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

# resetting index
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

#######################
# conditional formatting
df_tips = pd.read_csv("./additional_data/tips.csv")
bool_series = df_tips['total_bill'] > 40

df_tips_40 = df_tips[df_tips['total_bill'] > 40]
df_male = df_tips[df_tips['sex'] == 'Male']
# print(df_male)

first = df_tips['total_bill'] > 30
second = df_tips['sex'] == 'Male'
df_two_cond = df_tips[(df_tips['total_bill'] > 30) & (df_tips['sex'] == 'Male')]
# print(df_tips[first & second])

# filtering upon list
options = ['Sat', 'Sun']
days = df_tips['day'].isin(options)
# print(df_tips[df_tips['day'].isin(['Sat', 'Sun'])])

#######################
# .apply()
def last_four(number):
    return str(number)[:4]
# print(df['CC Number'].apply(last_four))

# print(df['total_bill'].mean())
def yelp(price):
    if price < 10:
        return '$'
    elif price >= 10 and price < 30:
        return '$$'
    else:
        return '$$$'

df_tips['yelp'] = df_tips['total_bill'].apply(yelp)
# print(df_tips.head())

# lambda
# df_tips['total_bill'] = df_tips['total_bill'].apply(lambda num: num * 2)

def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return "Generous"
    else:
        return "Other"

# print(quality(16.99, 1.01))
# axis = 1 --> for columns, lambda df --> df name of dataframe
df_tips['Quality'] = df_tips[['total_bill', 'tip']].apply(lambda df_tips: quality(df_tips['total_bill'], df_tips['tip']), axis=1)

# vectorize - faster
df_tips['Quality'] = np.vectorize(quality)(df_tips['total_bill'], df_tips['tip'])
# print(df_tips['Quality'])


import timeit
setup = """
import numpy as np
import pandas as pd
df_tips = pd.read_csv('./additional_data/tips.csv')

def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return "Generous"
    else:
        return "Other"
"""

stmt_one = """
df_tips['Quality'] = df_tips[['total_bill', 'tip']].apply(lambda df_tips: quality(df_tips['total_bill'], df_tips['tip']), axis=1)
"""
stmt_two = """df_tips['Quality'] = np.vectorize(quality)(df_tips['total_bill'], df_tips['tip'])
"""
# print(timeit.timeit(setup=setup, stmt=stmt_one, number=1000))
# print(timeit.timeit(setup=setup, stmt=stmt_two, number=1000))

#######################
# sorting
df = pd.read_csv("./additional_data/tips.csv")
# print(df.sort_values('tip', ascending=False))

# by multiple values
# print(df.sort_values(['tip', 'size']))

# same for min()
# print(df['total_bill'].max())
# print(df['total_bill'].idxmax())
# print(df.iloc[170])

# print(df.corr())

# counts
# print(df['day'].value_counts())
# print(df['day'].unique())
# print(df['day'].nunique())


# replacing values
# df['sex'].replace(to_replace=['Female', 'Male'], value=['F', 'M'], inplace=True)
# print(df['sex'])

# mymap = {'Female': 'F',
#          'Male': 'M'}
# df['sex'] = df['sex'].map(mymap)
# print(df['sex'])

# print(df.duplicated())

# simple_df = pd.DataFrame([1, 2, 2], ['a', 'b', 'c'])
# print(simple_df)
# print(simple_df.duplicated())

# df_filtered = df[df['total_bill'].between(10, 20, inclusive=True)]
# print(df_filtered)

# amount of largest values
# print(df.nlargest(2, 'tip'))
# print(df.sort_values('tip', ascending=False).iloc[:2])

# sampling
# print(df.sample(5))
# print(df.sample(frac=0.1))  # samples 10% of dataframe