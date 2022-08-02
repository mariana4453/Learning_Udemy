import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/index.html

df = pd.read_csv('./additional_data/dm_office_sales.csv')
# print(df.head())

# SCATTER PLOT
# possibly to define the figure size
# plt.figure(figsize=(12, 4), dpi=100)

# hue - color the points by some column, if categorical data - uses different colors,
# if continues data - then color range
# sns.scatterplot(x='salary', y='sales', data=df, hue='level of education')

# color pallets - https://matplotlib.org/stable/tutorials/colors/colormaps.html
# sns.scatterplot(x='salary', y='sales', data=df, hue='salary',
#                 palette='Dark2')

# sns.scatterplot(x='salary', y='sales', data=df, size='salary')

# alpha - transparency, s - size of a bubble
# sns.scatterplot(x='salary', y='sales', data=df, s=150, alpha=0.8)

# sns.scatterplot(x='salary', y='sales', data=df, s=150, style='level of education',
#                 hue='level of education')
# plt.show()
