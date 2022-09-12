import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/index.html

# df = pd.read_csv('./additional_data/dm_office_sales.csv')
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

# DISTRIBUTION PLOTS

# height - y axis is not informational
# sns.rugplot(x='salary', data=df, height=0.5)

# don't use DISTPLOT !! - old version
# styles - darkgrid, whitegrid, white, dark, ticks
# sns.set(style='darkgrid')

# many parameters from matplotlib are also available in sns
# sns.displot(x='salary', data=df, bins=20, color='red', edgecolor='blue',
#             linewidth=3, ls='--')

# sns.displot(x='salary', data=df, bins=20, rug=True, kde=True)
# sns.histplot(data=df, x='salary', kde=True)
# sns.kdeplot(data=df, x='salary')


# np.random.seed(42)
# sample_ages = np.random.randint(0, 100, 200)
# sample_ages = pd.DataFrame(sample_ages, columns=['age'])

# sns.rugplot(data=sample_ages, x='age')
# sns.displot(data=sample_ages, x='age', rug=True, kde=True)

# cutting start and end of the plot
# bw_adjust - adjust variance
# sns.kdeplot(data=sample_ages, x='age', clip=[0, 100], bw_adjust=0.1, shade=True)
# plt.show()


# CATEGORICAL PLOTS
# hue - adds division, countplot shows only counts
# sns.countplot(data=df, x='level of education', hue='division', palette='Set2')
# plt.ylim(100, 250)
# print(df['level of education'].value_counts())

# ci - confidence interval ( default - 95, sd - standard deviation)
# sns.barplot(data=df, x='level of education', y='salary', estimator=np.mean, ci='sd',
#             hue='division')
# plt.legend(bbox_to_anchor=(1.05, 1))
# plt.show()

# df = pd.read_csv('./additional_data/StudentsPerformance.csv')
# print(df.dtypes)

# sns.boxplot(data=df, y='math score')
# sns.boxplot(data=df, y='math score', x='parental level of education')

# sns.boxplot(data=df, y='math score', x='parental level of education', hue='test preparation course')
# plt.legend(bbox_to_anchor=(0.3, 0.2))
# sns.boxplot(data=df, x='math score', y='parental level of education', hue='test preparation course')

# sns.violinplot(data=df, x='math score', y='parental level of education')

# inner = quartile, stick(line of any instance)
# sns.violinplot(data=df, x='math score', y='parental level of education', hue='test preparation course',
#                split=True, inner=None)

# sns.violinplot(data=df, x='math score', y='parental level of education',
#                split=True, bw=0.01)

# dodge - splits into different categories (hue)
# sns.swarmplot(data=df, x='math score', y='gender', size=3, hue='test preparation course', dodge=True)

# sns.boxenplot(x='math score', data=df, y='test preparation course', hue='gender')
# plt.legend(bbox_to_anchor=(0.3, 0.2))


# COMPARISON PLOTS
# sns.jointplot(data=df, x='math score', y='reading score')

# kind --> hex, hist, scatter, kde
# sns.jointplot(data=df, x='math score', y='reading score', kind='kde')
# sns.jointplot(data=df, x='math score', y='reading score', hue='gender')
# sns.pairplot(df, hue='gender', diag_kind='hist', corner=True)


# GRIDS
# row or column
# sns.catplot(data=df, x='gender', y='math score', kind='box', row='lunch',
#             col='test preparation course')


# g = sns.PairGrid(df, hue='gender')
# g = g.map_upper(sns.scatterplot)
# g = g.map_lower(sns.kdeplot)
# g = g.map_diag(sns.histplot)
# g = g.add_legend()


# MATRIX PLOTS
df = pd.read_csv('./additional_data/country_table.csv')
df = df.set_index('Countries')

# sns.heatmap(df.drop('Life expectancy', axis=1), linewidth=0.5, annot=True,
#             cmap='viridis')

# sns.heatmap(df.drop('Life expectancy', axis=1), linewidth=0.5, annot=True,
#             cmap='viridis', center=40)

# sns.clustermap(df.drop('Life expectancy', axis=1), linewidth=0.5, annot=True,
#             cmap='viridis', col_cluster=False)
plt.show()

# todo
# nsmallest