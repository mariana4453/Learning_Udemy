import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./additional_data/Advertising.csv')

df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
# print(df.head())

# sns.scatterplot(data=df, x='total_spend', y='sales')
# plots regression line
# sns.regplot(data=df, x='total_spend', y='sales')
# plt.show()

# y = mx + b - best fitting line // y = B1x + B0
X = df['total_spend']
y = df['sales']

poly = np.polyfit(X, y, deg=1)
# print(poly)

potential_spend = np.linspace(0, 500, 100)
predicted_sales = 0.04868788 * potential_spend + 4.24302822
# print(predicted_sales)

# sns.scatterplot(data=df, x='total_spend', y='sales')
# plt.plot(potential_spend, predicted_sales, color='red')
# plt.show()

spend = 200
predicted_sales_200 = 0.04868788 * spend + 4.24302822
# print(predicted_sales_200)

### ### ### ###
# example 2 : y = B3x**3 + B2*x**2 + B1x + B0
poly = np.polyfit(X, y, deg=3)
print(poly)

pot_spend = np.linspace(0, 500, 100)
pred_sales = 3.07615033e-07 * pot_spend**3 + -1.89392449e-04 *pot_spend**2 + 8.20886302e-02 * pot_spend + 2.70495053e+00

sns.scatterplot(data=df, x='total_spend', y='sales')
plt.plot(pot_spend, pred_sales, color='red')
plt.show()
