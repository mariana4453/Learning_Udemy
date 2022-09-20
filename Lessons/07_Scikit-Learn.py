import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./additional_data/Advertising.csv')
# print(df.head())
# print(df.info())


# visualizing
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
#
# axes[0].plot(df['TV'], df['sales'], 'o')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV spend")
#
# axes[1].plot(df['radio'], df['sales'], 'o')
# axes[1].set_ylabel("Sales")
# axes[1].set_title("Radio spend")
#
# axes[2].plot(df['newspaper'], df['sales'], 'o')
# axes[2].set_ylabel("Sales")
# axes[2].set_title("Newspaper spend")
#
# plt.show()


# separate into features and labels
X = df.drop('sales', axis=1)
y = df['sales']


from sklearn.model_selection import train_test_split

# test_size - what % of the data should go to the test set
# random_state - similar to random seed in np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# print(X_train) - random Shaffling
# print(y_train)

# creating a model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# predictions for the model
test_predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(df['sales'].mean())

# sns.histplot(data=df, x='sales')
# plt.show()

# mean that error is around 10% compared to mean value - 14.0225
m_abs_error = mean_absolute_error(y_test, test_predictions)
# print(m_abs_error)

# smth like standard deviation
m_sq_error = np.sqrt(mean_squared_error(y_test, test_predictions))
# print(m_sq_error)


# Residuals
test_residuals = y_test - test_predictions
# print(test_residuals)

# residual plot - should be random (with no evident pattern)
# sns.scatterplot(x=y_test, y=test_residuals)
# plt.axhline(y=0, color='r', ls='--')

# sns.displot(test_residuals, bins=25, kde=True)
# plt.show()

import scipy as sp
# Create a figure and axis to plot on
# fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
# probplot returns the raw values if needed
# we just want to see the plot, so we assign these values to -- >
# sp.stats.probplot(test_residuals, plot=ax)
# plt.show()


# Model Deployment - loading and saving the model for future use
final_model = LinearRegression()
# fit all the data
final_model.fit(X, y) # full dataset
# model coefficients:
# [ 0.04576465  0.18853002 -0.00103749] - 1st for TV, 2nd - for radio, 3rd - for newspaper
# coeff means --> increase of TV for one unit means increase of sales for 0.045
# print(final_model.coef_)

y_hat = final_model.predict(X)
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))

# comparing liner-r-predictions vs real dataset
# axes[0].plot(df['TV'], df['sales'], 'o')
# axes[0].plot(df['TV'], y_hat, 'o', color='red')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV spend")
#
# axes[1].plot(df['radio'], df['sales'], 'o')
# axes[1].plot(df['radio'], y_hat, 'o', color='red')
# axes[1].set_ylabel("Sales")
# axes[1].set_title("Radio spend")
#
# axes[2].plot(df['newspaper'], df['sales'], 'o')
# axes[2].plot(df['newspaper'], y_hat, 'o', color='red')
# axes[2].set_ylabel("Sales")
# axes[2].set_title("Newspaper spend")
#
# plt.show()

from joblib import dump, load
# dump(final_model, 'final_sales_model.joblib')
loaded_model = load('final_sales_model.joblib')
# print(loaded_model.coef_)

print(X.shape)
# imaginary campaign - 149 TV, 22 radio, 12 newspaper
campaign = [[149, 22, 12]]
print(loaded_model.predict(campaign))
