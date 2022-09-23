import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./additional_data/Advertising.csv")
# print(df.head())

X = df.drop('sales', axis=1)
y = df['sales']

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# degree - what order should be polynomial, ex. 2 - squared
polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
# no need to split into test and learning set
polynomial_converter.fit(X)
poly_features = polynomial_converter.transform(X)
# polynomial_converter.fit_transform(X) - two steps in one


# same test size and random state as in linear regression
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)


model = LinearRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
# print(model.coef_)    - should be 9 coef


MAE = mean_absolute_error(y_test, test_predictions)
MSE = mean_squared_error(y_test, test_predictions)
RMSE = np.sqrt(MSE)
# print(MAE)
# print(RMSE)


# creating loop to check best model degree
# create the different order poly
# split poly feat train/test
# fit on train
# strove/ save the rmse for BOTH the train and test
# PLOT the results (error vs poly order)

train_rmse_errors = []
test_rmse_errors = []

for d in range(1, 10):
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)

# print(train_rmse_errors)
# print(test_rmse_errors)

plt.plot(range(1, 6), train_rmse_errors[:5], label='TRAIN RMSE')
plt.plot(range(1, 6), test_rmse_errors[:5], label='TEST RMSE')

plt.legend()
plt.xlabel('Degree of Poly')
plt.ylabel('RMSE')

# plt.show()

# final model
final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)
final_model = LinearRegression()

full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X, y)


from joblib import dump, load
# dump(final_model, "final_poly_model.joblib")
# dump(final_poly_converter, 'final_converter.joblib')
#
# loaded_converter = load('final_converter.joblib')
# loaded_model = load('final_poly_model.joblib')
#
# campaign = [[149, 22, 12]]
# transformed_data = loaded_converter.fit_transform(campaign)
# predicted_sales = loaded_model.predict(transformed_data)
# print(predicted_sales)
