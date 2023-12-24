import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor_2 = LinearRegression()
regressor_2.fit(X_poly,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.show()

plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

regressor.predict([[4.5]])
regressor_2.predict(poly_reg.fit_transform([[4.5]]))