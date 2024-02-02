import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X, Y)

print(reg.predict([[6.5]]))

plt.scatter(X, Y, color = 'red')
plt.plot(X, reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

preg = PolynomialFeatures(degree=4)
xpoly = preg.fit_transform(X)
reg2 = LinearRegression()
reg2.fit(xpoly, Y)

print(reg2.predict(preg.fit_transform([[6.5]])))

plt.scatter(X, Y, color = 'red')
plt.plot(X, reg2.predict(preg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, reg2.predict(preg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()