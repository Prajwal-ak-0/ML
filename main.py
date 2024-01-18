import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('Decision Tree Regression')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

print(X)