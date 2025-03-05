# Linear Regression

import numpy as np


X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + Gaussian noise
print(X)
input()

print(y)
input()

X_new = np.array([[0],[2]])
print(X_new)
input()

from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X,y)

print(Lin_reg.intercept_, Lin_reg.coef_)
input()
print(Lin_reg.predict(X_new))
input()

# Batch Gradient Descent
eta = 0.1 # learning rate
n_iterations = 100
m = 100

X_b = np.c_[np.ones((100,1)),X] #1을 더하는 이유는 x0(바이어스)
print(X_b)
input()

theta = np.random.randn(2,1)
print("before",theta)
input()

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta -eta * gradients

print("after", theta)

