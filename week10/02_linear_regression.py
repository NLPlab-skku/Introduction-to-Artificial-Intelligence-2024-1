############################
# O'reilly
# Hands-On Machine Learning
############################

# Linear Regression

import numpy as np

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + Gaussian noise

#print(X)

X_b = np.c_[np.ones((100,1)),X] #1을 더하는 이유는 x0(바이어스)
print(X_b)
input()

X_new = np.array([[0],[2]])
#print(X_new)

#X_new_b = np.c_[np.ones((2,1)),X_new]
#print(X_new_b)
#input()

from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X,y)

#print(X)
#print("Y")
#print(y)

print(Lin_reg.intercept_, Lin_reg.coef_)
print(Lin_reg.predict(X_new))
input()

# Batch Gradient Descent
eta = 0.1 # learning rate
n_iterations = 100
m = 100

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta -eta * gradients

print(theta)

# Stochastic Gradient Descent

n_epochs =50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients

print(theta)

# SGD regressor
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print(sgd_reg.intercept_, sgd_reg.coef_)