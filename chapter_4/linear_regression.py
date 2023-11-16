import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 100
X = 2 * np.random.rand(n, 1)
y = 4 + 3 * X + np.random.randn(n, 1)

plt.xlabel('theta0', x=0.2)
plt.ylabel('theta1', rotation=0, y=0.25)
plt.axis([2, 5, 2, 5])
plt.grid(True)

# # # Compute normal equation
# # X_b = add_dummy_feature(X)
# # theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
# # print(theta_best) # [[4.21509616] [2.77011339]]

# # lin_reg = LinearRegression()
# # lin_reg.fit(X, y)
# # print(lin_reg.intercept_, lin_reg.coef_) # [4.21509616] [[2.77011339]]

# # # Make predictions using theta_best
# # X_new = np.array([[0], [2]])
# # X_new_b = add_dummy_feature(X_new)
# # y_predict = lin_reg.predict(X_new)

def minibatch_gd(X, y, learning_rate=0.1, n_epochs=30, batch_size=3):
    m = len(X)
    theta = np.random.randn(2, 1) # random initialization
    theta_path = [theta]
    
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            gradients = 2 / batch_size * xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradients
            theta_path.append(theta)
    
    return theta, np.array(theta_path)

def batch_gd(X, y, n_epochs=100):
    m = len(X)
    eta = 0.1 # learning rate
    theta = np.random.randn(2, 1) # random initialization
    theta_path = [theta]
    
    for epoch in range(n_epochs):
        gradients = 2 / m * X.T @ (X @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)
    
    return theta, np.array(theta_path)

def stochastic_gd(X, y, n_epochs=100):
    t0 = 10
    t1 = 50
    learning_schedule = lambda t: t0 / (t + t1)
        
    m = len(X)
    theta = np.random.randn(2, 1) # random initialization
    theta_path = [theta]
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = 2 * xi.T @ (xi @ theta - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
            theta_path.append(theta)
    
    return theta, np.array(theta_path)

X_b = add_dummy_feature(X)
theta_mb, theta_path_mb = minibatch_gd(X_b, y)
theta_b, theta_path_b = batch_gd(X_b, y)
theta_s, theta_path_s = stochastic_gd(X_b, y)

# print(theta_path[:, 1])
plt.plot(theta_path_s[:, 0], theta_path_s[:, 1], 'g-', label='Stochastic')
plt.plot(theta_path_mb[:, 0], theta_path_mb[:, 1], 'b-', label='Mini-batch', linewidth=2)
plt.plot(theta_path_b[:, 0], theta_path_b[:, 1], 'r-', label='Batch', linewidth=2)
plt.legend(loc='upper left')
plt.show()