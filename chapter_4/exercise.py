import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Compute gradients
def compute_gradients(X, y_true, y_pred):
    return np.dot(X.T, (y_pred - y_true)) / y_true.shape[0]

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=42)

# Add intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize parameters
num_features = X_train.shape[1]
num_classes = y_onehot.shape[1]
weights = np.random.rand(num_features, num_classes)

# Training settings
num_epochs = 500
learning_rate = 0.1

best_loss = np.inf

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    logits = np.dot(X_train, weights)
    y_pred = softmax(logits)

    # Compute loss
    loss = cross_entropy_loss(y_pred, y_train)

    # Backward pass
    gradients = compute_gradients(X_train, y_train, y_pred)

    # Update weights
    weights -= learning_rate * gradients

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    if loss < best_loss:
        best_loss = loss
    else:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print("Early stopping")
        break
    

# Evaluate the model
logits_test = np.dot(X_test, weights)
y_pred_test = softmax(logits_test)
test_loss = cross_entropy_loss(y_pred_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
