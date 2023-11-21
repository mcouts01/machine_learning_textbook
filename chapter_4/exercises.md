# Chapter 4 Exercises
Below are my solutions to the exercises presented at the end of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.

### 1. Which linear regression training algorithm can you use if you have a training set with millions of features? 
Closed-from training algorithms become very slow as the number of features grows. Gradient descent can better handle a large number of features and should be chosen to train on a training set with millions of features.

### 2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it? 
Gradient descent algorithms will suffer from a training set containing features of several different scales. When the features are scaled differently, the shape of the cost function is elongated, meaning it will take longer for GD to converge. To solve this, scale the data before training.


### 3. Can gradient descent get stuck in a local minimum when training a logistic regression model?
No. The logistic regression is a convex function. Therefore, there are no local minima.

### 4. Do all gradient descent algorithms lead to the same model, provided you let them run long enough?
Not necessarily. Batch and mini-batch gradient descent algorithms can gets stuck in local minima, meaning they will have found a model different than the optimal model. Also, unless you slowly decrement the learning rate, stochastic and mini-batch will never settle at the global minimum.

### 5. Suppose you use batch gradient descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this? 
Your learning rate could be too high, causing the algorithm to diverge. This is definitely the case if the training error is also increasing consistently. If the training error is fine, your model has overfitted the training data.

### 6. Is it a good idea to stop mini-batch gradient descent immediately when the validation error goes up? 
No. Due to the random nature of mini-batch gradient descent, the validation error is bound to go up. Stopping immediately when the validation error goes up would probably stop the model prematurely.

### 7. Which gradient descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well? 
Stochastic gradient descent will reach the vicininty of the optimal solution the fastest. Batch gradient will converge to the optimal solution. You can make stochastic and mini-batch gradient descent converge by decreasing the learning rate over time. 

### 8. Suppose you are using polynomial regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this? 
If there is a large gap between the training error and the validation error, you have overfit the training data. To combat this, you could limit the degress of freedom of the model. Since we're dealing with a polynomial regression here, this would mean restricting the number of polynomial degrees.

### 9. Suppose you are using ridge regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it? 
The model suffers from high bias - an assumption about the model is wrong. To reduce bias, we need to increase the model complexity. In this case, we should lower the intensity of regularization by lower alpha.

### 10. Why would you want to use: 
#### Ridge regression instead of plain linear regression (i.e., without any regularization)? 
Regularization helps restrict model complexity.  This important in reducing reducing variance which leads to overfitting of the training data.

#### Lasso instead of ridge regression? 
The lasso regression tends to set the weights of the least important features to 0. It automatically performs feature selection.

#### Elastic net instead of lasso regression? 
Elastic net is the middleground between ridge regression and lasso regression. It is preferred over lasso regression because the lasso regression and behave erratically at times.

### 11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two logistic regression classifiers or one softmax regression classifier?
You should implement two logistic regression classifiers because you are assigning classes from one of two categories. Each category has 2 classes and thus one logisitic regression classifier per category will sucessfully handle classification. In other words, the classes are not exclusive.

### 12. Implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn, only NumPy. Use it on a classification task such as the iris dataset.


Exercises from: Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (pp. 293-295). O'Reilly Media. Kindle Edition. 