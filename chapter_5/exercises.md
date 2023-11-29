# Chapter 5 Exercises
Below are my solutions to the exercises presented at the end of chapter 5 of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.

### 1. What is the fundamental idea behind support vector machines?
The idea befind support vector machines is to fit a decision boundary in between classes that is as large as possible, allowing for some "margin violations" to maintain a more flexible model. (Soft margin classification).

### 2. What is a support vector?
Instance that determine the edge of the "street"

### 3. Why is it important to scale the inputs when using SVMs?
SVMs are sensitive to feature scale. Without scaling, the decision boundary may not look as good as it should, as the SVM will tend to neglect small features.

### 4. Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?
An SVM classifier can output a confidence score based on the instances distance from the decision boundary. These cannot be directly converted into an estimation.

### 5. How can you choose between LinearSVC, SVC, and SGDClassifier?
The SVC class supports the kernel task, which allows it to handle nonlinear tasks. However, SVC does not scale well with many instances. LinearSVC uses an optimized version of linear SVMs. SGDClassifier uses stochastic gradient descent.

### 6. Say you've trained an SVM classifier with an RBF kernel, but it seems to underfit the training set. Should you increase or decrease gamma? What about C?
 Increase gamma or C, or both.

### 7. What does it mean for a model to be ϵ-insensitive?
If you add instances to a SVM regression model, the model will not be affected at all.

### 8. What is the point of using the kernel trick?
The kernel trick allows a nonlinear SVM to be trained. It finds the optimal nonlinear model without needing to change the inputs at all.

### 9. Train a LinearSVC on a linearly separable dataset. Then train a SVC and a SGDClassifier on the same dataset. See if you can get them to produce roughly the same model.


### 10. Train an SVM classifier on the wine dataset, which you can load using sklearn.datasets.load_wine(). This dataset contains the chemical analyses of 178 wine samples produced by 3 different cultivators: the goal is to train a classification model capable of predicting the cultivator based on the wine’s chemical analysis. Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all three classes. What accuracy can you reach?


### 11. Train and fine-tune an SVM regressor on the California housing dataset. You can use the original dataset rather than the tweaked version we used in Chapter 2, which you can load using sklearn.datasets.fetch_california_housing(). The targets represent hundreds of thousands of dollars. Since there are over 20,000 instances, SVMs can be slow, so for hyperparameter tuning you should use far fewer instances (e.g., 2,000) to test many more hyperparameter combinations. What is your best model’s RMSE?
