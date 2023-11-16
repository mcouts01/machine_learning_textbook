from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

mnist = fetch_openml('mnist_784', as_frame=False)

x = mnist['data']
y = mnist['target']
    
x_train, x_test, y_train, y_test = x[:60_000], x[60_000:], y[:60_000], y[60_000:]

parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance']}

knn_classifier = KNeighborsClassifier()
gs_knn_classifier = GridSearchCV(knn_classifier, parameters, scoring='accuracy', cv=3)
gs_knn_classifier.fit(x_train, y_train)

# print(knn_classifier.predict(x_train[:10]))
# print(y_train[:10])

print(cross_val_score(gs_knn_classifier, x_train, y_train, cv=3, scoring='accuracy'))