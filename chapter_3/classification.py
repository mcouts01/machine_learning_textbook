from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

x = mnist['data']
y = mnist['target']

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    
x_train, x_test, y_train, y_test = x[:60_000], x[60_000:], y[:60_000], y[60_000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

dummy_clf = DummyClassifier()
dummy_clf.fit(x_train, y_train_5)
# print(any(dummy_clf.predict(x_train)))

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_train_pred)

# plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
# plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
# plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="threshold")

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()



# cm = confusion_matrix(y_train_5, y_train_pred)
# print(cm)

# print(precision_score(y_train_5, y_train_pred)) # 0.8370879772350012
# print(recall_score(y_train_5, y_train_pred)) # 0.6511713705958311

#               correct | incorrect
#  non-5 images[[53892   687]
#      5 images[ 1891  3530]]

# print(cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy'))

# print(cross_val_score(dummy_clf, x_train, y_train_5, cv=3, scoring='accuracy'))