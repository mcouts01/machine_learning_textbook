import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

train = pandas.read_csv('./dataset/titanic/train.csv')
test = pandas.read_csv('./dataset/titanic/train.csv')
x_train = train.drop(['Survived', 'Name'], axis=1)
y_train = train['Survived']
x_test = test.drop(['Survived', 'Name'], axis=1)
y_test = test['Survived']

print(x_train.head())
print(y_train.head())

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x_train, y_train)

print(cross_val_score(tree_clf, x_test, y_test, cv=3, scoring='accuracy'))