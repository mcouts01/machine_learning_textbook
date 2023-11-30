# from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# iris = load_iris(as_frame=True)
# X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
# y_iris = iris.target

# tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# tree_clf.fit(X_iris, y_iris)
# export_graphviz(
#     tree_clf,
#     out_file="iris_tree.dot",
#     feature_names=["petal length (cm)", "petal width (cm)"],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )

# Regularization
# from sklearn.datasets import make_moons

# X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# tree_clf1 = DecisionTreeClassifier(random_state=42)
# tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
# tree_clf1.fit(X_moons, y_moons)
# tree_clf2.fit(X_moons, y_moons)

# X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)
# print(f"Unregularized classifier score: ", tree_clf1.score(X_moons_test, y_moons_test)) # 0.898
# print(f"Regularized classifier score: ", tree_clf2.score(X_moons_test, y_moons_test)) # 0.92

# Regularized model performs better

# Regression Tasks
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
export_graphviz(
    tree_reg,
    out_file="tree_reg.dot",
    rounded=True,
    filled=True
)