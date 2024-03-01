from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression

x,y = make_clasification(n_samples=100, class_sep=2, n_features=30, random_state=42)