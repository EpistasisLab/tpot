
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=200, n_features=800,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)


# too many trees, take too much time
fa_make2 = make_pipeline(
    PolynomialFeatures(degree=2,interaction_only=False),# include_bias=False, interaction_only=False),
    ExtraTreesClassifier(max_features=1.0, n_estimators=500)#criterion="entropy", max_features=1.0, n_estimators=500)
)

#fa_make2.fit(X_train,y_train[:,0])
fa = ExtraTreesClassifier(max_features=1.0, n_estimators=100)

fa_make2 = make_pipeline(
    PolynomialFeatures(degree=2,interaction_only=False),# include_bias=False, interaction_only=False),
    ExtraTreesClassifier(max_features=1.0, n_estimators=100)#criterion="entropy", max_features=1.0, n_estimators=500)
)

fa.fit(X,y)
print('pass fa')
print('start with ployfeat')
fa_make2.fit(X,y)
print('pass fa_make')


