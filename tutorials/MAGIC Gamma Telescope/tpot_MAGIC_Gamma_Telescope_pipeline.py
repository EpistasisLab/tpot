import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.853347788745
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=10.0, dual=False, penalty="l2")),
    DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=5, min_samples_split=7)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
