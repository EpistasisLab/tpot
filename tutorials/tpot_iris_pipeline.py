import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_classes, testing_classes = \
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.8500000000000001),
    DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=4, min_samples_split=9)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
