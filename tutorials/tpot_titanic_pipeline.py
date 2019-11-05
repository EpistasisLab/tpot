import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_classes, testing_classes = \
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = RandomForestClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=9)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
