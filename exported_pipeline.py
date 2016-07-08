import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)

exported_pipeline = Pipeline([
    ("PolynomialFeatures", PolynomialFeatures(interaction_only=False, degree=2, include_bias=False)),
    ("ExtraTreesClassifier", ExtraTreesClassifier(max_features=0.93, n_estimators=500, criterion="gini", min_weight_fraction_leaf=0.5))
])

exported_pipeline.fit(tpot_data.loc[training_indices].drop('class', axis=1).values,
                      tpot_data.loc[training_indices, 'class'].values)
results = exported_pipeline.predict(tpot_data.loc[testing_indices].drop('class', axis=1))
