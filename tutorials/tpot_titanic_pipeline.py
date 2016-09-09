import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import PolynomialFeatures

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Use Scikit-learn's PolynomialFeatures to construct new features from the existing feature set
training_features = result1.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0 and len(training_features.columns.values) <= 700:
    # The feature constructor must be fit on only the training data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(training_features.values.astype(np.float64))
    constructed_features = poly.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result1 = pd.DataFrame(data=constructed_features)
    result1['class'] = result1['class'].values
else:
    result1 = result1.copy()

result2 = result1.copy()
# Perform classification with an Ada Boost classifier
adab2 = AdaBoostClassifier(learning_rate=0.15, n_estimators=500, random_state=42)
adab2.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['adab2-classification'] = adab2.predict(result2.drop('class', axis=1).values)
