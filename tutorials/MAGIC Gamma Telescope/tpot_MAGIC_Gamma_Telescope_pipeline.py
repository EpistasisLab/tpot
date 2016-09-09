import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a gradient boosting classifier
gbc1 = GradientBoostingClassifier(learning_rate=0.49, max_features=1.0, min_weight_fraction_leaf=0.09, n_estimators=500, random_state=42)
gbc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['gbc1-classification'] = gbc1.predict(result1.drop('class', axis=1).values)
