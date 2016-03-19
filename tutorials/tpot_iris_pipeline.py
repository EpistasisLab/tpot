import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

# Perform classification with a C-support vector classifier
svc1 = SVC(C=0.1)
svc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['svc1-classification'] = svc1.predict(result1.drop('class', axis=1).values)

# Subset the data columns
subset_df1 = result1[sorted(result1.columns.values)[4042:5640]]
subset_df2 = result1[[column for column in ['class'] if column not in subset_df1.columns.values]]
result2 = subset_df1.join(subset_df2)

# Perform classification with a k-nearest neighbor classifier
knnc3 = KNeighborsClassifier(n_neighbors=min(131, len(training_indices)))
knnc3.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)
result3 = result2
result3['knnc3-classification'] = knnc3.predict(result3.drop('class', axis=1).values)
