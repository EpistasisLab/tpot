# IRIS Example

The following code illustrates the usage of TPOT with the IRIS data set. 

```python
from tpot import TPOT
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.75, test_size=0.25)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
```

Running this code should discover a pipeline that achieves ~92% testing accuracy. Note that sometimes when both `train_size` and `test_size` aren't specified in `train_test_split()` calls, the split doesn't use the entire data set, so we need to specify both.

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

After running the above code, the corresponding Python code should be exported to the `tpot_iris_pipeline.py` file and look similar to the following:

```python
import numpy as np
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
svc1.fit(result1.loc[training_indeces].drop('class', axis=1).values, result1.loc[training_indeces, 'class'].values)
result1['svc1-classification'] = svc1.predict(result1.drop('class', axis=1).values)

# Subset the data columns
subset_df1 = result1[sorted(result1.columns.values)[4042:5640]]
subset_df2 = result1[[column for column in ['class'] if column not in subset_df1.columns.values]]
result2 = subset_df1.join(subset_df2)

# Perform classification with a k-nearest neighbor classifier
knnc3 = KNeighborsClassifier(n_neighbors=min(131, len(training_indeces)))
knnc3.fit(result2.loc[training_indeces].drop('class', axis=1).values, result2.loc[training_indeces, 'class'].values)
result3 = result2
result3['knnc3-classification'] = knnc3.predict(result3.drop('class', axis=1).values)
```
