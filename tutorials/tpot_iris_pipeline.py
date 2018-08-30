import numpy as np

from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=None)

exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.8500000000000001),
    DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=4, min_samples_split=9)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
