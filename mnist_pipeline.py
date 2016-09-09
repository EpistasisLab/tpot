import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.01, loss="ls", max_depth=9, max_features=0.45, min_samples_leaf=9, min_samples_split=7, n_estimators=500, subsample=0.7500000000000001)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
