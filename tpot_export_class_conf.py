import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(lambda X: X),
        FunctionTransformer(lambda X: X)
    ),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=5, max_features=0.8500000000000001, min_samples_leaf=17, min_samples_split=11, subsample=0.45)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
