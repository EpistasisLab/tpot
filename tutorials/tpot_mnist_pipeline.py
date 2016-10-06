import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, FunctionTransformer
from sklearn.svm import SVC
from tpot.operators.preprocessors import ZeroCount
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        make_union(VotingClassifier([('branch',
            make_pipeline(
                Binarizer(threshold=0.62),
                XGBClassifier(learning_rate=1.0, max_depth=10, min_child_weight=20, n_estimators=500, subsample=1.0)
            )
        )]), FunctionTransformer(lambda X: X)),
        make_pipeline(
            RFE(estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=42, shrinking=True,
              tol=0.001, verbose=False), step=0.99),
            ZeroCount()
        )
    ),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
