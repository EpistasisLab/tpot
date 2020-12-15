import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
from tpot.builtins import ColumnTransformer, OneHotEncoder, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9628858598375327
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.05),
    ColumnTransformer(transformer_0=Binarizer(threshold=0.65), transformer_1=FastICA(tol=0.35000000000000003), transformer_10=RobustScaler(), transformer_11=StandardScaler(), transformer_12=ZeroCount(), transformer_13=OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10), transformer_2=FeatureAgglomeration(affinity="l2", linkage="average"), transformer_3=MaxAbsScaler(), transformer_4=MinMaxScaler(), transformer_5=Normalizer(norm="l2"), transformer_6=Nystroem(gamma=0.1, kernel="sigmoid", n_components=10), transformer_7=PCA(iterated_power=2, svd_solver="randomized"), transformer_8=PolynomialFeatures(degree=2, include_bias=False, interaction_only=False), transformer_9=RBFSampler(gamma=0.2), choice=0, include_col_0=True, include_col_1=True, include_col_10=False, include_col_11=True, include_col_12=True, include_col_13=False, include_col_14=True, include_col_15=False, include_col_16=True, include_col_17=False, include_col_18=True, include_col_19=True, include_col_2=False, include_col_20=False, include_col_21=False, include_col_22=False, include_col_23=False, include_col_24=True, include_col_25=True, include_col_26=True, include_col_27=False, include_col_28=False, include_col_29=True, include_col_3=False, include_col_30=True, include_col_31=True, include_col_32=False, include_col_33=False, include_col_34=True, include_col_35=False, include_col_36=False, include_col_37=False, include_col_38=True, include_col_39=False, include_col_4=False, include_col_40=False, include_col_41=True, include_col_42=False, include_col_43=True, include_col_44=True, include_col_45=True, include_col_46=True, include_col_47=False, include_col_48=False, include_col_49=True, include_col_5=True, include_col_50=True, include_col_51=False, include_col_52=False, include_col_53=True, include_col_54=True, include_col_55=True, include_col_56=True, include_col_57=True, include_col_58=True, include_col_59=True, include_col_6=True, include_col_60=False, include_col_61=True, include_col_62=False, include_col_63=False, include_col_7=False, include_col_8=False, include_col_9=False, remainder="drop"),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.2, min_samples_leaf=6, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
