import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
from tpot.builtins import ColumnTransformer, OneHotEncoder, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9547074211758227
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.022),
    ColumnTransformer(transformer_0=Binarizer(threshold=0.35000000000000003), transformer_1=FastICA(tol=0.30000000000000004), transformer_10=RobustScaler(), transformer_11=StandardScaler(), transformer_12=ZeroCount(), transformer_13=OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10), transformer_2=FeatureAgglomeration(affinity="manhattan", linkage="average"), transformer_3=MaxAbsScaler(), transformer_4=MinMaxScaler(), transformer_5=Normalizer(norm="l1"), transformer_6=Nystroem(gamma=0.65, kernel="cosine", n_components=9), transformer_7=PCA(iterated_power=7, svd_solver="randomized"), transformer_8=PolynomialFeatures(degree=2, include_bias=False, interaction_only=False), transformer_9=RBFSampler(gamma=0.0), choice=0, include_col_0=False, include_col_1=False, include_col_10=True, include_col_11=False, include_col_12=False, include_col_13=False, include_col_14=False, include_col_15=False, include_col_16=False, include_col_17=True, include_col_18=True, include_col_19=True, include_col_2=True, include_col_20=False, include_col_21=True, include_col_22=True, include_col_23=False, include_col_24=True, include_col_25=False, include_col_26=True, include_col_27=False, include_col_28=True, include_col_29=True, include_col_3=False, include_col_30=False, include_col_31=False, include_col_32=False, include_col_33=True, include_col_34=True, include_col_35=True, include_col_36=False, include_col_37=True, include_col_38=True, include_col_39=True, include_col_4=False, include_col_40=False, include_col_41=True, include_col_42=True, include_col_43=True, include_col_44=True, include_col_45=True, include_col_46=False, include_col_47=False, include_col_48=False, include_col_49=False, include_col_5=True, include_col_50=True, include_col_51=True, include_col_52=False, include_col_53=False, include_col_54=True, include_col_55=True, include_col_56=True, include_col_57=True, include_col_58=False, include_col_59=False, include_col_6=True, include_col_60=False, include_col_61=True, include_col_62=True, include_col_63=True, include_col_7=False, include_col_8=True, include_col_9=True, remainder="passthrough"),
    SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=True, l1_ratio=1.0, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
