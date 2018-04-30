# coding: utf-8
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from tpot.config import classifier_config_dict
from tpot.builtins import DatasetSelector
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

personal_config = classifier_config_dict
personal_config['tpot.builtins.DatasetSelector'] = {
    'subset_dir': ['./tests/test_subset_dir/'],
    'sel_subset_fname': ['test_subset_1.snp', 'test_subset_2.snp']
}
# print(personal_config)

tpot_data = pd.read_csv(
    './tests/tests.csv')
Xdata = tpot_data.loc[:, tpot_data.columns != 'class']
Ydata = tpot_data['class']

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                    train_size=0.75, test_size=0.25)


tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3,
                      config_dict=personal_config,
                      template='DatasetSelector-Classifier',
                      random_state=42)
tpot.fit(X_train, y_train)
print('Holdout Score',tpot.score(X_test, y_test))
