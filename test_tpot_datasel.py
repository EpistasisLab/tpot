# coding: utf-8
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tpot.config import classifier_config_dict
import pandas as pd
import numpy as np

personal_config = classifier_config_dict
personal_config['tpot.builtins.DatasetSelector'] = {
    'subset_dir': ['./tests/test_subset_dir/'],
    'sel_subset_idx': range(0, 1)
}
# print(personal_config)

tpot_data = pd.read_csv(
    './tests/tests.csv')
Xdata = tpot_data.loc[:, tpot_data.columns != 'class']
Ydata = tpot_data[['class']]

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                    train_size=0.75, test_size=0.25)
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
#                                                     train_size=0.75, test_size=0.25)


tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      config_dict=personal_config,
                      template='DatasetSelector-Transformer-Classifier')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
                      
