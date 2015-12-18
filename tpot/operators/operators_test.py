from operators import operator_registry
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import pandas as pd
from collections import Counter
import numpy as np

iris = load_iris()
training_features, testing_features, training_classes, testing_classes =\
    train_test_split(iris.data, iris.target, train_size=0.75, test_size=0.25)

training_data = pd.DataFrame(training_features)
training_data['class'] = training_classes
training_data['group'] = 'training'

testing_data = pd.DataFrame(testing_features)
testing_data['class'] = 0
testing_data['group'] = 'testing'

training_testing_data = pd.concat([training_data, testing_data])
most_frequent_class = Counter(training_classes).most_common(1)[0][0]
training_testing_data['guess'] = most_frequent_class

def _balanced_accuracy(result):
    """Default scoring function: balanced class accuracy

    Parameters
    ----------
    result: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class']}
        A DataFrame containing a pipeline's predictions and the corresponding classes for the testing data

    Returns
    -------
    fitness: float
        Returns a float value indicating the `individual`'s balanced accuracy on the testing data

    """
    all_classes = list(set(result['class'].values))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_accuracy = len(result[(result['guess'] == this_class) \
            & (result['class'] == this_class)])\
            / float(len(result[result['class'] == this_class]))
        all_class_accuracies.append(this_class_accuracy)

    balanced_accuracy = np.mean(all_class_accuracies)

    return balanced_accuracy

for operator_name, Operator in operator_registry.iteritems():
    print operator_name
    operator_obj = Operator()
    result = operator_obj.evaluate_operator( \
        input_df=training_testing_data, n_estimators=100, max_features='auto')
    result.loc[result['group'] == 'testing','class'] = testing_classes
    print _balanced_accuracy(result)