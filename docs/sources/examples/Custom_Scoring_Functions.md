# Custom Scoring Functions

Below is a minimal working example of different scoring metrics/fitness functions used with the MNIST dataset.

```python

from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

def precision(result):
    all_classes = list(set(result['class'].values))
    all_class_tps = []
    all_class_tps_fps = []
    for this_class in all_classes:
        #True Positives are those examples that belong to a class and whose class was guessed correctly
        this_class_tps = len(result[(result['guess'] == this_class) \
            & (result['class'] == this_class)])
        all_class_tps.append(this_class_tps)
        #False Positives are those examples that were guessed to belong to a class 
        this_class_tps_fps = len(result[(result['guess'] == this_class) \
            | (result['class'] == this_class)])
        all_class_tps_fps.append(this_class_tps_fps)

    micro_avg_precision = float(np.sum(all_class_tps)) / np.sum(all_class_tps_fps)

    return micro_avg_precision
    
def recall(result):
    all_classes = list(set(result['class'].values))
    all_class_tps = []
    all_class_tps_fns = []
    for this_class in all_classes:
        this_class_tps = len(result[(result['guess'] == this_class) \
            & (result['class'] == this_class)]) 
        #True Positives and False Negatives are those examples that belong to a specific class regardless of guess
        this_class_tps_fns = len(result[(result['class'] == this_class)])
        all_class_tps.append(this_class_tps)
        all_class_tps_fns.append(this_class_tps_fns)

    micro_avg_recall = float(np.sum(all_class_tps)) / np.sum(all_class_tps_fns)
    return micro_avg_recall

def f1(result):
    all_classes = list(set(result['class'].values))
    all_class_tps = []
    all_class_tps_fps = []
    all_class_tps_fns = []
    for this_class in all_classes:
        this_class_tps = len(result[(result['guess'] == this_class) \
            & (result['class'] == this_class)])
        this_class_tps_fns = len(result[(result['class'] == this_class)])
        this_class_tps_fps = len(result[(result['guess'] == this_class) \
            | (result['class'] == this_class)])
        all_class_tps.append(this_class_tps)
        all_class_tps_fps.append(this_class_tps_fps)
        all_class_tps_fns.append(this_class_tps_fns)
    micro_avg_precision = float(np.sum(all_class_tps)) / np.sum(all_class_tps_fps)
    micro_avg_recall = float(np.sum(all_class_tps)) / np.sum(all_class_tps_fns)
    micro_avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)

    return micro_avg_f1

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print 'acc: ', tpot.score(X_test, y_test)

tpot = TPOT(generations=5, scoring_function=precision)
tpot.fit(X_train, y_train)
print 'precision: ', tpot.score(X_test, y_test)

tpot = TPOT(generations=5, scoring_function=recall)
tpot.fit(X_train, y_train)
print 'recall: ', tpot.score(X_test, y_test)

tpot = TPOT(generations=5, scoring_function=f1)
tpot.fit(X_train, y_train)
print 'f1: ', tpot.score(X_test, y_test)

```

Running this example should discover a pipeline that achieves ~98% testing accuracy, ~93% testing precision, ~97% testing recall, and ~95% testing f1.
