## TPOT
* * *
__tpot.fit(self, features, classes)__


Uses genetic programming to optimize a Machine Learning pipeline that
   maximizes classification accuracy on the provided `features` and `classes`.
   Performs an internal stratified training/testing cross-validaton split to avoid
   overfitting on the provided data.

_Parameters_

    features: array-like {n_samples, n_features}
        Feature matrix
    classes: array-like {n_samples}
        List of class labels for prediction

_Returns_

    None

* * *
__tpot.predict(self, testing_features)__


Uses the optimized pipeline to predict the classes for a feature set.

_Parameters_

    testing_features: array-like {n_samples, n_features}
        Feature matrix of the testing set

_Returns_

    array-like: {n_samples}
        Predicted classes for the testing set

* * *

__tpot.score(self, testing_features, testing_classes)__

    Estimates the testing accuracy of the optimized pipeline.

_Parameters_

    testing_features: array-like {n_samples, n_features}
        Feature matrix of the testing set
    testing_classes: array-like {n_samples}
        List of class labels for prediction in the testing set

_Returns_

    accuracy_score: float
        The estimated test set accuracy


* * *
__tpot.export(self, output_file_name)__

    Exports the current optimized pipeline as Python code.

_Parameters_

    output_file_name: string
        String containing the path and file name of the desired output file

_Returns_

    None

