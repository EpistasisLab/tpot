# TPOT

* * *
## Fit

###Description
   Uses genetic programming to optimize a Machine Learning pipeline that
   maximizes classification accuracy on the provided `features` and `classes`.
   Optionally, name the features in the data frame according to `feature_names`.
   Performs a stratified training/testing cross-validaton split to avoid
   overfitting on the provided data.

###Parameters
    features: array-like {n_samples, n_features}
        Feature matrix
    classes: array-like {n_samples}
        List of class labels for prediction
    feature_names: array-like {n_features} (default: None)
        List of feature names as strings

###Returns
    None

* * *

## Predict

###Description
    Uses the optimized pipeline to predict the classes for a feature set.

###Parameters
    training_features: array-like {n_samples, n_features}
        Feature matrix of the training set
    training_classes: array-like {n_samples}
        List of class labels for prediction in the training set
    testing_features: array-like {n_samples, n_features}
        Feature matrix of the testing set

###Returns
    array-like: {n_samples}
        Predicted classes for the testing set

* * *
## Score



###Description
    Estimates the testing accuracy of the optimized pipeline.

###Parameters
    training_features: array-like {n_samples, n_features}
        Feature matrix of the training set
    training_classes: array-like {n_samples}
        List of class labels for prediction in the training set
    testing_features: array-like {n_samples, n_features}
        Feature matrix of the testing set
    testing_classes: array-like {n_samples}
        List of class labels for prediction in the testing set

###Returns
    accuracy_score: float
        The estimated test set accuracy


* * *
## Export

###Description
    Exports the current optimized pipeline as Python code.

###Parameters
    output_file_name: string
        String containing the path and file name of the desired output file

###Returns
    None

