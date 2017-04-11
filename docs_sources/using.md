# TPOT on the command line

To use TPOT via the command line, enter the following command with a path to the data file:

```Shell
tpot /path_to/data_file.csv
```

TPOT offers several arguments that can be provided at the command line:

<table>
<tr>
<th>Argument</th>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>-is</td>
<td>INPUT_SEPARATOR</td>
<td>Any string</td>
<td>Character used to separate columns in the input file.</td>
</tr>
<tr>
<td>-target</td>
<td>TARGET_NAME</td>
<td>Any string</td>
<td>Name of the target column in the input file.</td>
</tr>
<tr>
<td>-mode</td>
<td>TPOT_MODE</td>
<td>['classification', 'regression']</td>
<td>Whether TPOT is being used for a supervised classification or regression problem.</td>
</tr>
<tr>
<td>-o</td>
<td>OUTPUT_FILE</td>
<td>String path to a file</td>
<td>File to export the code for the final optimized pipeline.</td>
</tr>
<tr>
<td>-g</td>
<td>GENERATIONS</td>
<td>Any positive integer</td>
<td>Number of iterations to run the pipeline optimization process. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.</td>
</tr>
<tr>
<td>-p</td>
<td>POPULATION_SIZE</td>
<td>Any positive integer</td>
<td>Number of individuals to retain in the GP population every generation. Generally, TPOT will work better when you give it more individuals (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.</td>
</tr>
<tr>
<td>-os</td>
<td>OFFSPRING_SIZE</td>
<td>Any positive integer</td>
<td>Number of offspring to produce in each GP generation.
<br /><br />
By default, OFFSPRING_SIZE = POPULATION_SIZE.</td>
</tr>
<tr>
<td>-mr</td>
<td>MUTATION_RATE</td>
<td>[0.0, 1.0]</td>
<td>GP mutation rate in the range [0.0, 1.0]. This tells the GP algorithm how many pipelines to apply random changes to every generation.
<br /><br />
We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.</td>
</tr>
<tr>
<td>-xr</td>
<td>CROSSOVER_RATE</td>
<td>[0.0, 1.0]</td>
<td>GP crossover rate in the range [0.0, 1.0]. This tells the GP algorithm how many pipelines to "breed" every generation.
<br /><br />
We recommend using the default parameter unless you understand how the crossover rate affects GP algorithms.</td>
</tr>
<tr>
<td>-scoring</td>
<td>SCORING_FN</td>
<td>'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',<br />'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'</td>
<td>Function used to evaluate the quality of a given pipeline for the problem. By default, accuracy is used for classification and mean squared error (MSE) is used for regression.
<br /><br />
TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized.
<br /><br />
See the section on <a href="#scoringfunctions">scoring functions</a> for more details.</td>
</tr>
<tr>
<td>-cv</td>
<td>NUM_CV_FOLDS</td>
<td>Any integer >1</td>
<td>Number of folds to evaluate each pipeline over in 'k-fold cross-validation during the TPOT optimization process.</td>
</tr>
<tr>
<td>-njobs</td>
<td>NUM_JOBS</td>
<td>Any positive integer or -1</td>
<td>Number of CPUs for evaluating pipelines in parallel during the TPOT optimization process.
<br /><br />
Assigning this to -1 will use as many cores as available on the computer.</td>
</tr>
<tr>
<td>-maxtime</td>
<td>MAX_TIME_MINS</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize the pipeline.
<br /><br />
If provided, this setting will override the "generations" parameter and allow TPOT to run until it runs out of time.</td>
</tr>
<tr>
<td>-maxeval</td>
<td>MAX_EVAL_MINS</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to evaluate a single pipeline.
<br /><br />
Setting this parameter to higher values will allow TPOT to explore more complex pipelines but will also allow TPOT to run longer.</td>
</tr>
<tr>
<td>-s</td>
<td>RANDOM_STATE</td>
<td>Any positive integer</td>
<td>Random number generator seed for reproducibility.
<br /><br />
Set this seed if you want your TPOT run to be reproducible with the same seed and data set in the future.</td>
</tr>
<tr>
<td>-config</td>
<td>CONFIG_FILE</td>
<td>String path to a file</td>
<td>Configuration file for customizing the operators and parameters that TPOT uses in the optimization process.
<br /><br />
See the <a href="#customconfig">custom configuration</a> section for more information and examples.</td>
</tr>
<tr>
<td>-v</td>
<td>VERBOSITY</td>
<td>{0, 1, 2, 3}</td>
<td>How much information TPOT communicates while it is running.
<br /><br />
0 = none, 1 = minimal, 2 = high, 3 = all.
<br /><br />
A setting of 2 or higher will add a progress bar during the optimization procedure.</td>
</tr>
<tr>
<td colspan=3>--no-update-check</td>
<td>Flag indicating whether the TPOT version checker should be disabled.</td>
</tr>
<tr>
<td colspan=3>--version</td>
<td>Show TPOT's version number and exit.</td>
</tr>
<tr>
<td colspan=3>--help</td>
<td>Show TPOT's help documentation and exit.</td>
</tr>
</table>

An example command-line call to TPOT may look like:

```Shell
tpot data/mnist.csv -is , -target class -o tpot_exported_pipeline.py -g 5 -p 20 -cv 5 -s 42 -v 2
```

# TPOT with code

We've taken care to design the TPOT interface to be as similar as possible to scikit-learn.

TPOT can be imported just like any regular Python module. To import TPOT, type:

```Python
from tpot import TPOTClassifier
```

then create an instance of TPOT as follows:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier()
```

It's also possible to use TPOT for regression problems with the `TPOTRegressor` class. Other than the class name, a `TPOTRegressor` is used the same way as a `TPOTClassifier`.

Note that you can pass several parameters to the TPOT instantiation call:

<table>
<tr>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>generations</td>
<td>Any positive integer</td>
<td>Number of iterations to the run pipeline optimization process. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.</td>
</tr>
<tr>
<td>population_size</td>
<td>Any positive integer</td>
<td>Number of individuals to retain in the GP population every generation. Generally, TPOT will work better when you give it more individuals (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.</td>
</tr>
<tr>
<td>offspring_size</td>
<td>Any positive integer</td>
<td>Number of offspring to produce in each GP generation.
<br /><br />
By default, offspring_size = population_size.</td>
</tr>
<tr>
<td>mutation_rate</td>
<td>[0.0, 1.0]</td>
<td>Mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the GP algorithm how many pipelines to apply random changes to every generation.
<br /><br />
We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.</td>
</tr>
<tr>
<td>crossover_rate</td>
<td>[0.0, 1.0]</td>
<td>Crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the genetic programming algorithm how many pipelines to "breed" every generation.
<br /><br />
We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.</td>
</tr>
<tr>
<td>scoring</td>
<td>'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',<br />'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc' or a callable function with signature <b>scorer(y_true, y_pred)</b></td>
<td>Function used to evaluate the quality of a given pipeline for the problem. By default, accuracy is used for classification and mean squared error (MSE) is used for regression.
<br /><br />
TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized.
<br /><br />
See the section on <a href="#scoringfunctions">scoring functions</a> for more details.</td>
</tr>
<tr>
<td>cv</td>
<td>Any integer >1</td>
<td>Number of folds to evaluate each pipeline over in k-fold cross-validation during the TPOT optimization process.</td>
</tr>
<tr>
<td>n_jobs</td>
<td>Any positive integer or -1</td>
<td>Number of CPUs for evaluating pipelines in parallel during the TPOT optimization process.
<br /><br />
Assigning this to -1 will use as many cores as available on the computer.</td>
</tr>
<tr>
<td>max_time_mins</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize the pipeline.
<br /><br />
If provided, this setting will override the "generations" parameter and allow TPOT to run until it runs out of time.</td>
</tr>
<tr>
<td>max_eval_time_mins</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize a single pipeline.
<br /><br />
Setting this parameter to higher values will allow TPOT to explore more complex pipelines, but will also allow TPOT to run longer.</td>
</tr>
<tr>
<td>random_state</td>
<td>Any positive integer</td>
<td>Random number generator seed for TPOT.
<br /><br />
Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.</td>
</tr>
<tr>
<td>config_dict</td>
<td>Python dictionary</td>
<td>Configuration dictionary for customizing the operators and parameters that TPOT uses in the optimization process.
<br /><br />
See the <a href="#customconfig">custom configuration</a> section for more information and examples.
</pre>
</td>
</tr>
<tr>
<td>warm_start</td>
<td>[True, False]</td>
<td>Flag indicating whether the TPOT instance will reuse the population from previous calls to fit().</td>
</tr>
<tr>
<td>verbosity</td>
<td>{0, 1, 2, 3}</td>
<td>How much information TPOT communicates while it's running.
<br /><br />
0 = none, 1 = minimal, 2 = high, 3 = all.
<br /><br />
A setting of 2 or higher will add a progress bar during the optimization procedure.</td>
</tr>
<tr>
<td>disable_update_check</td>
<td>[True, False]</td>
<td>Flag indicating whether the TPOT version checker should be disabled.</td>
</tr>
</table>

Some example code with custom TPOT parameters might look like:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
```

Now TPOT is ready to optimize a pipeline for you. You can tell TPOT to optimize a pipeline based on a data set with the `fit` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
```

The `fit()` function takes in a training data set and uses k-fold cross-validation when evaluating pipelines. It then initializes the genetic programming algoritm to find the best pipeline based on average k-fold score.

You can then proceed to evaluate the final pipeline on the testing set with the `score()` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
```

Finally, you can tell TPOT to export the corresponding Python code for the optimized pipeline to a text file with the `export()` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

Check our [examples](examples/) to see TPOT applied to some specific data sets.

<a name="scoringfunctions"></a>
## Scoring functions

TPOT makes use of `sklearn.model_selection.cross_val_score` for evaluating pipelines, and as such offers the same support for scoring functions. There are two ways to make use of scoring functions with TPOT:

1. You can pass in a string to the `scoring` parameter from the list above. Any other strings will cause TPOT to throw an exception.

2. You can pass a function with the signature `scorer(y_true, y_pred)`, where `y_true` are the true target values and `y_pred` are the predicted target values from an estimator. To do this, you should implement your own function. See the example below for further explanation.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

def accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      scoring=accuracy)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

<a name="customconfig"></a>
## Customizing TPOT's operators and parameters

TPOT comes with a handful of default operators and parameter configurations that we believe work well for optimizing machine learning pipelines. However, in some cases it is useful to limit the algorithms and parameters that TPOT explores. For that reason, we allow users to provide TPOT with a custom configuration for its operators and parameters.

The custom TPOT configuration must be in nested dictionary format, where the first level key is the path and name of the operator (e.g., `sklearn.naive_bayes.MultinomialNB`) and the second level key is the corresponding parameter name for that operator (e.g., `fit_prior`). The second level key should point to a list of parameter values for that parameter, e.g., `'fit_prior': [True, False]`.

For a simple example, the configuration could be:

```Python
classifier_config_dict = {
    'sklearn.naive_bayes.GaussianNB': {
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    }
}
```

in which case TPOT would only explore pipelines containing `GaussianNB`, `BernoulliNB`, `MultinomialNB`, and tune those algorithm's parameters in the ranges provided. This dictionary can be passed directly within the code to the `TPOTClassifier`/`TPOTRegressor` `config_dict` parameter, described above. For example:

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

classifier_config_dict = {
    'sklearn.naive_bayes.GaussianNB': {
    },
    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    }
}

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      config_dict=classifier_config_dict)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Command-line users must create a separate `.py` file with the custom configuration and provide the path to the file to the `tpot` call. For example, if the simple example configuration above is saved in `tpot_classifier_config.py`, that configuration could be used on the command line with the command:

```
tpot data/mnist.csv -is , -target class -config tpot_classifier_config.py -g 5 -p 20 -v 2 -o tpot_exported_pipeline.py
```

For more detailed examples of how to customize TPOT's operator configuration, see the default configurations for [classification](https://github.com/rhiever/tpot/blob/master/tpot/config_classifier.py) and [regression](https://github.com/rhiever/tpot/blob/master/tpot/config_regressor.py) in TPOT's source code.

Note that you must have all of the corresponding packages for the operators installed on your computer, otherwise TPOT will not be able to use them. For example, if XGBoost is not installed on your computer, then TPOT will simply not import nor use XGBoost in the pipelines it explores.
