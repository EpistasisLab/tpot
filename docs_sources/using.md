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
<td>Whether TPOT is being used for a classification or regression problem.</td>
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
<td>Number of generations to run pipeline optimization over. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize over. TPOT will evaluate GENERATIONS x POPULATION_SIZE number of pipelines in total.</td>
</tr>
<tr>
<td>-p</td>
<td>POPULATION_SIZE</td>
<td>Any positive integer</td>
<td>Number of individuals in the GP population. Generally, TPOT will work better when you give it more individuals (and therefore time) to optimize over. TPOT will evaluate GENERATIONS x POPULATION_SIZE number of pipelines in total.</td>
</tr>
<tr>
<td>-mr</td>
<td>MUTATION_RATE</td>
<td>[0.0, 1.0]</td>
<td>GP mutation rate. We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.</td>
</tr>
<tr>
<td>-xr</td>
<td>CROSSOVER_RATE</td>
<td>[0.0, 1.0]</td>
<td>GP crossover rate in the range [0.0, 1.0]. We recommend using the default parameter unless you understand how the crossover rate affects GP algorithms.</td>
</tr>
<tr>
<td>-cv</td>
<td>NUM_CV_FOLDS</td>
<td>Any integer >2</td>
<td>The number of folds to evaluate each pipeline over in k-fold cross-validation during the TPOT pipeline optimization process.</td>
</tr>
<tr>
<td>-scoring</td>
<td>SCORING_FN</td>
<td>'accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'</td>
<td>Function used to evaluate the quality of a given pipeline for the problem. By default, balanced accuracy is used for classification and mean squared error is used for regression. TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized. See the section on <a href="#scoringfunctions">scoring functions</a> for more details.</td>
</tr>
<tr>
<td>-maxtime</td>
<td>MAX_TIME_MINS</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize the pipeline. This setting will override the GENERATIONS parameter and allow TPOT to run until it runs out of time.</td>
</tr>
<tr>
<td>-maxeval</td>
<td>MAX_EVAL_MINS</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize a single pipeline. Setting this parameter to higher values will allow TPOT to explore more complex pipelines but will also allow TPOT to run longer.</td>
</tr>
<tr>
<td>-s</td>
<td>RANDOM_STATE</td>
<td>Any positive integer</td>
<td>Random number generator seed for reproducibility. Set this seed if you want your TPOT run to be reproducible with the same seed and data set in the future.</td>
</tr>
<tr>
<td>-v</td>
<td>VERBOSITY</td>
<td>{0, 1, 2, 3}</td>
<td>How much information TPOT communicates while it is running: 0 = none, 1 = minimal, 2 = all. A setting of 2 or higher will add a progress bar during the optimization procedure.</td>
</tr>
<tr>
<td colspan=2>--no-update-check</td>
<td>N/A</td>
<td>Flag indicating whether the TPOT version checker should be disabled.</td>
</tr>
<tr>
<td colspan=2>--version</td>
<td>N/A</td>
<td>Show TPOT's version number and exit.</td>
</tr>
<tr>
<td colspan=2>--help</td>
<td>N/A</td>
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
<td>generation</td>
<td>Any positive integer</td>
<td>The number of generations to run pipeline optimization over. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize over. TPOT will evaluate generations x population_size number of pipelines in total.</td>
</tr>
<tr>
<td>population_size</td>
<td>Any positive integer</td>
<td>The number of individuals in the GP population. Generally, TPOT will work better when you give it more individuals (and therefore time) to optimize over. TPOT will evaluate generations x population_size number of pipelines in total.</td>
</tr>
<tr>
<td>mutation_rate</td>
<td>[0.0, 1.0]</td>
<td>The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.</td>
</tr>
<tr>
<td>crossover_rate</td>
<td>[0.0, 1.0]</td>
<td>The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.</td>
</tr>
<tr>
<td>num_cv_folds</td>
<td>[2, 10]</td>
<td>The number of folds to evaluate each pipeline over in k-fold cross-validation during the TPOT pipeline optimization process.</td>
</tr>
<tr>
<td>scoring</td>
<td>'accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc' or a callable function with signature <b>scorer(y_true, y_pred)</b></td>
<td>Function used to evaluate the quality of a given pipeline for the problem. By default, balanced accuracy is used for classification and mean squared error is used for regression. TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized. See the section on <a href="#scoringfunctions">scoring functions</a> for more details.</td>
</tr>
<tr>
<td>max_time_mins</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize the pipeline. This setting will override the generations parameter.</td>
</tr>
<tr>
<td>max_eval_time_mins</td>
<td>Any positive integer</td>
<td>How many minutes TPOT has to optimize a single pipeline. Setting this parameter to higher values will allow TPOT to explore more complex pipelines but will also allow TPOT to run longer.</td>
</tr>
<tr>
<td>random_state</td>
<td>Any positive integer</td>
<td>The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.</td>
</tr>
<tr>
<td>verbosity</td>
<td>{0, 1, 2, 3}</td>
<td>How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = high, 3 = all. A setting of 2 or higher will add a progress bar to calls to fit().</td>
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

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
```

Now TPOT is ready to optimize a pipeline for you. You can tell TPOT to optimize a pipeline based on a data set with the `fit` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
```

The `fit()` function takes in a training data set and uses k-fold cross-validation when evaluating pipelines. It then initializes the genetic programming algoritm to find the best pipeline based on average k-fold score.

You can then proceed to evaluate the final pipeline on the testing set with the `score()` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
```

Finally, you can tell TPOT to export the corresponding Python code for the optimized pipeline to a text file with the `export()` function:

```Python
from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

Check our [examples](examples/MNIST_Example/) to see TPOT applied to some specific data sets.

<a name="scoringfunctions"></a>
## Scoring functions

TPOT makes use of `sklearn.model_selection.cross_val_score`, and as such offers the same support for scoring functions. There are two ways to make use of scoring functions with TPOT:

1. You can pass in a string from the list described in the table above. Any other strings will cause internal issues that may break your code down the line.

2. You can pass in a function with the signature `scorer(y_true, y_pred)`, where `y_true` are the true target values and `y_pred` are the predicted target values from an estimator. To do this, you should implement your own function. See the example below for further explanation.

```Python
def accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)
```
