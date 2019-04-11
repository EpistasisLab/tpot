# What to expect from AutoML software

Automated machine learning (AutoML) takes a higher-level approach to machine learning than most practitioners are used to,
so we've gathered a handful of guidelines on what to expect when running AutoML software such as TPOT.

<h5>AutoML algorithms aren't intended to run for only a few minutes</h5>

Of course, you *can* run TPOT for only a few minutes and it will find a reasonably good pipeline for your dataset.
However, if you don't run TPOT for long enough, it may not find the best possible pipeline for your dataset. It may even not
find any suitable pipeline at all, in which case a `RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')`
will be raised.
Often it is worthwhile to run multiple instances of TPOT in parallel for a long time (hours to days) to allow TPOT to thoroughly search
the pipeline space for your dataset.

<h5>AutoML algorithms can take a long time to finish their search</h5>

AutoML algorithms aren't as simple as fitting one model on the dataset; they are considering multiple machine learning algorithms
(random forests, linear models, SVMs, etc.) in a pipeline with multiple preprocessing steps (missing value imputation, scaling,
PCA, feature selection, etc.), the hyperparameters for all of the models and preprocessing steps, as well as multiple ways
to ensemble or stack the algorithms within the pipeline.

As such, TPOT will take a while to run on larger datasets, but it's important to realize why. With the default TPOT settings
(100 generations with 100 population size), TPOT will evaluate 10,000 pipeline configurations before finishing.
To put this number into context, think about a grid search of 10,000 hyperparameter combinations for a machine learning algorithm
and how long that grid search will take. That is 10,000 model configurations to evaluate with 10-fold cross-validation,
which means that roughly 100,000 models are fit and evaluated on the training data in one grid search.
That's a time-consuming procedure, even for simpler models like decision trees.

Typical TPOT runs will take hours to days to finish (unless it's a small dataset), but you can always interrupt
the run partway through and see the best results so far. TPOT also [provides](/api/) a `warm_start` parameter that
lets you restart a TPOT run from where it left off.

<h5>AutoML algorithms can recommend different solutions for the same dataset</h5>

If you're working with a reasonably complex dataset or run TPOT for a short amount of time, different TPOT runs
may result in different pipeline recommendations. TPOT's optimization algorithm is stochastic in nature, which means
that it uses randomness (in part) to search the possible pipeline space. When two TPOT runs recommend different
pipelines, this means that the TPOT runs didn't converge due to lack of time *or* that multiple pipelines
perform more-or-less the same on your dataset.

This is actually an advantage over fixed grid search techniques: TPOT is meant to be an assistant that gives
you ideas on how to solve a particular machine learning problem by exploring pipeline configurations that you
might have never considered, then leaves the fine-tuning to more constrained parameter tuning techniques such
as grid search.


# TPOT with code

We've taken care to design the TPOT interface to be as similar as possible to scikit-learn.

TPOT can be imported just like any regular Python module. To import TPOT, type:

```Python
from tpot import TPOTClassifier
```

then create an instance of TPOT as follows:

```Python
pipeline_optimizer = TPOTClassifier()
```

It's also possible to use TPOT for regression problems with the `TPOTRegressor` class. Other than the class name,
a `TPOTRegressor` is used the same way as a `TPOTClassifier`. You can read more about the `TPOTClassifier` and `TPOTRegressor` classes in the [API documentation](/api/).

Some example code with custom TPOT parameters might look like:

```Python
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
```

Now TPOT is ready to optimize a pipeline for you. You can tell TPOT to optimize a pipeline based on a data set with the `fit` function:

```Python
pipeline_optimizer.fit(X_train, y_train)
```

The `fit` function initializes the genetic programming algorithm to find the highest-scoring pipeline based on average k-fold cross-validation
Then, the pipeline is trained on the entire set of provided samples, and the TPOT instance can be used as a fitted model.

You can then proceed to evaluate the final pipeline on the testing set with the `score` function:

```Python
print(pipeline_optimizer.score(X_test, y_test))
```

Finally, you can tell TPOT to export the corresponding Python code for the optimized pipeline to a text file with the `export` function:

```Python
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

Below is a full example script using TPOT to optimize a pipeline, score it, and export the best pipeline to a file.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Check our [examples](examples/) to see TPOT applied to some specific data sets.

# TPOT on the command line

To use TPOT via the command line, enter the following command with a path to the data file:

```Shell
tpot /path_to/data_file.csv
```

An example command-line call to TPOT may look like:

```Shell
tpot data/mnist.csv -is , -target class -o tpot_exported_pipeline.py -g 5 -p 20 -cv 5 -s 42 -v 2
```

TPOT offers several arguments that can be provided at the command line. To see brief descriptions of these arguments,
enter the following command:

```Shell
tpot --help
```

Detailed descriptions of the command-line arguments are below.

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
<td>'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',<br />'f1',
'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error',
'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro',
'precision_samples', 'precision_weighted',<br />'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
'recall_weighted', 'roc_auc', 'my_module.scorer_name*'</td>
<td>Function used to evaluate the quality of a given pipeline for the problem. By default, accuracy is used for classification and mean squared error (MSE) is used for regression.
<br /><br />
TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized.
<br /><br />
my_module.scorer_name: You can also specify your own function or a full python path to an existing one.
<br /><br />
See the section on <a href="#scoring-functions">scoring functions</a> for more details.</td>
</tr>
<tr>
<td>-cv</td>
<td>CV</td>
<td>Any integer > 1</td>
<td>Number of folds to evaluate each pipeline over in k-fold cross-validation during the TPOT optimization process.</td>
</tr>
<td>-sub</td>
<td>SUBSAMPLE</td>
<td>(0.0, 1.0]</td>
<td>Subsample ratio of the training instance. Setting it to 0.5 means that TPOT randomly collects half of training samples for pipeline optimization process.</td>
</tr>
<tr>
<td>-njobs</td>
<td>NUM_JOBS</td>
<td>Any positive integer or -1</td>
<td>Number of CPUs for evaluating pipelines in parallel during the TPOT optimization process.
<br /><br />
Assigning this to -1 will use as many cores as available on the computer. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.</td>
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
<td>Any positive float</td>
<td>How many minutes TPOT has to evaluate a single pipeline.
<br /><br />
Setting this parameter to higher values will allow TPOT to consider more complex pipelines but will also allow TPOT to run longer.</td>
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
<td>String or file path</td>
<td>Operators and parameter configurations in TPOT:
<br /><br />
<ul>
<li>Path for configuration file: TPOT will use the path to a configuration file for customizing the operators and parameters that TPOT uses in the optimization process</li>
<li>string 'TPOT light', TPOT will use a built-in configuration with only fast models and preprocessors</li>
<li>string 'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies</li>
<li>string 'TPOT sparse': TPOT will use a configuration dictionary with a one-hot encoder and the operators normally included in TPOT that also support sparse matrices.</li>
</ul>
See the <a href="../using/#built-in-tpot-configurations">built-in configurations</a> section for the list of configurations included with TPOT, and the <a href="../using/#customizing-tpots-operators-and-parameters">custom configuration</a> section for more information and examples of how to create your own TPOT configurations.
</td>
</tr>
<tr>
<td>-template</td>
<td>TEMPLATE</td>
<td>String</td>
<td>Template of predefined pipeline structure. The option is for specifying a desired structure for the machine learning pipeline evaluated in TPOT. So far this option only supports linear pipeline structure. Each step in the pipeline should be a main class of operators (Selector, Transformer, Classifier or Regressor) or a specific operator (e.g. `SelectPercentile`) defined in TPOT operator configuration. If one step is a main class, TPOT will randomly assign all subclass operators (subclasses of [`SelectorMixin`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/base.py#L17), [`TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html), [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html) or [`RegressorMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) in scikit-learn) to that step. Steps in the template are delimited by "-", e.g. "SelectPercentile-Transformer-Classifier". By default value of template is "RandomTree", TPOT generates tree-based pipeline randomly.

See the <a href="../using/#template-option-in-tpot"> template option in tpot</a> section for more details.
</td>
</tr>
<tr>
<td>-memory</td>
<td>MEMORY</td>
<td>String or file path</td>
<td>If supplied, pipeline will cache each transformer after calling fit. This feature is used to avoid computing the fit transformers within a pipeline if the parameters and input data are identical with another fitted pipeline during optimization process. Memory caching mode in TPOT:
<br /><br />
<ul>
<li>Path for a caching directory: TPOT uses memory caching with the provided directory and TPOT does NOT clean the caching directory up upon shutdown.</li>
<li>string 'auto': TPOT uses memory caching with a temporary directory and cleans it up upon shutdown.</li>
</ul>
</td>
</tr>
<tr>
<td>-cf</td>
<td>CHECKPOINT_FOLDER</td>
<td>Folder path</td>
<td>
If supplied, a folder you created, in which tpot will periodically save pipelines in pareto front so far while optimizing.
<br /><br />
This is useful in multiple cases:
<ul>
<li>sudden death before tpot could save an optimized pipeline</li>
<li>progress tracking</li>
<li>grabbing a pipeline while tpot is working</li>
</ul>
<br /><br />
Example:
<br />
mkdir my_checkpoints
<br />
-cf ./my_checkpoints
</tr>
<tr>
<td>-es</td>
<td>EARLY_STOP</td>
<td>Any positive integer</td>
<td>
How many generations TPOT checks whether there is no improvement in optimization process.
<br /><br />
End optimization process if there is no improvement in the set number of generations.
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

# Scoring functions

TPOT makes use of `sklearn.model_selection.cross_val_score` for evaluating pipelines, and as such offers the same support for scoring functions. There are two ways to make use of scoring functions with TPOT:

- You can pass in a string to the `scoring` parameter from the list above. Any other strings will cause TPOT to throw an exception.

- You can pass the callable object/function with signature `scorer(estimator, X, y)`, where `estimator` is trained estimator to use for scoring, `X` are features that will be passed to `estimator.predict` and `y` are target values for `X`. To do this, you should implement your own function. See the example below for further explanation.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
# Make a custom metric function
def my_custom_accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)

# Make a custom a scorer from the custom metric function
# Note: greater_is_better=False in make_scorer below would mean that the scoring function should be minimized.
my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      scoring=my_custom_scorer)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

- You can pass a metric function with the signature `score_func(y_true, y_pred)` (e.g. `my_custom_accuracy` in the example above), where `y_true` are the true target values and `y_pred` are the predicted target values from an estimator. To do this, you should implement your own function. See the example above for further explanation. TPOT assumes that any function with "error" or "loss" in the function name is meant to be minimized (`greater_is_better=False` in [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)), whereas any other functions will be maximized. This scoring type was deprecated in version 0.9.1 and will be removed in version 0.11.


* **my_module.scorer_name**: You can also use a custom `score_func(y_true, y_pred)` or `scorer(estimator, X, y)` function through the command line by adding the argument `-scoring my_module.scorer` to your command-line call. TPOT will import your module and use the custom scoring function from there. TPOT will include your current working directory when importing the module, so you can place it in the same directory where you are going to run TPOT.
Example: `-scoring sklearn.metrics.auc` will use the function auc from sklearn.metrics module.

# Built-in TPOT configurations

TPOT comes with a handful of default operators and parameter configurations that we believe work well for optimizing machine learning pipelines. Below is a list of the current built-in configurations that come with TPOT.

<table>
<tr>
<th align="left">Configuration Name</th>
<th align="left">Description</th>
<th align="left">Operators</th>
</tr>

<tr>
<td>Default TPOT</td>
<td>TPOT will search over a broad range of preprocessors, feature constructors, feature selectors, models, and parameters to find a series of operators that minimize the error of the model predictions. Some of these operators are complex and may take a long time to run, especially on larger datasets.
<br /><br />
<strong>Note: This is the default configuration for TPOT.</strong> To use this configuration, use the default value (None) for the config_dict parameter.</td>
<td align="center"><a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier.py">Classification</a>
<br /><br />
<a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py">Regression</a></td>
</tr>

<tr>
<td>TPOT light</td>
<td>TPOT will search over a restricted range of preprocessors, feature constructors, feature selectors, models, and parameters to find a series of operators that minimize the error of the model predictions. Only simpler and fast-running operators will be used in these pipelines, so TPOT light is useful for finding quick and simple pipelines for a classification or regression problem.
<br /><br />
This configuration works for both the TPOTClassifier and TPOTRegressor.</td>
<td align="center"><a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier_light.py">Classification</a>
<br /><br />
<a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor_light.py">Regression</a></td>
</tr>

<tr>
<td>TPOT MDR</td>
<td>TPOT will search over a series of feature selectors and <a href="https://en.wikipedia.org/wiki/Multifactor_dimensionality_reduction">Multifactor Dimensionality Reduction</a> models to find a series of operators that maximize prediction accuracy. The TPOT MDR configuration is specialized for <a href="https://en.wikipedia.org/wiki/Genome-wide_association_study">genome-wide association studies (GWAS)</a>, and is described in detail online <a href="https://arxiv.org/abs/1702.01780">here</a>.
<br /><br />
Note that TPOT MDR may be slow to run because the feature selection routines are computationally expensive, especially on large datasets.</td>
<td align="center"><a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier_mdr.py">Classification</a>
<br /><br />
<a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor_mdr.py">Regression</a></td>
</tr>

<tr>
<td>TPOT sparse</td>
<td>TPOT uses a configuration dictionary with a one-hot encoder and the operators normally included in TPOT that also support sparse matrices.
<br /><br />
This configuration works for both the TPOTClassifier and TPOTRegressor.</td>
<td align="center"><a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier_sparse.py">Classification</a>
<br /><br />
<a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor_sparse.py">Regression</a></td>
</tr>

</table>

To use any of these configurations, simply pass the string name of the configuration to the `config_dict` parameter (or `-config` on the command line). For example, to use the "TPOT light" configuration:

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      config_dict='TPOT light')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')

```

# Customizing TPOT's operators and parameters

Beyond the default configurations that come with TPOT, in some cases it is useful to limit the algorithms and parameters that TPOT considers. For that reason, we allow users to provide TPOT with a custom configuration for its operators and parameters.

The custom TPOT configuration must be in nested dictionary format, where the first level key is the path and name of the operator (e.g., `sklearn.naive_bayes.MultinomialNB`) and the second level key is the corresponding parameter name for that operator (e.g., `fit_prior`). The second level key should point to a list of parameter values for that parameter, e.g., `'fit_prior': [True, False]`.

For a simple example, the configuration could be:

```Python
tpot_config = {
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

in which case TPOT would only consider pipelines containing `GaussianNB`, `BernoulliNB`, `MultinomialNB`, and tune those algorithm's parameters in the ranges provided. This dictionary can be passed directly within the code to the `TPOTClassifier`/`TPOTRegressor` `config_dict` parameter, described above. For example:

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot_config = {
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
                      config_dict=tpot_config)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Command-line users must create a separate `.py` file with the custom configuration and provide the path to the file to the `tpot` call. For example, if the simple example configuration above is saved in `tpot_classifier_config.py`, that configuration could be used on the command line with the command:

```
tpot data/mnist.csv -is , -target class -config tpot_classifier_config.py -g 5 -p 20 -v 2 -o tpot_exported_pipeline.py
```

When using the command-line interface, the configuration file specified in the `-config` parameter *must* name its custom TPOT configuration `tpot_config`. Otherwise, TPOT will not be able to locate the configuration dictionary.

For more detailed examples of how to customize TPOT's operator configuration, see the default configurations for [classification](https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier.py) and [regression](https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py) in TPOT's source code.

Note that you must have all of the corresponding packages for the operators installed on your computer, otherwise TPOT will not be able to use them. For example, if XGBoost is not installed on your computer, then TPOT will simply not import nor use XGBoost in the pipelines it considers.


# Template option in TPOT

Template option provides a way to specify a desired structure for machine learning pipeline, which may reduce TPOT computation time and potentially provide more interpretable results. Current implementation only supports linear pipelines.

Below is a simple example to use `template` option. The pipelines generated/evaluated in TPOT will follow this structure: 1st step is a feature selector (a subclass of [`SelectorMixin`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/base.py#L17)), 2nd step is a feature transformer (a subclass of [`TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)) and 3rd step is a classifier for classification (a subclass of [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html)). The last step must be `Classifier` for `TPOTClassifier`'s template but `Regressor` for `TPOTRegressor`. **Note: although `SelectorMixin` is subclass of `TransformerMixin` in scikit-leawrn, but `Transformer` in this option excludes those subclasses of `SelectorMixin`.**

```Python
tpot_obj = TPOTClassifier(
                template='Selector-Transformer-Classifier'
                )
```

If a specific operator, e.g. `SelectPercentile`, is prefered to used in the 1st step of pipeline, the template can be defined like 'SelectPercentile-Transformer-Classifier'.


# FeatureSetSelector in TPOT

`FeatureSetSelector` is a special new operator in TPOT. This operator enables feature selection based on *priori* export knowledge. For example, in RNA-seq gene expression analysis, this operator can be used to select one or more gene (feature) set(s) based on GO (Gene Ontology) terms or annotated gene sets Molecular Signatures Database ([MSigDB](http://software.broadinstitute.org/gsea/msigdb/index.jsp)) in the 1st step of pipeline via `template` option above, in order to reduce dimensions and TPOT computation time. This operator requires a dataset list in csv format. In this csv file, there are only three columns: 1st column is feature set names, 2nd column is the total number of features in one set and 3rd column is a list of feature names (if input X is pandas.DataFrame) or indexes (if input X is numpy.ndarray) delimited by ";". Below is a example how to use this operator in TPOT.

Please check our [preprint paper](https://www.biorxiv.org/content/10.1101/502484v1.article-info) for more details.

```Python
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from tpot.config import classifier_config_dict
test_data = pd.read_csv("https://raw.githubusercontent.com/EpistasisLab/tpot/master/tests/tests.csv")
test_X = test_data.drop("class", axis=1)
test_y = test_data['class']

# add FeatureSetSelector into tpot configuration
classifier_config_dict['tpot.builtins.FeatureSetSelector'] = {
    'subset_list': ['https://raw.githubusercontent.com/EpistasisLab/tpot/master/tests/subset_test.csv'],
    'sel_subset': [0,1] # select only one feature set, a list of index of subset in the list above
    #'sel_subset': list(combinations(range(3), 2)) # select two feature sets
}


tpot = TPOTClassifier(generations=5,
                           population_size=50, verbosity=2,
                           template='FeatureSetSelector-Transformer-Classifier',
                           config_dict=classifier_config_dict)
tpot.fit(test_X, test_y)
```


# Pipeline caching in TPOT

With the `memory` parameter, pipelines can cache the results of each transformer after fitting them. This feature is used to avoid repeated computation by transformers within a pipeline if the parameters and input data are identical to another fitted pipeline during optimization process. TPOT allows users to specify a custom directory path or [`sklearn.external.joblib.Memory`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/externals/joblib/memory.py#L847) in case they want to re-use the memory cache in future TPOT runs (or a `warm_start` run).

There are three methods for enabling memory caching in TPOT:

```Python
from tpot import TPOTClassifier
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory
from shutil import rmtree

# Method 1, auto mode: TPOT uses memory caching with a temporary directory and cleans it up upon shutdown
tpot = TPOTClassifier(memory='auto')

# Method 2, with a custom directory for memory caching
tpot = TPOTClassifier(memory='/to/your/path')

# Method 3, with a Memory object
cachedir = mkdtemp() # Create a temporary folder
memory = Memory(cachedir=cachedir, verbose=0)
tpot = TPOTClassifier(memory=memory)

# Clear the cache directory when you don't need it anymore
rmtree(cachedir)
```

**Note: TPOT does NOT clean up memory caches if users set a custom directory path or Memory object. We recommend that you clean up the memory caches when you don't need it anymore.**

# Crash/freeze issue with n_jobs > 1 under OSX or Linux

Internally, TPOT uses [joblib](http://joblib.readthedocs.io/) to fit estimators in parallel.
This is the same parallelization framework used by scikit-learn. But it may crash/freeze with n_jobs > 1 under OSX or Linux [as scikit-learn does](http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux), especially with large datasets.

One solution is to configure Python's `multiprocessing` module to use the `forkserver` start method (instead of the default `fork`) to manage the process pools. You can enable the `forkserver` mode globally for your program by putting the following codes into your main script:

```Python
import multiprocessing

# other imports, custom code, load data, define model...

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    # call scikit-learn utils or tpot utils with n_jobs > 1 here
```

More information about these start methods can be found in the [multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods).

# Parallel Training with Dask

For large problems or working on Jupyter notebook, we highly recommend that you can distribute the work on a [Dask](http://dask.pydata.org/en/latest/) cluster.
The [dask-examples binder](https://mybinder.org/v2/gh/dask/dask-examples/master?filepath=machine-learning%2Ftpot.ipynb) has a runnable example
with a small dask cluster.

To use your Dask cluster to fit a TPOT model, specify the ``use_dask`` keyword when you create the TPOT estimator. **Note: if `use_dask=True`, TPOT will use as many cores as available on the your Dask cluster. If `n_jobs` is specified, then it will control the chunk size (10*`n_jobs` if it is less then offspring size) of parallel training. **

```python
estimator = TPOTEstimator(use_dask=True, n_jobs=-1)
```

This will use use all the workers on your cluster to do the training, and use [Dask-ML's pipeline rewriting](https://dask-ml.readthedocs.io/en/latest/hyper-parameter-search.html#avoid-repeated-work) to avoid re-fitting estimators multiple times on the same set of data.
It will also provide fine-grained diagnostics in the [distributed scheduler UI](https://distributed.readthedocs.io/en/latest/web.html).

Alternatively, Dask implements a joblib backend.
You can instruct TPOT to use the distribued backend during training by specifying a ``joblib.parallel_backend``:

```python
from sklearn.externals import joblib
import distributed.joblib
from dask.distributed import Client

# connect to the cluster
client = Client('schedueler-address')

# create the estimator normally
estimator = TPOTClassifier(n_jobs=-1)

# perform the fit in this context manager
with joblib.parallel_backend("dask"):
    estimator.fit(X, y)
```

See [dask's distributed joblib integration](https://distributed.readthedocs.io/en/latest/joblib.html) for more.
