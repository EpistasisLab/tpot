# Classification

<pre><em>class</em> tpot.<strong style="color:#008AB8">TPOTClassifier</strong>(<em><strong>generations</strong>=100, <strong>population_size</strong>=100,
                          <strong>offspring_size</strong>=None, <strong>mutation_rate</strong>=0.9,
                          <strong>crossover_rate</strong>=0.1,
                          <strong>scoring</strong>='accuracy', <strong>cv</strong>=5,
                          <strong>subsample</strong>=1.0, <strong>n_jobs</strong>=1,
                          <strong>max_time_mins</strong>=None, <strong>max_eval_time_mins</strong>=5,
                          <strong>random_state</strong>=None, <strong>config_dict</strong>=None,
                          <strong>warm_start</strong>=False,
                          <strong>memory</strong>=None,
                          <strong>periodic_checkpoint_folder</strong>=None,
                          <strong>verbosity</strong>=0,
                          <strong>disable_update_check</strong>=False</em>)</pre>
<div align="right"><a href="https://github.com/rhiever/tpot/blob/master/tpot/base.py">source</a></div>

Automated machine learning for supervised classification tasks.

The TPOTClassifier performs an intelligent search over machine learning pipelines that can contain supervised classification models,
preprocessors, feature selection techniques, and any other estimator or transformer that follows the [scikit-learn API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).
The TPOTClassifier will also search over the hyperparameters of all objects in the pipeline.

By default, TPOTClassifier will search over a broad range of supervised classification algorithms, transformers, and their parameters.
However, the algorithms, transformers, and hyperparameters that the TPOTClassifier searches over can be fully customized using the `config_dict` parameter.

Read more in the [User Guide](using/#tpot-with-code).

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>generations</strong>: int, optional (default=100)
<blockquote>
Number of iterations to the run pipeline optimization process. Must be a positive number.
<br /><br />
Generally, TPOT will work better when you give it more generations (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate <em>population_size</em> + <em>generations</em> × <em>offspring_size</em> pipelines in total.
</blockquote>

<strong>population_size</strong>: int, optional (default=100)
<blockquote>
Number of individuals to retain in the genetic programming population every generation. Must be a positive number.
<br /><br />
Generally, TPOT will work better when you give it more individuals with which to optimize the pipeline.
</blockquote>

<strong>offspring_size</strong>: int, optional (default=100)
<blockquote>
Number of offspring to produce in each genetic programming generation. Must be a positive number.
</blockquote>

<strong>mutation_rate</strong>: float, optional (default=0.9)
<blockquote>
Mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the GP algorithm how many pipelines to apply random changes to every generation.
<br /><br />
<em>mutation_rate</em> + <em>crossover_rate</em> cannot exceed 1.0.
<br /><br />
We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.
</blockquote>

<strong>crossover_rate</strong>: float, optional (default=0.1)
<blockquote>
Crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the genetic programming algorithm how many pipelines to "breed" every generation.
<br /><br />
<em>mutation_rate</em> + <em>crossover_rate</em> cannot exceed 1.0.
<br /><br />
We recommend using the default parameter unless you understand how the crossover rate affects GP algorithms.
</blockquote>

<strong>scoring</strong>: string or callable, optional (default='accuracy')
<blockquote>
Function used to evaluate the quality of a given pipeline for the classification problem. The following built-in scoring functions can be used:
<br /><br/>
'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss','precision',
'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'
<br /><br/>
If you would like to use a custom scorer, you can pass the callable object/function with signature <em>scorer(estimator, X, y)</em>.
<br /><br/>
If you would like to use a metric function, you can pass the callable function to this parameter with the signature <em>score_func(y_true, y_pred)</em>. TPOT assumes that any function with "error" or "loss" in the function name is meant to be minimized, whereas any other functions will be maximized. This scoring type was deprecated in version 0.9.1 and will be removed in version 0.11.
<br /><br/>
See the section on <a href="../using/#scoring-functions">scoring functions</a> for more details.

</blockquote>

<strong>cv</strong>: int, cross-validation generator, or an iterable, optional (default=5)
<blockquote>
Cross-validation strategy used when evaluating pipelines.
<br /><br />
Possible inputs:
<ul>
<li>integer, to specify the number of folds in a StratifiedKFold,</li>
<li>An object to be used as a cross-validation generator, or</li>
<li>An iterable yielding train/test splits.</li>
</blockquote>

<strong>subsample</strong>: float, optional (default=1.0)
<blockquote>
Fraction of training samples that are used during the TPOT optimization process. Must be in the range (0.0, 1.0].
<br /><br />
Setting <em>subsample</em>=0.5 tells TPOT to use a random subsample of half of the training data. This subsample will remain the same during the entire pipeline optimization process.
</blockquote>

<strong>n_jobs</strong>: integer, optional (default=1)
<blockquote>
Number of processes to use in parallel for evaluating pipelines during the TPOT optimization process.
<br /><br />
Setting <em>n_jobs</em>=-1 will use as many cores as available on the computer. Beware that using multiple processes on the same machine may cause memory issues for large datasets
</blockquote>

<strong>max_time_mins</strong>: integer or None, optional (default=None)
<blockquote>
How many minutes TPOT has to optimize the pipeline.
<br /><br />
If not None, this setting will override the <em>generations</em> parameter and allow TPOT to run until <em>max_time_mins</em> minutes elapse.
</blockquote>

<strong>max_eval_time_mins</strong>: integer, optional (default=5)
<blockquote>
How many minutes TPOT has to evaluate a single pipeline.
<br /><br />
Setting this parameter to higher values will allow TPOT to evaluate more complex pipelines, but will also allow TPOT to run longer. Use this parameter to help prevent TPOT from wasting time on evaluating time-consuming pipelines.
</blockquote>

<strong>random_state</strong>: integer or None, optional (default=None)
<blockquote>
The seed of the pseudo random number generator used in TPOT.
<br /><br />
Use this parameter to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
</blockquote>

<strong>config_dict</strong>: Python dictionary, string, or None, optional (default=None)
<blockquote>
A configuration dictionary for customizing the operators and parameters that TPOT searches in the optimization process.
<br /><br />
Possible inputs are:
<ul>
<li>Python dictionary, TPOT will use your custom configuration,</li>
<li>string 'TPOT light', TPOT will use a built-in configuration with only fast models and preprocessors, or</li>
<li>string 'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies, or</li>
<li>string 'TPOT sparse': TPOT will use a configuration dictionary with a one-hot encoder and the operators normally included in TPOT that also support sparse matrices, or</li>
<li>None, TPOT will use the default TPOTClassifier configuration.</li>
</ul>
See the <a href="../using/#built-in-tpot-configurations">built-in configurations</a> section for the list of configurations included with TPOT, and the <a href="../using/#customizing-tpots-operators-and-parameters">custom configuration</a> section for more information and examples of how to create your own TPOT configurations.
</blockquote>

<strong>warm_start</strong>: boolean, optional (default=False)
<blockquote>
Flag indicating whether the TPOT instance will reuse the population from previous calls to <em>fit()</em>.
<br /><br />
Setting <em>warm_start</em>=True can be useful for running TPOT for a short time on a dataset, checking the results, then resuming the TPOT run from where it left off.
</blockquote>

<strong>memory</strong>: a sklearn.external.joblib.Memory object or string, optional (default=None)
<blockquote>
If supplied, pipeline will cache each transformer after calling fit. This feature is used to avoid computing the fit transformers within a pipeline if the parameters and input data are identical with another fitted pipeline during optimization process. More details about memory caching in [scikit-learn documentation](http://scikit-learn.org/stable/modules/pipeline.html#caching-transformers-avoid-repeated-computation)
<br /><br />
Possible inputs are:
<ul>
<li>String 'auto': TPOT uses memory caching with a temporary directory and cleans it up upon shutdown, or</li>
<li>Path of a caching directory, TPOT uses memory caching with the provided directory and TPOT does NOT clean the caching directory up upon shutdown, or</li>
<li>Memory object, TPOT uses the instance of sklearn.external.joblib.Memory for memory caching and TPOT does NOT clean the caching directory up upon shutdown, or</li>
<li>None, TPOT does not use memory caching.</li>
</ul>
</blockquote>

<strong>periodic_checkpoint_folder</strong>: path string, optional (default: None)
<blockquote>
If supplied, a folder in which TPOT will periodically save the best pipeline so far while optimizing.<br /><br />
Currently once per generation but not more often than once per 30 seconds.<br /><br />
Useful in multiple cases:
<ul>
<li>Sudden death before TPOT could save optimized pipeline</li>
<li>Track its progress</li>
<li>Grab pipelines while it's still optimizing</li>
</ul>
</blockquote>

<strong>early_stop</strong>: integer, optional (default: None)
<blockquote>
How many generations TPOT checks whether there is no improvement in optimization process.
<br /><br />
Ends the optimization process if there is no improvement in the given number of generations.
</blockquote>

<strong>verbosity</strong>: integer, optional (default=0)
<blockquote>
How much information TPOT communicates while it's running.
<br /><br />
Possible inputs are:
<ul>
<li>0, TPOT will print nothing,</li>
<li>1, TPOT will print minimal information,</li>
<li>2, TPOT will print more information and provide a progress bar, or</li>
<li>3, TPOT will print everything and provide a progress bar.</li>
</ul>
</blockquote>

<strong>disable_update_check</strong>: boolean, optional (default=False)
<blockquote>
Flag indicating whether the TPOT version checker should be disabled.
<br /><br />
The update checker will tell you when a new version of TPOT has been released.
</blockquote>
</td>
</tr>

<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
<strong>fitted_pipeline_</strong>: scikit-learn Pipeline object
<blockquote>
The best pipeline that TPOT discovered during the pipeline optimization process, fitted on the entire training dataset.
</blockquote>

<strong>pareto_front_fitted_pipelines_</strong>: Python dictionary
<blockquote>
Dictionary containing the all pipelines on the TPOT Pareto front, where the key is the string representation of the pipeline and the value is the corresponding pipeline fitted on the entire training dataset.
<br /><br />
The TPOT Pareto front provides a trade-off between pipeline complexity (i.e., the number of steps in the pipeline) and the predictive performance of the pipeline.
<br /><br />
Note: <em>pareto_front_fitted_pipelines_</em> is only available when <em>verbosity</em>=3.
</blockquote>

<strong>evaluated_individuals_</strong>: Python dictionary
<blockquote>
Dictionary containing all pipelines that were evaluated during the pipeline optimization process, where the key is the string representation of the pipeline and the value is a tuple containing (# of steps in pipeline, accuracy metric for the pipeline).
<br /><br />
This attribute is primarily for internal use, but may be useful for looking at the other pipelines that TPOT evaluated.
</blockquote>
</td>
<tr>
</table>

<strong>Example</strong>

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

<strong>Functions</strong>

<table width="100%">
<tr>
<td width="25%"><a href="#tpotclassifier-fit">fit</a>(features, classes[, sample_weight, groups])</td>
<td>Run the TPOT optimization process on the given training data.</td>
</tr>

<tr>
<td><a href="#tpotclassifier-predict">predict</a>(features)</td>
<td>Use the optimized pipeline to predict the classes for a feature set.</td>
</tr>

<tr>
<td><a href="#tpotclassifier-predict-proba">predict_proba</a>(features)</td>
<td>Use the optimized pipeline to estimate the class probabilities for a feature set.</td>
</tr>

<tr>
<td><a href="#tpotclassifier-score">score</a>(testing_features, testing_classes)</td>
<td>Returns the optimized pipeline's score on the given testing data using the user-specified scoring function.</td>
</tr>

<tr>
<td><a href="#tpotclassifier-export">export</a>(output_file_name)</td>
<td>Export the optimized pipeline as Python code.</td>
</tr>
</table>


<a name="tpotclassifier-fit"></a>
```Python
fit(features, classes, sample_weight=None, groups=None)
```

<div style="padding-left:5%" width="100%">
Run the TPOT optimization process on the given training data.
<br /><br />
Uses genetic programming to optimize a machine learning pipeline that maximizes the score on the provided features and target. This pipeline optimization procedure uses internal k-fold cross-validaton to avoid overfitting on the provided data. At the end of the pipeline optimization procedure, the best pipeline is then trained on the entire set of provided samples.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix
<br /><br />
TPOT and all scikit-learn algorithms assume that the features will be numerical and there will be no missing values.
As such, when a feature matrix is provided to TPOT, all missing values will automatically be replaced (i.e., imputed)
using <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html">median value imputation</a>.
<br /><br />
If you wish to use a different imputation strategy than median imputation, please make sure to apply imputation to your feature set prior to passing it to TPOT.
</blockquote>

<strong>classes</strong>: array-like {n_samples}
<blockquote>
List of class labels for prediction
</blockquote>

<strong>sample_weight</strong>: array-like {n_samples}, optional
<blockquote>
Per-sample weights. Higher weights force TPOT to put more emphasis on those points.
</blockquote>

<strong>groups</strong>: array-like, with shape {n_samples, }, optional
<blockquote>
Group labels for the samples used when performing cross-validation.
<br /><br />
This parameter should only be used in conjunction with sklearn's Group cross-validation functions, such as <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html">sklearn.model_selection.GroupKFold</a>.
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>self</strong>: object
<blockquote>
Returns a copy of the fitted TPOT object
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotclassifier-predict"></a>
```Python
predict(features)
```

<div style="padding-left:5%" width="100%">
Use the optimized pipeline to predict the classes for a feature set.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>predictions</strong>: array-like {n_samples}
<blockquote>
Predicted classes for the samples in the feature matrix
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotclassifier-predict-proba"></a>
```Python
predict_proba(features)
```

<div style="padding-left:5%" width="100%">
Use the optimized pipeline to estimate the class probabilities for a feature set.
<br /><br />
Note: This function will only work for pipelines whose final classifier supports the <em>predict_proba</em> function. TPOT will raise an error otherwise.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>predictions</strong>: array-like {n_samples, n_classes}
<blockquote>
The class probabilities of the input samples
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotclassifier-score"></a>
```Python
score(testing_features, testing_classes)
```

<div style="padding-left:5%" width="100%">
Returns the optimized pipeline's score on the given testing data using the user-specified scoring function.
<br /><br />
The default scoring function for TPOTClassifier is 'accuracy'.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>testing_features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix of the testing set
</blockquote>

<strong>testing_classes</strong>: array-like {n_samples}
<blockquote>
List of class labels for prediction in the testing set
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>accuracy_score</strong>: float
<blockquote>
The estimated test set accuracy according to the user-specified scoring function.
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotclassifier-export"></a>
```Python
export(output_file_name)
```

<div style="padding-left:5%" width="100%">
Export the optimized pipeline as Python code.
<br /><br />
See the <a href="../using/#tpot-with-code">usage documentation</a> for example usage of the export function.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>output_file_name</strong>: string
<blockquote>
String containing the path and file name of the desired output file
</blockquote>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
Does not return anything
</td>
</tr>
</table>
</div>




# Regression

<pre><em>class</em> tpot.<strong style="color:#008AB8">TPOTRegressor</strong>(<em><strong>generations</strong>=100, <strong>population_size</strong>=100,
                         <strong>offspring_size</strong>=None, <strong>mutation_rate</strong>=0.9,
                         <strong>crossover_rate</strong>=0.1,
                         <strong>scoring</strong>='neg_mean_squared_error', <strong>cv</strong>=5,
                         <strong>subsample</strong>=1.0, <strong>n_jobs</strong>=1,
                         <strong>max_time_mins</strong>=None, <strong>max_eval_time_mins</strong>=5,
                         <strong>random_state</strong>=None, <strong>config_dict</strong>=None,
                         <strong>warm_start</strong>=False,
                         <strong>memory</strong>=None,
                         <strong>periodic_checkpoint_folder</strong>=None,
                         <strong>verbosity</strong>=0,
                         <strong>disable_update_check</strong>=False</em>)</pre>
<div align="right"><a href="https://github.com/rhiever/tpot/blob/master/tpot/base.py">source</a></div>

Automated machine learning for supervised regression tasks.

The TPOTRegressor performs an intelligent search over machine learning pipelines that can contain supervised regression models,
preprocessors, feature selection techniques, and any other estimator or transformer that follows the [scikit-learn API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).
The TPOTRegressor will also search over the hyperparameters of all objects in the pipeline.

By default, TPOTRegressor will search over a broad range of supervised regression models, transformers, and their hyperparameters.
However, the models, transformers, and parameters that the TPOTRegressor searches over can be fully customized using the `config_dict` parameter.

Read more in the [User Guide](using/#tpot-with-code).

<table>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>generations</strong>: int, optional (default=100)
<blockquote>
Number of iterations to the run pipeline optimization process. Must be a positive number.
<br /><br />
Generally, TPOT will work better when you give it more generations (and therefore time) to optimize the pipeline.
<br /><br />
TPOT will evaluate <em>population_size</em> + <em>generations</em> × <em>offspring_size</em> pipelines in total.
</blockquote>

<strong>population_size</strong>: int, optional (default=100)
<blockquote>
Number of individuals to retain in the genetic programming population every generation. Must be a positive number.
<br /><br />
Generally, TPOT will work better when you give it more individuals with which to optimize the pipeline.
</blockquote>

<strong>offspring_size</strong>: int, optional (default=100)
<blockquote>
Number of offspring to produce in each genetic programming generation. Must be a positive number.
</blockquote>

<strong>mutation_rate</strong>: float, optional (default=0.9)
<blockquote>
Mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the GP algorithm how many pipelines to apply random changes to every generation.
<br /><br />
<em>mutation_rate</em> + <em>crossover_rate</em> cannot exceed 1.0.
<br /><br />
We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.
</blockquote>

<strong>crossover_rate</strong>: float, optional (default=0.1)
<blockquote>
Crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the genetic programming algorithm how many pipelines to "breed" every generation.
<br /><br />
<em>mutation_rate</em> + <em>crossover_rate</em> cannot exceed 1.0.
<br /><br />
We recommend using the default parameter unless you understand how the crossover rate affects GP algorithms.
</blockquote>

<strong>scoring</strong>: string or callable, optional (default='neg_mean_squared_error')
<blockquote>
Function used to evaluate the quality of a given pipeline for the regression problem. The following built-in scoring functions can be used:
<br /><br/>
'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'
<br /><br/>
Note that we recommend using the <em>neg</em> version of mean squared error and related metrics so TPOT will minimize (instead of maximize) the metric.
<br /><br/>
If you would like to use a custom scorer, you can pass the callable object/function with signature <em>scorer(estimator, X, y)</em>.
<br /><br/>
If you would like to use a metric function, you can pass the callable function to this parameter with the signature <em>score_func(y_true, y_pred)</em>. TPOT assumes that any function with "error" or "loss" in the function name is meant to be minimized, whereas any other functions will be maximized. This scoring type was deprecated in version 0.9.1 and will be removed in version 0.11.
<br /><br/>
See the section on <a href="../using/#scoring-functions">scoring functions</a> for more details.
</blockquote>

<strong>cv</strong>: int, cross-validation generator, or an iterable, optional (default=5)
<blockquote>
Cross-validation strategy used when evaluating pipelines.
<br /><br />
Possible inputs:
<ul>
<li>integer, to specify the number of folds in a KFold,</li>
<li>An object to be used as a cross-validation generator, or</li>
<li>An iterable yielding train/test splits.</li>
</ul>
</blockquote>

<strong>subsample</strong>: float, optional (default=1.0)
<blockquote>
Fraction of training samples that are used during the TPOT optimization process. Must be in the range (0.0, 1.0].
<br /><br />
Setting <em>subsample</em>=0.5 tells TPOT to use a random subsample of half of the training data. This subsample will remain the same during the entire pipeline optimization process.
</blockquote>

<strong>n_jobs</strong>: integer, optional (default=1)
<blockquote>
Number of processes to use in parallel for evaluating pipelines during the TPOT optimization process.
<br /><br />
Setting <em>n_jobs</em>=-1 will use as many cores as available on the computer. Beware that using multiple processes on the same machine may cause memory issues for large datasets
</blockquote>

<strong>max_time_mins</strong>: integer or None, optional (default=None)
<blockquote>
How many minutes TPOT has to optimize the pipeline.
<br /><br />
If not None, this setting will override the <em>generations</em> parameter and allow TPOT to run until <em>max_time_mins</em> minutes elapse.
</blockquote>

<strong>max_eval_time_mins</strong>: integer, optional (default=5)
<blockquote>
How many minutes TPOT has to evaluate a single pipeline.
<br /><br />
Setting this parameter to higher values will allow TPOT to evaluate more complex pipelines, but will also allow TPOT to run longer. Use this parameter to help prevent TPOT from wasting time on evaluating time-consuming pipelines.
</blockquote>

<strong>random_state</strong>: integer or None, optional (default=None)
<blockquote>
The seed of the pseudo random number generator used in TPOT.
<br /><br />
Use this parameter to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
</blockquote>

<strong>config_dict</strong>: Python dictionary, string, or None, optional (default=None)
<blockquote>
A configuration dictionary for customizing the operators and parameters that TPOT searches in the optimization process.
<br /><br />
Possible inputs are:
<ul>
<li>Python dictionary, TPOT will use your custom configuration,</li>
<li>string 'TPOT light', TPOT will use a built-in configuration with only fast models and preprocessors, or</li>
<li>string 'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies, or</li>
<li>string 'TPOT sparse': TPOT will use a configuration dictionary with a one-hot encoder and the operators normally included in TPOT that also support sparse matrices, or</li>
<li>None, TPOT will use the default TPOTRegressor configuration.</li>
</ul>
See the <a href="../using/#built-in-tpot-configurations">built-in configurations</a> section for the list of configurations included with TPOT, and the <a href="../using/#customizing-tpots-operators-and-parameters">custom configuration</a> section for more information and examples of how to create your own TPOT configurations.
</blockquote>

<strong>warm_start</strong>: boolean, optional (default=False)
<blockquote>
Flag indicating whether the TPOT instance will reuse the population from previous calls to <em>fit()</em>.
<br /><br />
Setting <em>warm_start</em>=True can be useful for running TPOT for a short time on a dataset, checking the results, then resuming the TPOT run from where it left off.
</blockquote>

<strong>memory</strong>: a sklearn.external.joblib.Memory object or string, optional (default=None)
<blockquote>
If supplied, pipeline will cache each transformer after calling fit. This feature is used to avoid computing the fit transformers within a pipeline if the parameters and input data are identical with another fitted pipeline during optimization process. More details about memory caching in [scikit-learn documentation](http://scikit-learn.org/stable/modules/pipeline.html#caching-transformers-avoid-repeated-computation)
<br /><br />
Possible inputs are:
<ul>
<li>String 'auto': TPOT uses memory caching with a temporary directory and cleans it up upon shutdown, or</li>
<li>Path of a caching directory, TPOT uses memory caching with the provided directory and TPOT does NOT clean the caching directory up upon shutdown, or</li>
<li>Memory object, TPOT uses the instance of sklearn.external.joblib.Memory for memory caching and TPOT does NOT clean the caching directory up upon shutdown, or</li>
<li>None, TPOT does not use memory caching.</li>
</ul>
</blockquote>

<strong>periodic_checkpoint_folder</strong>: path string, optional (default: None)
<blockquote>
If supplied, a folder in which TPOT will periodically save the best pipeline so far while optimizing.<br /><br />
Currently once per generation but not more often than once per 30 seconds.<br /><br />
Useful in multiple cases:
<ul>
<li>Sudden death before TPOT could save optimized pipeline</li>
<li>Track its progress</li>
<li>Grab pipelines while it's still optimizing</li>
</ul>
</blockquote>

<strong>early_stop</strong>: integer, optional (default: None)
<blockquote>
How many generations TPOT checks whether there is no improvement in optimization process.
<br /><br />
Ends the optimization process if there is no improvement in the given number of generations.
</blockquote>

<strong>verbosity</strong>: integer, optional (default=0)
<blockquote>
How much information TPOT communicates while it's running.
<br /><br />
Possible inputs are:
<ul>
<li>0, TPOT will print nothing,</li>
<li>1, TPOT will print minimal information,</li>
<li>2, TPOT will print more information and provide a progress bar, or</li>
<li>3, TPOT will print everything and provide a progress bar.</li>
</ul>
</blockquote>

<strong>disable_update_check</strong>: boolean, optional (default=False)
<blockquote>
Flag indicating whether the TPOT version checker should be disabled.
<br /><br />
The update checker will tell you when a new version of TPOT has been released.
</blockquote>
</td>
</tr>

<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Attributes:</strong></td>
<td width="80%" style="background:white;">
<strong>fitted_pipeline_</strong>: scikit-learn Pipeline object
<blockquote>
The best pipeline that TPOT discovered during the pipeline optimization process, fitted on the entire training dataset.
</blockquote>

<strong>pareto_front_fitted_pipelines_</strong>: Python dictionary
<blockquote>
Dictionary containing the all pipelines on the TPOT Pareto front, where the key is the string representation of the pipeline and the value is the corresponding pipeline fitted on the entire training dataset.
<br /><br />
The TPOT Pareto front provides a trade-off between pipeline complexity (i.e., the number of steps in the pipeline) and the predictive performance of the pipeline.
<br /><br />
Note: <em>_pareto_front_fitted_pipelines</em> is only available when <em>verbosity</em>=3.
</blockquote>

<strong>evaluated_individuals_</strong>: Python dictionary
<blockquote>
Dictionary containing all pipelines that were evaluated during the pipeline optimization process, where the key is the string representation of the pipeline and the value is a tuple containing (# of steps in pipeline, accuracy metric for the pipeline).
<br /><br />
This attribute is primarily for internal use, but may be useful for looking at the other pipelines that TPOT evaluated.
</blockquote>
</td>
<tr>
</table>

<strong>Example</strong>

```Python
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```

<strong>Functions</strong>

<table width="100%">
<tr>
<td width="25%"><a href="#tpotregressor-fit">fit</a>(features, target[, sample_weight, groups])</td>
<td>Run the TPOT optimization process on the given training data.</td>
</tr>

<tr>
<td><a href="#tpotregressor-predict">predict</a>(features)</td>
<td>Use the optimized pipeline to predict the target values for a feature set.</td>
</tr>

<tr>
<td><a href="#tpotregressor-score">score</a>(testing_features, testing_target)</td>
<td>Returns the optimized pipeline's score on the given testing data using the user-specified scoring function.</td>
</tr>

<tr>
<td><a href="#tpotregressor-export">export</a>(output_file_name)</td>
<td>Export the optimized pipeline as Python code.</td>
</tr>
</table>


<a name="tpotregressor-fit"></a>
```Python
fit(features, target, sample_weight=None, groups=None)
```

<div style="padding-left:5%" width="100%">
Run the TPOT optimization process on the given training data.
<br /><br />
Uses genetic programming to optimize a machine learning pipeline that maximizes the score on the provided features and target. This pipeline optimization procedure uses internal k-fold cross-validaton to avoid overfitting on the provided data. At the end of the pipeline optimization procedure, the best pipeline is then trained on the entire set of provided samples.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix
<br /><br />
TPOT and all scikit-learn algorithms assume that the features will be numerical and there will be no missing values.
As such, when a feature matrix is provided to TPOT, all missing values will automatically be replaced (i.e., imputed)
using <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html">median value imputation</a>.
<br /><br />
If you wish to use a different imputation strategy than median imputation, please make sure to apply imputation to your feature set prior to passing it to TPOT.
</blockquote>

<strong>target</strong>: array-like {n_samples}
<blockquote>
List of target labels for prediction
</blockquote>

<strong>sample_weight</strong>: array-like {n_samples}, optional
<blockquote>
Per-sample weights. Higher weights force TPOT to put more emphasis on those points.
</blockquote>

<strong>groups</strong>: array-like, with shape {n_samples, }, optional
<blockquote>
Group labels for the samples used when performing cross-validation.
<br /><br />
This parameter should only be used in conjunction with sklearn's Group cross-validation functions, such as <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html">sklearn.model_selection.GroupKFold</a>.
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>self</strong>: object
<blockquote>
Returns a copy of the fitted TPOT object
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotregressor-predict"></a>
```Python
predict(features)
```

<div style="padding-left:5%" width="100%">
Use the optimized pipeline to predict the target values for a feature set.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>predictions</strong>: array-like {n_samples}
<blockquote>
Predicted target values for the samples in the feature matrix
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotregressor-score"></a>
```Python
score(testing_features, testing_target)
```

<div style="padding-left:5%" width="100%">
Returns the optimized pipeline's score on the given testing data using the user-specified scoring function.
<br /><br />
The default scoring function for TPOTClassifier is 'mean_squared_error'.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>testing_features</strong>: array-like {n_samples, n_features}
<blockquote>
Feature matrix of the testing set
</blockquote>

<strong>testing_target</strong>: array-like {n_samples}
<blockquote>
List of target labels for prediction in the testing set
</blockquote>
</td>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
<strong>accuracy_score</strong>: float
<blockquote>
The estimated test set accuracy according to the user-specified scoring function.
</blockquote>
</td>
</tr>
</table>
</div>


<a name="tpotregressor-export"></a>
```Python
export(output_file_name)
```

<div style="padding-left:5%" width="100%">
Export the optimized pipeline as Python code.
<br /><br />
See the <a href="../using/#tpot-with-code">usage documentation</a> for example usage of the export function.
<br /><br />
<table width="100%">
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Parameters:</strong></td>
<td width="80%" style="background:white;">
<strong>output_file_name</strong>: string
<blockquote>
String containing the path and file name of the desired output file
</blockquote>
</tr>
<tr>
<td width="20%" style="vertical-align:top; background:#F5F5F5;"><strong>Returns:</strong></td>
<td width="80%" style="background:white;">
Does not return anything
</td>
</tr>
</table>
</div>
