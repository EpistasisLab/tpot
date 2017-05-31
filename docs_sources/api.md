# Classification

<pre><em>class</em> tpot.<strong style="color:#008AB8">TPOTClassifier</strong>(<em><strong>generations</strong>=100, <strong>population_size</strong>=100,
                          <strong>offspring_size</strong>=None, <strong>mutation_rate</strong>=0.9,
                          <strong>crossover_rate</strong>=0.1, <strong>scoring='accuracy'</strong>, <strong>cv</strong>=5,
                          <strong>subsample</strong>=1.0, <strong>n_jobs</strong>=1,
                          <strong>max_time_mins</strong>=None, <strong>max_eval_time_mins</strong>=5,
                          <strong>random_state</strong>=None, <strong>config_dict</strong>=None,
                          <strong>warm_start</strong>=False, <strong>verbosity</strong>=0,
                          <strong>disable_update_check</strong>=False</em>)</pre>
<div align="right"><a href="https://github.com/rhiever/tpot/blob/master/tpot/base.py#L77">source</a></div>

Automated machine learning for supervised classification tasks.

The TPOTClassifier performs an intelligent search over machine learning pipelines that can contain supervised machine learning models, preprocessors, feature selection techniques, and any other estimator or transformer that follows the [scikit-learn API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects). The TPOTClassifier will also search over the hyperparameters of all objects in the pipeline.

By default, the TPOTClassifier will search over a broad range of supervised learning models, transformers, and their parameters. However, the models, transformers, and parameters that the TPOTClassifier searches over can be fully customized using the `config_dict` parameter.

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
'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'
<br /><br/>
If you would like to use a custom scoring function, you can pass a callable function to this parameter with the signature <em>scorer(y_true, y_pred)</em>. See the section on <a href="/using/#scoring-functions">scoring functions</a> for more details.
<br /><br />
TPOT assumes that any function with "error" or "loss" in the name is meant to be minimized, whereas any other functions will be maximized.
</blockquote>

<strong>cv</strong>: int, cross-validation generator, or an iterable, optional (default=5)
<blockquote>
Cross-validation strategy used when evaluating pipelines.
<br /><br />
Possible inputs:
<ul>
<li>integer, to specify the number of folds in a StratifiedKFold, or</li>
<li>An object to be used as a cross-validation generator</li>
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
Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
</blockquote>

<strong>config_dict</strong>: Python dictionary, string, or None, optional (default=None)
<blockquote>
A configuration dictionary for customizing the operators and parameters that TPOT searches in the optimization process.
<br /><br />
Possible inputs are:
<ul>
<li>Python dictionary, TPOT will use your custom configuration,</li>
<li>string, TPOT will use one of the built-in configurations, or</li>
<li>None, TPOT will use the default TPOTClassifier configuration.</li>
</ul>
See the <a href="/using/#built-in-tpot-configurations">built-in configurations</a> section for the list of configurations included with TPOT, and the <a href="/using/#customizing-tpots-operators-and-parameters">custom configuration</a> section for more information and examples of how to create your own TPOT configurations.
</blockquote>

<strong>warm_start</strong>: boolean, optional (default=False)
<blockquote>
Flag indicating whether the TPOT instance will reuse the population from previous calls to <em>fit()</em>.
<br /><br />
Setting <em>warm_start</em>=True can be useful for running TPOT for a short time on a dataset, checking the results, then resuming the TPOT run from where it left off.
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
<strong>_fitted_pipeline</strong>: scikit-learn Pipeline object
<blockquote>
The best pipeline that TPOT discovered during the pipeline optimization process, fitted on the entire training dataset.
</blockquote>

<strong>_pareto_front_fitted_pipelines</strong>: Python dictionary
<blockquote>
Dictionary containing the all pipelines on the TPOT Pareto front, where the key is the string representation of the pipeline and the value is the corresponding pipeline fitted on the entire training dataset.
<br /><br />
The TPOT Pareto front provides a trade-off between pipeline complexity (i.e., the number of steps in the pipeline) and the predictive performance of the pipeline.
<br /><br />
Note: <em>_pareto_front_fitted_pipelines</em> is only available when <em>verbosity</em>=3.
</blockquote>

<strong>_evaluated_individuals</strong>: Python dictionary
<blockquote>
Dictionary containing all pipelines that were evaluated during the pipeline optimization process, where the key is the string representation of the pipeline and the value is a tuple containing (# of steps in pipeline, quality metric for the pipeline).
<br /><br />
This attribute is primarily for internal use, but may be useful for looking at the other pipelines that TPOT evaluated.
</blockquote>
</td>
<tr>
</table>

# Regression

TPOTRegressor

