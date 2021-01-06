# Release Notes

## Version 0.11.7

- Fix compatibility issue with scikit-learn 0.24 and xgboost 1.3.0
- Fix a bug causing that TPOT does not work when classifying more than 50 classes
- Add initial support `Resampler` from `imblearn`
- Fix minor bugs


## Version 0.11.6

- Fix a bug causing point mutation function does not work properly with using `template` option
- Add a new built configuration called "TPOT cuML" which TPOT will search over a restricted configuration using the GPU-accelerated estimators in [RAPIDS cuML](https://github.com/rapidsai/cuml) and [DMLC XGBoost](https://github.com/dmlc/xgboost). **This configuration requires an NVIDIA Pascal architecture or better GPU with [compute capability 6.0+](https://developer.nvidia.com/cuda-gpus), and that the library cuML is installed.**
- Add string path support for log/log_file parameter
- Fix a bug in version 0.11.5 causing no update in stdout after each generation
- Fix minor bugs


## Version 0.11.5

- Make `Pytorch` as an optional dependency
- Refine installation documentation

## Version 0.11.4

- Add a new built configuration "TPOT NN" which includes all operators in "Default TPOT" plus additional neural network estimators written in PyTorch (currently `tpot.builtins.PytorchLRClassifier` and `tpot.builtins.PytorchMLPClassifier` for classification tasks only)
- Refine `log_file` parameter's behavior

## Version 0.11.3

- Fix a bug in TPOTRegressor in v0.11.2
- Add `-log` option in command line interface to save process log to a file.

## Version 0.11.2

- Fix `early_stop` parameter does not work properly
- TPOT built-in `OneHotEncoder` can refit to different datasets
- Fix the issue that the attribute `evaluated_individuals_` cannot record correct generation info.
- Add a new parameter `log_file` to output logs to a file instead of `sys.stdout`
- Fix some code quality issues and mistakes in documentations
- Fix minor bugs

## Version 0.11.1

- Fix compatibility issue with scikit-learn v0.22
- `warm_start` now saves both Primitive Sets and evaluated_pipelines_ from previous runs;
- Fix the error that TPOT assign wrong fitness scores to non-evaluated pipelines (interrupted by `max_min_mins` or `KeyboardInterrupt`) ;
- Fix the bug that mutation operator cannot generate new pipeline when template is not default value and `warm_start` is True;
- Fix the bug that `max_time_mins` cannot stop optimization process when search space is limited.  
- Fix a bug in exported codes when the exported pipeline is only 1 estimator
- Fix spelling mistakes in documentations
- Fix some code quality issues

## Version 0.11.0

- **Support for Python 3.4 and below has been officially dropped.** Also support for scikit-learn 0.20 or below has been dropped.
- The support of a metric function with the signature `score_func(y_true, y_pred)` for `scoring parameter` has been dropped.
- Refine `StackingEstimator` for not stacking NaN/Infinity predication probabilities.
- Fix a bug that population doesn't persist by `warm_start=True` when `max_time_mins` is not default value.
- Now the `random_state` parameter in TPOT is used for pipeline evaluation instead of using a fixed random seed of 42 before. The `set_param_recursive` function has been moved to `export_utils.py` and it can be used in exported codes for setting `random_state` recursively in scikit-learn Pipeline. It is used to set `random_state` in `fitted_pipeline_` attribute and exported pipelines.
- TPOT can independently use `generations` and `max_time_mins` to limit the optimization process through using one of the parameters or both.
- `.export()` function will return string of exported pipeline if output filename is not specified.
- Add [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) and [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) into TPOT default configs.
- Documentation has been updated
- Fix minor bugs.

## Version 0.10.2

- **TPOT v0.10.2 is the last version to support Python 2.7 and Python 3.4.**
- Minor updates for fixing compatibility issues with the latest version of scikit-learn (version > 0.21) and xgboost (v0.90)
- Default value of `template` parameter is changed to `None` instead.
- Fix errors in documentation

## Version 0.10.1

- Add `data_file_path` option into `expert` function for replacing `'PATH/TO/DATA/FILE'` to customized dataset path in exported scripts. (Related issue #838)
- Change python version in CI tests to 3.7
- Add CI tests for macOS.

## Version 0.10.0

- Add a new `template` option to specify a desired structure for machine learning pipeline in TPOT. Check [TPOT API](https://epistasislab.github.io/tpot/api/) (it will be updated once it is merge to master branch).
- Add `FeatureSetSelector` operator into TPOT for feature selection based on *priori* export knowledge. Please check our [preprint paper](https://www.biorxiv.org/content/10.1101/502484v1.article-info) for more details (*Note: it was named `DatasetSelector` in 1st version paper but we will rename to FeatureSetSelector in next version of the paper*)
- Refine `n_jobs` parameter to accept value below -1. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
- Now `memory`  parameter can create memory cache directory if it does not exist.
- Fix minor bugs.

## Version 0.9.6

- Fix a bug causing that `max_time_mins` parameter doesn't work when `use_dask=True` in TPOT 0.9.5
- Now TPOT saves best pareto values best pareto pipeline s in checkpoint folder
- TPOT raises `ImportError` if operators in the TPOT configuration are not available when `verbosity>2`
- Thank @PGijsbers for the suggestions. Now TPOT can save scores of individuals already evaluated in any generation even the evaluation process of that generation is interrupted/stopped. But it is noted that, in this case, TPOT will raise this **warning message**: `WARNING: TPOT may not provide a good pipeline if TPOT is stopped/interrupted in a early generation.`, because the pipelines in early generation, e.g. 1st generation, are evolved/modified very limited times via evolutionary algorithm.
- Fix bugs in configuration of `TPOTRegressor`
- Error fixes in documentation

## Version 0.9.5

- **TPOT now supports integration with Dask for parallelization + smart caching**. Big thanks to the Dask dev team for making this happen!

- TPOT now supports for imputation/sparse matrices into `predict` and `predict_proba` functions.

- `TPOTClassifier` and `TPOTRegressor` now follows scikit-learn estimator API.

- We refined scoring parameter in TPOT API for accepting [`Scorer` object](http://jaquesgrobler.github.io/online-sklearn-build/modules/generated/sklearn.metrics.Scorer.html).

- We refined parameters in VarianceThreshold and FeatureAgglomeration.

- TPOT now supports using memory caching within a Pipeline via an optional `memory` parameter.

- We improved documentation of TPOT.

## Version 0.9

* **TPOT now supports sparse matrices** with a new built-in TPOT configuration, "TPOT sparse". We are using a custom OneHotEncoder implementation that supports missing values and continuous features.

* We have added an "early stopping" option for stopping the optimization process if no improvement is made within a set number of generations. Look up the `early_stop` parameter to access this functionality.

* TPOT now reduces the number of duplicated pipelines between generations, which saves you time during the optimization process.

* TPOT now supports custom scoring functions via the command-line mode.

* We have added a new optional argument, `periodic_checkpoint_folder`, that allows TPOT to periodically save the best pipeline so far to a local folder during optimization process.

* TPOT no longer uses `sklearn.externals.joblib` when `n_jobs=1` to avoid the potential freezing issue [that scikit-learn suffers from](http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux).

* We have added `pandas` as a dependency to read input datasets instead of `numpy.recfromcsv`. NumPy's `recfromcsv` function is unable to parse datasets with complex data types.

* Fixed a bug that `DEFAULT` in the parameter(s) of nested estimator raises `KeyError` when exporting pipelines.

* Fixed a bug related to setting `random_state` in nested estimators. The issue would happen with pipeline with `SelectFromModel` (`ExtraTreesClassifier` as nested estimator) or `StackingEstimator` if nested estimator has `random_state` parameter.

* Fixed a bug in the missing value imputation function in TPOT to impute along columns instead rows.

* Refined input checking for sparse matrices in TPOT.

* Refined the TPOT pipeline mutation operator.


## Version 0.8

* **TPOT now detects whether there are missing values in your dataset** and replaces them with the median value of the column.

* TPOT now allows you to set a `group` parameter in the `fit` function so you can use the [GroupKFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) cross-validation strategy.

* TPOT now allows you to set a subsample ratio of the training instance with the `subsample` parameter. For example, setting `subsample`=0.5 tells TPOT to create a fixed subsample of half of the training data for the pipeline optimization process. This parameter can be useful for speeding up the pipeline optimization process, but may give less accurate performance estimates from cross-validation.

* **TPOT now has more [built-in configurations](/using/#built-in-tpot-configurations)**, including TPOT MDR and TPOT light, for both classification and regression problems.

* `TPOTClassifier` and `TPOTRegressor` now expose three useful internal attributes, `fitted_pipeline_`, `pareto_front_fitted_pipelines_`, and `evaluated_individuals_`. These attributes are described in the [API documentation](/api/).

* Oh, **TPOT now has [thorough API documentation](/api/)**. Check it out!

* Fixed a reproducibility issue where setting `random_seed` didn't necessarily result in the same results every time. This bug was present since TPOT v0.7.

* Refined input checking in TPOT.

* Removed Python 2 uncompliant code.


## Version 0.7

* **TPOT now has multiprocessing support.** TPOT allows you to use multiple processes in parallel to accelerate the pipeline optimization process in TPOT with the `n_jobs` parameter.

* TPOT now allows you to **customize the operators and parameters considered during the optimization process**, which can be accomplished with the new `config_dict` parameter. The format of this customized dictionary can be found in the [online documentation](/using/#customizing-tpots-operators-and-parameters), along with a list of [built-in configurations](/using/#built-in-tpot-configurations).

* TPOT now allows you to **specify a time limit for evaluating a single pipeline**  (default limit is 5 minutes) in optimization process with the `max_eval_time_mins` parameter, so TPOT won't spend hours evaluating overly-complex pipelines.

* We tweaked TPOT's underlying evolutionary optimization algorithm to work even better, including using the [mu+lambda algorithm](http://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaMuPlusLambda). This algorithm gives you more control of how many pipelines are generated every iteration with the `offspring_size` parameter.

* Refined the default operators and parameters in TPOT, so TPOT 0.7 should work even better than 0.6.

* TPOT now supports sample weights in the fitness function if some if your samples are more important to classify correctly than others. The sample weights option works the same as in scikit-learn, e.g., `tpot.fit(x_train, y_train, sample_weights=sample_weights)`.

* The default scoring metric in TPOT has been changed from balanced accuracy to accuracy, the same default metric for classification algorithms in scikit-learn. Balanced accuracy can still be used by setting `scoring='balanced_accuracy'` when creating a TPOT instance.


## Version 0.6

* **TPOT now supports regression problems!** We have created two separate `TPOTClassifier` and `TPOTRegressor` classes to support classification and regression problems, respectively. The [command-line interface](/using/#tpot-on-the-command-line) also supports this feature through the `-mode` parameter.

* TPOT now allows you to **specify a time limit** for the optimization process with the `max_time_mins` parameter, so you don't need to guess how long TPOT will take any more to recommend a pipeline to you.

* Added a new operator that performs feature selection using [ExtraTrees](http://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees) feature importance scores.

* **[XGBoost](https://github.com/dmlc/xgboost) has been added as an optional dependency to TPOT.** If you have XGBoost installed, TPOT will automatically detect your installation and use the `XGBoostClassifier` and `XGBoostRegressor` in its pipelines.

* TPOT now offers a verbosity level of 3 ("science mode"), which outputs the entire Pareto front instead of only the current best score. This feature may be useful for users looking to make a trade-off between pipeline complexity and score.

## Version 0.5

* Major refactor: Each operator is defined in a separate class file. Hooray for easier-to-maintain code!
* TPOT now **exports directly to scikit-learn Pipelines** instead of hacky code.
* Internal representation of individuals now uses scikit-learn pipelines.
* Parameters for each operator have been optimized so TPOT spends less time exploring useless parameters.
* We have removed pandas as a dependency and instead use numpy matrices to store the data.
* TPOT now uses **k-fold cross-validation** when evaluating pipelines, with a default k = 3. This k parameter can be tuned when creating a new TPOT instance.
* Improved **scoring function support**: Even though TPOT uses balanced accuracy by default, you can now have TPOT use [any of the scoring functions](http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values) that `cross_val_score` supports.
* Added the scikit-learn [Normalizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) preprocessor.
* [Minor text fixes.](http://knowyourmeme.com/memes/pokemon-go-updates-controversy)

## Version 0.4

In TPOT 0.4, we've made some major changes to the internals of TPOT and added some convenience functions. We've summarized the changes below.

<ul>
<li>Added new sklearn models and preprocessors

<ul>
<li>AdaBoostClassifier</li>
<li>BernoulliNB</li>
<li>ExtraTreesClassifier</li>
<li>GaussianNB</li>
<li>MultinomialNB</li>
<li>LinearSVC</li>
<li>PassiveAggressiveClassifier</li>
<li>GradientBoostingClassifier</li>
<li>RBFSampler</li>
<li>FastICA</li>
<li>FeatureAgglomeration</li>
<li>Nystroem</li>
</ul></li>
<li>Added operator that inserts virtual features for the count of features with values of zero</li>
<li>Reworked parameterization of TPOT operators
<ul>
<li>Reduced parameter search space with information from a scikit-learn benchmark</li>
<li>TPOT no longer generates arbitrary parameter values, but uses a fixed parameter set instead</li>
</ul></li>
<li>Removed XGBoost as a dependency
<ul>
<li>Too many users were having install issues with XGBoost</li>
<li>Replaced with scikit-learn's GradientBoostingClassifier</li>
</ul></li>
<li>Improved descriptiveness of TPOT command line parameter documentation</li>
<li>Removed min/max/avg details during fit() when verbosity &gt; 1

<ul>
<li>Replaced with tqdm progress bar</li>
<li>Added tqdm as a dependency</li>
</ul></li>
<li>Added <code>fit_predict()</code> convenience function</li>
<li>Added <code>get_params()</code> function so TPOT can operate in scikit-learn's <code>cross_val_score</code> & related functions</li>
</ul>

## Version 0.3

* We revised the internal optimization process of TPOT to make it more efficient, in particular in regards to the model parameters that TPOT optimizes over.

## Version 0.2

* TPOT now has the ability to export the optimized pipelines to sklearn code.

* Logistic regression, SVM, and k-nearest neighbors classifiers were added as pipeline operators. Previously, TPOT only included decision tree and random forest classifiers.

* TPOT can now use arbitrary scoring functions for the optimization process.

* TPOT now performs multi-objective Pareto optimization to balance model complexity (i.e., # of pipeline operators) and the score of the pipeline.

## Version 0.1

* First public release of TPOT.

* Optimizes pipelines with decision trees and random forest classifiers as the model, and uses a handful of feature preprocessors.
