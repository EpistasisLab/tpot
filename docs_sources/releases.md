# Version 0.7

* **TPOT now has multiprocessing support (Linux and macOS only).** TPOT allows you to use multiple processes for accelerating pipeline optimization in TPOT with the `n_jobs` parameter in both TPOTClassifier and TPOTRegressor.

* TPOT now allows you to **customize the operators and parameters explored during the optimization process.** TPOT allows you to customize the list of operators and parameters in optimization process of TPOT with the `config_dict` parameter. The format of this customized dictionary can be found in the [online documentation](/using/#tpot-with-code).

* TPOT now allows you to **specify a time limit for evaluating a single pipeline**  (default limit is 5 minutes) in optimization process with the `max_eval_time_mins` parameter, so TPOT won't spend hours evaluating overly-complex pipelines.

* We tweaked TPOT's underlying evolutionary optimization algorithm to work even better, including using the [mu+lambda algorithm](http://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaMuPlusLambda). This algorithm gives you more control of how many pipelines are generated every iteration with the `offspring_size` parameter.

* Fixed a reproducibility issue where setting `random_seed` didn't necessarily result in the same results every time. This bug was present since version 0.6.

* Refined the default operators and parameters in TPOT, so TPOT 0.7 should work even better than 0.6.

* TPOT now supports sample weights in the fitness function if some if your samples are more important to classify correctly than others. The sample weights option works the same as in scikit-learn, e.g., `tpot.fit(x_train, y_train, sample_weights=sample_weights)`.

* The default scoring metric in TPOT has been changed from balanced accuracy to accuracy, the same default metric for classification algorithms in scikit-learn. Balanced accuracy can still be used by setting `scoring='balanced_accuracy'` when creating a TPOT instance.


# Version 0.6

* **TPOT now supports regression problems!** We have created two separate `TPOTClassifier` and `TPOTRegressor` classes to support classification and regression problems, respectively. The [command-line interface](/using/#tpot-on-the-command-line) also supports this feature through the `-mode` parameter.

* TPOT now allows you to **specify a time limit** for the optimization process with the `max_time_mins` parameter, so you don't need to guess how long TPOT will take any more to recommend a pipeline to you.

* Added a new operator that performs feature selection using [ExtraTrees](http://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees) feature importance scores.

* **[XGBoost](https://github.com/dmlc/xgboost) has been added as an optional dependency to TPOT.** If you have XGBoost installed, TPOT will automatically detect your installation and use the `XGBoostClassifier` and `XGBoostRegressor` in its pipelines.

* TPOT now offers a verbosity level of 3 ("science mode"), which outputs the entire Pareto front instead of only the current best score. This feature may be useful for users looking to make a trade-off between pipeline complexity and score.

# Version 0.5

* Major refactor: Each operator is defined in a separate class file. Hooray for easier-to-maintain code!
* TPOT now **exports directly to scikit-learn Pipelines** instead of hacky code.
* Internal representation of individuals now uses scikit-learn pipelines.
* Parameters for each operator have been optimized so TPOT spends less time exploring useless parameters.
* We have removed pandas as a dependency and instead use numpy matrices to store the data.
* TPOT now uses **k-fold cross-validation** when evaluating pipelines, with a default k = 3. This k parameter can be tuned when creating a new TPOT instance.
* Improved **scoring function support**: Even though TPOT uses balanced accuracy by default, you can now have TPOT use [any of the scoring functions](http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values) that `cross_val_score` supports.
* Added the scikit-learn [Normalizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) preprocessor.
* [Minor text fixes.](http://knowyourmeme.com/memes/pokemon-go-updates-controversy)

# Version 0.4

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

# Version 0.3

* We revised the internal optimization process of TPOT to make it more efficient, in particular in regards to the model parameters that TPOT optimizes over.

# Version 0.2

* TPOT now has the ability to export the optimized pipelines to sklearn code.

* Logistic regression, SVM, and k-nearest neighbors classifiers were added as pipeline operators. Previously, TPOT only included decision tree and random forest classifiers.

* TPOT can now use arbitrary scoring functions for the optimization process.

* TPOT now performs multi-objective Pareto optimization to balance model complexity (i.e., # of pipeline operators) and the score of the pipeline.

# Version 0.1

* First public release of TPOT.

* Optimizes pipelines with decision trees and random forest classifiers as the model, and uses a handful of feature preprocessors.
