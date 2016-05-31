# TPOT on the command line

To use TPOT via the command line, enter the following command to see the parameters that TPOT can receive:

```Shell
tpot --help

usage: tpot [-h] [-is INPUT_SEPARATOR] [-o OUTPUT_FILE] [-g GENERATIONS]
            [-p POPULATION_SIZE] [-mr MUTATION_RATE] [-xr CROSSOVER_RATE]
            [-s RANDOM_STATE] [-v {0,1,2}] [--no-update-check] [--version]
            INPUT_FILE

A Python tool that automatically creates and optimizes machine learning
pipelines using genetic programming.

positional arguments:
  INPUT_FILE           Data file to optimize the pipeline on; ensure that the
                       class label column is labeled as "class".

optional arguments:
  -h, --help           Show this help message and exit.
  -is INPUT_SEPARATOR  Character used to separate columns in the input file.
  -o OUTPUT_FILE       File to export the final optimized pipeline.
  -g GENERATIONS       Number of generations to run pipeline optimization
                       over. Generally, TPOT will work better when you give it
                       more generations (and therefore time) to optimize over.
                       TPOT will evaluate GENERATIONS x POPULATION_SIZE number
                       of pipelines in total.
  -p POPULATION_SIZE   Number of individuals in the GP population. Generally,
                       TPOT will work better when you give it more individuals
                       (and therefore time) to optimize over. TPOT will
                       evaluate GENERATIONS x POPULATION_SIZE number of
                       pipelines in total.
  -mr MUTATION_RATE    GP mutation rate in the range [0.0, 1.0]. We recommend
                       using the default parameter unless you understand how
                       the mutation rate affects GP algorithms.
  -xr CROSSOVER_RATE   GP crossover rate in the range [0.0, 1.0]. We recommend
                       using the default parameter unless you understand how
                       the crossover rate affects GP algorithms.
  -s RANDOM_STATE      Random number generator seed for reproducibility. Set
                       this seed if you want your TPOT run to be reproducible
                       with the same seed and data set in the future.
  -v {0,1,2}           How much information TPOT communicates while it is
                       running: 0 = none, 1 = minimal, 2 = all.
  --no-update-check    Flag indicating whether the TPOT version checker should
                       be disabled.
  --version            Show TPOT's version number and exit.
```

An example command-line call to TPOT may look like:

```Shell
tpot data/mnist.csv -is , -o tpot_exported_pipeline.py -g 100 -s 42 -v 2
```

# TPOT with code

We've taken care to design the TPOT interface to be as similar as possible to scikit-learn.

TPOT can be imported just like any regular Python module. To import TPOT, type:

```Python
from tpot import TPOT
```

then create an instance of TPOT as follows:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT()
```

Note that you can pass several parameters to the TPOT instantiation call:

* `generations`: The number of generations to run pipeline optimization over. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize over. TPOT will evaluate `generations` x `population_size` number of pipelines in total.
* `population_size`: The number of individuals in the GP population. Generally, TPOT will work better when you give it more individuals (and therefore time) to optimize over. TPOT will evaluate `generations` x `population_size` number of pipelines in total.
* `mutation_rate`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `crossover_rate`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `random_state`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `verbosity`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all. A setting of 2 will add a progress bar to calls to fit()
* `scoring_function`: Function used to evaluate the goodness of a given pipeline for the classification problem. By default, balanced class accuracy is used. See [here](examples/Custom_Scoring_Functions.md) for more information on custom scoring functions.
* `disable_update_check`: Flag indicating whether the TPOT version checker should be disabled.

Some example code with custom TPOT parameters might look like:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
```

Now TPOT is ready to work! You can tell TPOT to optimize a pipeline based on a data set with the `fit` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
```
The `fit()` function takes in a training data set, then further divides it into a training and validation data set (so as to do cross-validation). It then initializes the Genetic Algoritm to find the best pipeline based on the validation set performance evaluated on the basis of a scoring function (generally the classification accuracy, but can be user defined as well like precision/recall/f1, etc).   

You can then proceed to evaluate the final pipeline on the test set with the `score()` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
```

Finally, you can tell TPOT to export the corresponding Python code for the optimized pipeline to a text file with the `export()` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(testing_features, testing_classes))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

Check our [examples](examples/MNIST_Example/) to see TPOT applied to some specific data sets.
