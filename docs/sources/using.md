# TPOT on the command line

To use TPOT via the command line, enter the following command to see the parameters that TPOT can receive:

```Shell
tpot --help
```

The following parameters will display along with their descriptions:

* `-i` / `INPUT_FILE`: The path to the data file to optimize the pipeline on. Make sure that the class column in the file is labeled as "class".
* `-is` / `INPUT_SEPARATOR`: The character used to separate columns in the input file. Commas (,) and tabs (\t) are the most common separators.
* `-o` / `OUTPUT_FILE`: The path to a file that you wish to export the pipeline code into. By default, exporting is disabled.
* `-g` / `GENERATIONS`: The number of generations to run pipeline optimization for. Must be > 0. The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
* `-p` / `POPULATION`: The number of pipelines in the genetic algorithm population. Must be > 0. The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
* `-mr` / `MUTATION_RATE`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-xr` / `CROSSOVER_RATE`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-s` / `RANDOM_STATE`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `-v` / `VERBOSITY`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all

An example command-line call to TPOT may look like:

```Shell
tpot -i data/mnist.csv -is , -o tpot_exported_pipeline.py -g 100 -s 42 -v 2
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

* `generations`: The number of generations to run pipeline optimization for. Must be > 0. The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
* `population_size`: The number of pipelines in the genetic algorithm population. Must be > 0. The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
* `mutation_rate`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `crossover_rate`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `random_state`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `verbosity`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all

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
print(pipeline_optimizer.score(training_features, training_classes,
                               testing_features, testing_classes))
```

Note that you currently need to pass the training data to the `score()` function so TPOT re-trains the optimized pipeline on the training data.

You also have the option to pass a user-defined scoring function to `score()`. For more information on this functionality, check [here](examples/Custom_Scoring_Functions.md). 

Finally, you can tell TPOT to export the corresponding Python code for the optimized pipeline to a text file with the `export()` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(training_features, training_classes, testing_features, testing_classes))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

Check our [examples](examples/MNIST_Example/) to see TPOT applied to some specific data sets.
