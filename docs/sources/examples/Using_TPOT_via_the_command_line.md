# Using TPOT via the command line

To use TPOT via the command line, enter the following command to see the parameters that TPOT can receive:

```Shell
tpot --help
```

The following parameters will display along with their descriptions:

* `-i` / `INPUT_FILE`: The path to the data file to optimize the pipeline on. Make sure that the class column in the file is labeled as "class".
* `-is` / `INPUT_SEPARATOR`: The character used to separate columns in the input file. Commas (,) and tabs (\t) are the most common separators.
* `-g` / `GENERATIONS`: The number of generations to run pipeline optimization for. Must be > 0. The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
* `-p` / `POPULATION`: The number of pipelines in the genetic algorithm population. Must be > 0. The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
* `-mr` / `MUTATION_RATE`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-xr` / `CROSSOVER_RATE`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-s` / `RANDOM_STATE`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `-v` / `VERBOSITY`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all

An example command-line call to TPOT may look like:

```Shell
tpot -i data/mnist.csv -is , -g 100 -s 42 -v 2
```
