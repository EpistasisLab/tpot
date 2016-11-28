from sklearn.datasets import make_classification
from tpot import TPOTClassifier

X, y = make_classification(n_samples=200, n_features=80,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)

tpot = TPOTClassifier(generations=5, crossover_rate= 0.5, population_size=20, verbosity = 2)
tpot.fit(X, y)
