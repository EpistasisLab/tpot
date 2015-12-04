# MNIST Example

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train, X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves ~97% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file.
