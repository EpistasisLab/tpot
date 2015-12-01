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
tpot.score(X_train, y_train, X_test, y_test)
```

Running this code should discover a pipeline that achieves ~98% testing accuracy.
