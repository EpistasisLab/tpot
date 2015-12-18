from .RandomForest import RandomForest
from .KNNc import KNNc

operator_registry = {
    'RandomForest':RandomForest(),
    'KNNc':KNNc()
}