import numpy as np

class TupleIndex():
    """
    TPOT2 uses tuples to create a unique id for some pipeline search spaces. However, tuples sometimes don't interact correctly with pandas indexes.
    This class is a wrapper around a tuple that allows it to be used as a key in a dictionary, without it being an itereable.

    An alternative could be to make unique id return a string, but this would not work with graphpipelines, which require a special object.
    This class allows linear pipelines to contain graph pipelines while still being able to be used as a key in a dictionary.
    
    """
    def __init__(self, tup):
        self.tup = tup

    def __eq__(self,other) -> bool:
        return self.tup == other
    
    def __hash__(self) -> int:
        return self.tup.__hash__()

    def __str__(self) -> str:
        return self.tup.__str__()
    
    def __repr__(self) -> str:
        return self.tup.__repr__()