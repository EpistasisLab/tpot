"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np

class TupleIndex():
    """
    TPOT uses tuples to create a unique id for some pipeline search spaces. However, tuples sometimes don't interact correctly with pandas indexes.
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