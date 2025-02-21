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
from ..graphsklearn import GraphPipeline
from sklearn.pipeline import Pipeline
import sklearn

def number_of_nodes_objective(est):
    """
    Calculates the number of leaves (input nodes) in an sklearn pipeline

    Parameters
    ----------
    est: GraphPipeline | Pipeline | FeatureUnion | BaseEstimator
        The pipeline to compute the number of nodes from.
    """
        
    if isinstance(est, GraphPipeline):
        return sum(number_of_nodes_objective(est.graph.nodes[node]["instance"]) for node in est.graph.nodes)
    if isinstance(est, Pipeline):
        return sum(number_of_nodes_objective(estimator) for _,estimator in est.steps)
    if isinstance(est, sklearn.pipeline.FeatureUnion):
        return sum(number_of_nodes_objective(estimator) for _,estimator in est.transformer_list)
    
    return 1