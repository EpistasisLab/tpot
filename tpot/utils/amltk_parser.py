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
from amltk.pipeline import Choice, Component, Sequential, Node, Fixed, Split, Join, Searchable
from tpot.search_spaces.pipelines import SequentialPipeline, ChoicePipeline, UnionPipeline
from tpot.search_spaces.nodes import EstimatorNode
from ConfigSpace import ConfigurationSpace

def component_to_estimatornode(component: Component) -> EstimatorNode:
    method = component.item
    space_dict = {}
    if component.space is not None:
        space_dict.update(component.space)
    if component.config is not None:
        space_dict.update(component.config)
    space = ConfigurationSpace(component.space)
    
    tpot_sp = EstimatorNode(method=method, space=space)
    return tpot_sp
    
def fixed_to_estimatornode(node: Fixed) -> EstimatorNode:
    method = node.item
    #check if method is a class or an object
    if not isinstance(method, type):
        method = type(method)
    
    #if baseestimator, get params
    if hasattr(node.item, 'get_params'):
        space_dict = node.item.get_params(deep=False)
    else:
        space_dict = {}
    if node.space is not None:
        space_dict.update(node.space)
    if node.config is not None:
        space_dict.update(node.config)

    tpot_sp = EstimatorNode(method=method, space=space_dict)
    return tpot_sp

def sequential_to_sequentialpipeline(sequential: Sequential) -> SequentialPipeline:
    nodes = [tpot_parser(node) for node in sequential.nodes]
    tpot_sp = SequentialPipeline(search_spaces=nodes)
    return tpot_sp

def choice_to_choicepipeline(choice: Choice) -> ChoicePipeline:
    nodes = [tpot_parser(node) for node in choice.nodes]
    tpot_sp = ChoicePipeline(search_spaces=nodes)
    return tpot_sp


def split_to_unionpipeline(split: Split) -> UnionPipeline:
    nodes = [tpot_parser(node) for node in split.nodes]
    tpot_sp = UnionPipeline(search_spaces=nodes)
    return tpot_sp

def tpot_parser(
    node: Node,
    ):
    """
    Convert amltk pipeline search space into a tpot pipeline search space.

    Parameters
    ----------
    node: amltk.pipeline.Node
        The node to convert.

    Returns
    -------
    tpot.search_spaces.base.SearchSpace
        The equivalent TPOT search space which can be optimized by TPOT.
    """

    if isinstance(node, Component):
        return component_to_estimatornode(node)
    elif isinstance(node, Sequential):
        return sequential_to_sequentialpipeline(node)
    elif isinstance(node, Choice):
        return choice_to_choicepipeline(node)
    elif isinstance(node, Fixed):
        return fixed_to_estimatornode(node)
    elif isinstance(node, Split):
        return split_to_unionpipeline(node)
    else:
        raise ValueError(f"Node type {type(node)} not supported")
