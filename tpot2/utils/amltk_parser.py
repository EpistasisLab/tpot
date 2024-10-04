from amltk.pipeline import Choice, Component, Sequential, Node, Fixed, Split, Join, Searchable
from tpot2.search_spaces.pipelines import SequentialPipeline, ChoicePipeline, UnionPipeline
from tpot2.search_spaces.nodes import EstimatorNode
from ConfigSpace import ConfigurationSpace

def component_to_estimatornode(component: Component) -> EstimatorNode:
    method = component.item
    space_dict = {}
    if component.space is not None:
        space_dict.update(component.space)
    if component.config is not None:
        space_dict.update(component.config)
    space = ConfigurationSpace(component.space)
    
    tpot2_sp = EstimatorNode(method=method, space=space)
    return tpot2_sp
    
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

    tpot2_sp = EstimatorNode(method=method, space=space_dict)
    return tpot2_sp

def sequential_to_sequentialpipeline(sequential: Sequential) -> SequentialPipeline:
    nodes = [tpot2_parser(node) for node in sequential.nodes]
    tpot2_sp = SequentialPipeline(search_spaces=nodes)
    return tpot2_sp

def choice_to_choicepipeline(choice: Choice) -> ChoicePipeline:
    nodes = [tpot2_parser(node) for node in choice.nodes]
    tpot2_sp = ChoicePipeline(search_spaces=nodes)
    return tpot2_sp


def split_to_unionpipeline(split: Split) -> UnionPipeline:
    nodes = [tpot2_parser(node) for node in split.nodes]
    tpot2_sp = UnionPipeline(search_spaces=nodes)
    return tpot2_sp

def tpot2_parser(
    node: Node,
    # *,
    # flat: bool = False,
    # conditionals: bool = False,
    # delim: str = ":",
    ):

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
