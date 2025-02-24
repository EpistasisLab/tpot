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
import networkx as nx
import numpy as np


def remove_and_stitch(graph, node):
    successors = graph.successors(node)
    predecessors = graph.predecessors(node)

    graph.remove_node(node)

    for s in successors:
        for p in predecessors:
            graph.add_edge(p,s)


def remove_nodes_disconnected_from_node(graph, node):
    descendants = nx.descendants(graph, node)
    for n in list(graph.nodes):
        if n not in descendants and n is not node:
            graph.remove_node(n)
    #graph.remove_nodes_from([n for n in graph.nodes if n not in nx.descendants(graph, node) and n is not node])


def get_roots(graph):
    return [v for v, d in graph.in_degree() if d == 0]

def get_leaves(graph):
    return [v for v, d in graph.out_degree() if d == 0]

def get_max_path_through_node(graph, root, node):
    if len(list(graph.successors(node)))==0:
        return get_max_path_size(graph, root, node)
    else:
        leaves = [n for n in nx.descendants(graph,node) if len(list(graph.successors(n)))==0]

        return max([get_max_path_size(graph, root, l) for l in leaves])


def get_max_path_size(graph, fromnode1,tonode2, return_path=False):
    if fromnode1 is tonode2:
        if return_path:
            return [fromnode1]
        return 1
    else:
        max_length_path = max(nx.all_simple_paths(graph, fromnode1, tonode2), key=lambda x: len(x))
        if return_path:
            return max_length_path
        return len(max_length_path) #gets the max path and finds the length of that path


def invert_dictionary(d):
    inv_map = {}
    for k, v in d.items():
        inv_map.setdefault(v, set()).add(k)

    return inv_map

def select_nodes_same_depth(g1, node1, g2, node2, rng=None):
    rng = np.random.default_rng(rng)

    g1_nodes = nx.shortest_path_length(g1, source=node1)
    g2_nodes = nx.shortest_path_length(g2, source=node2)

    max_depth = max(list(g1_nodes.values()) + list(g2_nodes.values()))

    g1_nodes = invert_dictionary(g1_nodes)
    g2_nodes = invert_dictionary(g2_nodes)

    # depth_number_of_nodes = []
    # for i in range(max_depth+1):
    #     n = 0
    #     if i in g1_nodes and i in g2_nodes:
    #         depth_number_of_nodes.append(len(g1_nodes[i])+len(g1_nodes[i]))
    #     else:
    #         break

    possible_pairs = []
    for i in range(max_depth+1):
        if i in g1_nodes and i in g2_nodes:
            for n1 in g1_nodes[i]:
                for n2 in g2_nodes[i]:
                    possible_pairs.append( (n1,n2) )

    rng.shuffle(possible_pairs)

    for p in possible_pairs:
        yield p[0], p[1]

def select_nodes_randomly(g1, g2, rng=None):
    rng = np.random.default_rng(rng)

    sorted_self_nodes_list = list(g1.nodes)
    rng.shuffle(sorted_self_nodes_list)

    sorted_other_nodes_list = list(g2.nodes)
    rng.shuffle(sorted_other_nodes_list)
    for node1 in sorted_self_nodes_list:
        for node2 in sorted_other_nodes_list:
            if node1 is node2:
                continue
            yield node1, node2