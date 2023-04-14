# Test all nodes have all dictionaries
import pytest
import tpot2.graphsklearn as GraphSklearn


@pytest.mark.skip(reason="Not yet implemented")
def test_BaseNode_static_methods():
    n1 = GraphSklearn.BaseNode()
    n2 = GraphSklearn.BaseNode()
    n3 = GraphSklearn.BaseNode()
    n4 = GraphSklearn.BaseNode()
    n5 = GraphSklearn.BaseNode()
    n6 = GraphSklearn.BaseNode()
    n7 = GraphSklearn.BaseNode()

    GraphSklearn.BaseNode._connect_nodes(child=n2,parent=n1)
    GraphSklearn.BaseNode._connect_nodes(child=n3,parent=n1)
    GraphSklearn.BaseNode._connect_nodes(child=n4,parent=n1)

    GraphSklearn.BaseNode._connect_nodes(child=n5,parent=n2)

    GraphSklearn.BaseNode._connect_nodes(child=n5,parent=n3)
    GraphSklearn.BaseNode._connect_nodes(child=n6,parent=n3)

    GraphSklearn.BaseNode._connect_nodes(child=n6,parent=n4)
    GraphSklearn.BaseNode._connect_nodes(child=n3,parent=n4)

    GraphSklearn.BaseNode._connect_nodes(child=n4,parent=n7) #This is a second root, allowed by basenode

    assert n2 in n1.children
    assert n3 in n1.children
    assert n4 in n1.children
    assert n5 in n2.children
    assert n5 in n3.children
    assert n6 in n3.children
    assert n6 in n4.children
    assert n3 in n4.children
    assert n4 in n7.children

    assert n1 in n2.parents
    assert n1 in n3.parents
    assert n1 in n4.parents
    assert n2 in n5.parents
    assert n3 in n5.parents
    assert n3 in n6.parents
    assert n4 in n6.parents
    assert n4 in n3.parents
    assert n7 in n4.parents

    assert n2 not in n1.parents
    assert n3 not in n1.parents
    assert n4 not in n1.parents
    assert n6 not in n2.parents
    assert n5 not in n3.parents
    assert n6 not in n3.parents
    assert n6 not in n4.parents

    #Make sure every node knows about every other node in the graph.
    for this_node in [n1,n2,n3,n4,n5,n6, n7]:
        for other_node in [n1,n2,n3,n4,n5,n6, n7]:
            assert this_node in other_node.node_set


    GraphSklearn.BaseNode._disconnect_nodes(child=n3, parent=n4)
    assert n4 not in n3.parents
    assert n3 not in n4.children

    n8 = GraphSklearn.BaseNode()
    GraphSklearn.BaseNode._insert_inner_node(child=n3,new_node=n8,parent=n4)
    assert n8 in n3.parents
    assert n4 in n8.parents

    assert n3 in n8.children
    assert n8 in n4.children

    n9 = GraphSklearn.BaseNode()

    GraphSklearn.BaseNode._remove_node(node=n8, replacement=n9)
    assert n8 not in n3.parents
    assert n4 not in n8.parents

    assert n3 not in n8.children
    assert n8 not in n4.children
    assert n8 not in n1.node_set

    assert n9 in n3.parents
    assert n4 in n9.parents

    assert n3 in n9.children
    assert n9 in n4.children

    for this_node in [n1,n2,n3,n4,n5,n6, n7, n9]:
        for other_node in [n1,n2,n3,n4,n5,n6, n7, n9]:
            assert this_node in other_node.node_set


@pytest.mark.skip(reason="Not yet implemented")
def test_BaseNode_static_crossover_methods():
    n1 = GraphSklearn.BaseNode()
    n2 = GraphSklearn.BaseNode()
    n3 = GraphSklearn.BaseNode()
    n4 = GraphSklearn.BaseNode()
    n5 = GraphSklearn.BaseNode()
    n6 = GraphSklearn.BaseNode()
    n7 = GraphSklearn.BaseNode()
    n8 = GraphSklearn.BaseNode()

    GraphSklearn.BaseNode._connect_nodes(child=n2,parent=n1)
    GraphSklearn.BaseNode._connect_nodes(child=n3,parent=n1)
    GraphSklearn.BaseNode._connect_nodes(child=n4,parent=n1)

    GraphSklearn.BaseNode._connect_nodes(child=n5,parent=n2)

    GraphSklearn.BaseNode._connect_nodes(child=n5,parent=n3)
    GraphSklearn.BaseNode._connect_nodes(child=n6,parent=n3)

    GraphSklearn.BaseNode._connect_nodes(child=n6,parent=n4)
    GraphSklearn.BaseNode._connect_nodes(child=n3,parent=n4)

    GraphSklearn.BaseNode._connect_nodes(child=n4,parent=n7) #This is a second root, allowed by basenode

    GraphSklearn.BaseNode._connect_nodes(child=n8,parent=n5)
    GraphSklearn.BaseNode._connect_nodes(child=n8,parent=n1)

    depth_dict = n1._get_depth_dictionary()

    assert depth_dict[n1] == 0
    assert depth_dict[n2] == 1
    assert depth_dict[n3] == 2
    assert depth_dict[n4] == 1
    assert depth_dict[n5] == 3
    assert depth_dict[n6] == 3
    assert depth_dict[n8] == 4
    assert n7 not in depth_dict