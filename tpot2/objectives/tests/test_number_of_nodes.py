import pytest
import tpot2
from sklearn.datasets import load_iris
import random
import sklearn

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import networkx as nx
import tpot2
from tpot2 import GraphPipeline
import sklearn.metrics

def test_number_of_nodes_objective_Graphpipeline():
    g = nx.DiGraph()

    g.add_node("scaler", instance=StandardScaler())
    g.add_node("svc", instance=SVC())
    g.add_node("LogisticRegression", instance=LogisticRegression())
    g.add_node("LogisticRegression2", instance=LogisticRegression())

    g.add_edge("svc","scaler")
    g.add_edge("LogisticRegression", "scaler")
    g.add_edge("LogisticRegression2", "LogisticRegression")
    g.add_edge("LogisticRegression2", "svc")

    est = GraphPipeline(g)

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(est) == 4

def test_number_of_nodes_objective_Pipeline():
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(pipe) == 2

def test_number_of_nodes_objective_not_pipeline_or_graphpipeline():
    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(SVC()) == 1
    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(StandardScaler()) == 1
    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(LogisticRegression()) == 1

def test_number_of_nodes_objective_pipeline_in_graphpipeline():
    g = nx.DiGraph()

    g.add_node("scaler", instance=StandardScaler())
    g.add_node("pipe", instance=Pipeline([("scaler", StandardScaler()), ("svc", SVC())]))

    g.add_edge("pipe","scaler")

    est = GraphPipeline(g)

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(est) == 3

def test_number_of_nodes_objective_graphpipeline_in_pipeline():
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

    g = nx.DiGraph()

    g.add_node("scaler", instance=StandardScaler())
    g.add_node("svc", instance=SVC())
    g.add_node("LogisticRegression", instance=LogisticRegression())
    g.add_node("LogisticRegression2", instance=LogisticRegression())

    g.add_edge("svc","scaler")
    g.add_edge("LogisticRegression", "scaler")
    g.add_edge("LogisticRegression2", "LogisticRegression")
    g.add_edge("LogisticRegression2", "svc")

    est = GraphPipeline(g)

    pipe.steps.append(("graphpipe", est))

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(pipe) == 6


def test_number_of_nodes_objective_graphpipeline_in_graphpipeline():
    g = nx.DiGraph()

    g.add_node("scaler", instance=StandardScaler())
    g.add_node("svc", instance=SVC())
    g.add_node("LogisticRegression", instance=LogisticRegression())
    g.add_node("LogisticRegression2", instance=LogisticRegression())

    g.add_edge("svc","scaler")
    g.add_edge("LogisticRegression", "scaler")
    g.add_edge("LogisticRegression2", "LogisticRegression")
    g.add_edge("LogisticRegression2", "svc")

    est = GraphPipeline(g)

    g2 = nx.DiGraph()

    g2.add_node("g1", instance=est)
    g2.add_node("svc", instance=SVC())
    g2.add_node("LogisticRegression", instance=LogisticRegression())
    g2.add_node("LogisticRegression2", instance=LogisticRegression())

    g2.add_edge("svc","g1")
    g2.add_edge("LogisticRegression", "g1")
    g2.add_edge("LogisticRegression2", "LogisticRegression")
    g2.add_edge("LogisticRegression2", "svc")

    est2 = GraphPipeline(g2)

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(est2) == 7

def test_number_of_nodes_objective_pipeline_in_pipeline():
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

    pipe2 = Pipeline([("pipe", pipe), ("svc", SVC())])

    assert tpot2.objectives.number_of_nodes.number_of_nodes_objective(pipe2) == 3
