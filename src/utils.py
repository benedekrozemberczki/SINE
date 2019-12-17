"""Data reading utils."""

import json
import pandas as pd
import networkx as nx
from texttable import Texttable

def read_graph(graph_path):
    """
    Method to read graph and create a target matrix adjacency matrix powers.
    :param args: Arguments object.
    :return graph: graph.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    return graph

def read_features(feature_path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return sample_features: Feature JSON for sampling.
    """
    features = json.load(open(feature_path))
    sample_features = {}
    index = 0
    for k, v in features.items():
        for val in v:
            sample_features[index] = (int(k), int(val))
            index = index + 1
    return sample_features

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())
