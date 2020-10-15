import os

import numpy as np
import pandas as pd
import itertools

from graph_nets import utils_tf
from tfgraphs import dataset_base


def make_graph(event, debug=False, data_dict=False):
    n_nodes = event['x'].shape[0]
    n_edges = event['edges'].shape[0]
    nodes = event['x']
    edges = np.zeros((n_edges, 1), dtype=np.float32)
    senders =  event['edges'][:, 0]
    receivers = event['edges'][:, 1]
    edge_target = event['edge_target']
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([n_nodes], dtype=np.float32)
    }
    n_edges_target = 1
    target_datadict = {
        "n_node": 1,
        "n_edge": n_edges_target,
        "nodes": np.zeros((1, 1), dtype=np.float32),
        "edges": edge_target,
        "senders": np.zeros((n_edges_target,), dtype=np.int32),
        "receivers": np.zeros((n_edges_target,), dtype=np.int32),
        "globals": np.zeros((1,), dtype=np.float32),
    }
    if data_dict:
        return [(input_datadict, target_datadict)]
    else:
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        return [(input_graph, target_graph)]

def read(filedir):
    files = os.listdir(filedir)
    for filename in files:
        array = np.load(os.path.join(filedir, filename))
        yield array


class DoubletsDataset(dataset_base.DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph