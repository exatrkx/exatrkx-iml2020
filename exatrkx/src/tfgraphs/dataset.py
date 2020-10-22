"""
base class defines the procedure with that the TFrecord data is produced.
"""
import time
import os
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from exatrkx.src.tfgraphs import graph

class DoubletsDataset(object):
    def __init__(self, num_workers=1, with_padding=False,
                n_graphs_per_evt=1, overwrite=False, edge_name='edge_index',
                truth_name='y'
        ):
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_padding = False
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.edge_name = edge_name
        self.truth_name = truth_name

    def make_graph(self, event, debug=False):
        """
        Convert the event into a graphs_tuple. 
        """
        edge_name = self.edge_name
        n_nodes = event['x'].shape[0]
        n_edges = event[edge_name].shape[1]
        nodes = event['x']
        edges = np.zeros((n_edges, 1), dtype=np.float32)
        senders =  event[edge_name][0, :]
        receivers = event[edge_name][1, :]
        edge_target = event[self.truth_name].numpy().astype(np.float32)
        
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
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        return [(input_graph, target_graph)]        

    def _get_signature(self, tensors):
        if self.input_dtype and self.target_dtype:
            return 

        ex_input, ex_target = tensors[0]
        self.input_dtype, self.input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        self.target_dtype, self.target_shape = graph.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)
    

    def process(self, indir, outdir):
        files = os.listdir(indir)
        ievt = 0
        now = time.time()
        for filename in files:
            infile = os.path.join(indir, filename)
            outname = os.path.join(outdir, filename)
            if os.path.exists(outname) and not self.overwrite:
                continue
            if "npz" in infile:
                array = np.load(infile)
            else:
                import torch
                array = torch.load(infile, map_location='cpu')
            # print(array)
            tensors = self.make_graph(array)
            def generator():
                for G in tensors:
                    yield (G[0], G[1])
            self._get_signature(tensors)
            dataset = tf.data.Dataset.from_generator(
                generator, 
                output_types=(self.input_dtype, self.target_dtype),
                output_shapes=(self.input_shape, self.target_shape),
                args=None
            )

            writer = tf.io.TFRecordWriter(outname)
            for data in dataset:
                example = graph.serialize_graph(*data)
                writer.write(example)
            writer.close()
            ievt += 1

        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            ievt, read_time/60.))