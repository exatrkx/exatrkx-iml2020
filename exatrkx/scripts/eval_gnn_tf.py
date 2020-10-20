#!/usr/bin/env python
import os

import tensorflow as tf
import sonnet as snt

import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from graph_nets import utils_tf

from exatrkx import graph
from exatrkx import SegmentClassifier
from exatrkx import plot_metrics
from exatrkx import plot_nx_with_edge_cmaps

ckpt_name = 'checkpoint'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained GNN model")
    add_arg = parser.add_argument
    add_arg("input_data", help='input tfrec data')
    add_arg("filter_dir", help='filtering directory')
    add_arg("model_dir", help='model directory')
    add_arg("outdir", help='output directory')
    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg('--inspect', help='inspect intermediate results', action='store_true')

    args = parser.parse_args()

    filenames = tf.io.gfile.glob(args.input_data)
    nevts = 1
    print("Input file names:", filenames)
    print("In total", len(filenames), "files")
    print("Process", nevts, "events")
    if not "gs://" in args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    # load data
    raw_dataset = tf.data.TFRecordDataset(filenames)
    dataset = raw_dataset.map(graph.parse_tfrec_function)
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = dataset.prefetch(AUTO)

    with_batch_dim = False
    inputs, targets = next(dataset.take(1).as_numpy_iterator())
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim)
    )

    num_processing_steps_tr = args.num_iters
    optimizer = snt.optimizers.Adam(0.001)
    model = SegmentClassifier()

    output_dir = args.model_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=5)
    if os.path.exists(os.path.join(output_dir, ckpt_name)):
        print("Find model:", output_dir)
        status = checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Loaded latest checkpoint from:", output_dir)
    else:
        raise ValueError("cannot find model at:", output_dir)

    evtids = [os.path.basename(args.input_data)]

    outputs_te_list = []
    targets_te_list = []
    ievt = 0
    for inputs in dataset.take(nevts).as_numpy_iterator():
        inputs_te, targets_te = inputs
        outputs_te = model(inputs_te, num_processing_steps_tr)
        
        output_graph = outputs_te[-1]
        target_graph = targets_te

        outputs_te_list.append(output_graph)
        targets_te_list.append(target_graph)

        evtid = evtids[ievt]
        print("processing event {}".format(evtid))

        filter_file = os.path.join(args.filter_dir, "{}".format(evtid))
        array = torch.load(filter_file, map_location='cpu')
        hits_id_nsecs = array['hid'].numpy()
        hits_pid_nsecs = array['pid'].numpy()

        output = os.path.join(args.outdir, "event{}.npz".format(evtid))
        array = {
            "receivers": inputs_te.receivers,
            "senders": inputs_te.senders,
            "score": tf.reshape(outputs_te[-1].edges, (-1, )),
            "truth": tf.reshape(targets_te.edges, (-1, )),
            "I": hits_id_nsecs,
            "pid": hits_pid_nsecs,
            "x": inputs_te.nodes, 
        }
        np.savez(output, **array)

        if args.inspect:

            y_test = array['truth']
            for i in range(num_processing_steps_tr):
                array['score'] = tf.reshape(outputs_te[i].edges, (-1, )).numpy()
                score =  array['score']
                plot_metrics(
                    score, y_test,
                    outname=os.path.join(args.outdir, "event{}_roc_{}.pdf".format(evtid, i)),
                    off_interactive=True
                )
                nx_filename = os.path.join(args.outdir, "event{}_nx_{}.pkl".format(evtid, i))
                if os.path.exists(nx_filename):
                    G = nx.read_gpickle(nx_filename)
                else:
                    G = np_to_nx(array)
                    nx.write_gpickle(G, nx_filename)
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G, weight_name='weight', threshold=0.01, ax=ax)
                plt.savefig(os.path.join(args.outdir, "event{}_display_all_{}.pdf".format(evtid, i)))
                plt.clf()

                # do truth
                G1 = nx.Graph()
                G1.add_nodes_from(G.nodes(data=True))
                G1.add_edges_from([edge for edge in G.edges(data=True) if edge[2]['solution'] == 1])
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G1, weight_name='weight', threshold=0.01, ax=ax, cmaps=plt.get_cmap("gray"))
                plt.savefig(os.path.join(args.outdir, "event{}_display_truth_{}.pdf".format(evtid, i)))
                plt.clf()

                # do fake 
                G2 = nx.Graph()
                G2.add_nodes_from(G.nodes(data=True))
                G2.add_edges_from([edge for edge in G.edges(data=True) if edge[2]['solution'] == 0])
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G2, weight_name='weight', threshold=0.01, ax=ax, cmaps=plt.get_cmap("Greys"))
                plt.savefig(os.path.join(args.outdir, "event{}_display_fake_{}.pdf".format(evtid, i)))
                plt.clf()

    outputs_te = utils_tf.concat(outputs_te_list, axis=0)
    targets_te = utils_tf.concat(targets_te_list, axis=0)
    prediction = tf.reshape(outputs_te.edges, (-1,))
    y_test = tf.reshape(targets_te.edges, (-1, ))
    plot_metrics(
        prediction, y_test,
        outname=os.path.join(args.outdir, "roc.pdf"),
        off_interactive=True
        )