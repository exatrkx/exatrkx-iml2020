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
from exatrkx import np_to_nx
from exatrkx import utils_dir

ckpt_name = 'checkpoint'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained GNN model")
    add_arg = parser.add_argument
    add_arg("--input-dir", help='input directory')
    add_arg("--output-dir", help='output directory')
    add_arg("--model-dir", help='model directory')
    add_arg("--filter-dir", help='filtering file directory')
    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg('--inspect', help='inspect intermediate results', action='store_true')
    add_arg("--overwrite", help="overwrite the output", action='store_true')
    add_arg("--max-evts", help='process maximum number of events', type=int, default=1)
    add_arg("--datatype", help="", default="test", choices=utils_dir.datatypes)
    add_arg("--ckpt-idx", help='index of which to which checkpoint model restores', type=int, default=-1)

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    gnn_input = os.path.join(utils_dir.gnn_inputs, args.datatype) if args.input_dir is None else args.input_dir
    filenames = tf.io.gfile.glob(os.path.join(gnn_input, "*"))

    nevts = args.max_evts
    outdir = os.path.join(utils_dir.gnn_output, args.datatype) if args.output_dir is None else args.output_dir
    os.makedirs(outdir, exist_ok=True)
    print("Input file names:", filenames)
    print("In total", len(filenames), "files")
    print("Process", nevts, "events")

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

    output_dir = utils_dir.gnn_models if args.model_dir is None else args.model_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
    if os.path.exists(os.path.join(output_dir, ckpt_name)):
        print("Find model:", output_dir)
        status = checkpoint.restore(ckpt_manager.checkpoints[args.ckpt_idx])
        print("Loaded {} checkpoint from {}".format(args.ckpt_idx, output_dir))
    else:
        raise ValueError("cannot find model at:", output_dir)


    filter_dir = os.path.join(utils_dir.filtering_outdir, 'test') if args.filter_dir is None else args.filter_dir
    outputs_te_list = []
    targets_te_list = []
    ievt = 0
    for inputs in dataset.take(nevts).as_numpy_iterator():
        evtid = int(os.path.basename(filenames[ievt]))
        print("processing event {}".format(evtid))


        inputs_te, targets_te = inputs
        outputs_te = model(inputs_te, num_processing_steps_tr)
        
        output_graph = outputs_te[-1]
        target_graph = targets_te

        filter_file = os.path.join(filter_dir, "{}".format(evtid))
        array = torch.load(filter_file, map_location='cpu')
        hits_id_nsecs = array['hid'].numpy()
        hits_pid_nsecs = array['pid'].numpy()
        
        array = {
            "receivers": inputs_te.receivers,
            "senders": inputs_te.senders,
            "score": tf.reshape(outputs_te[-1].edges, (-1, )).numpy(),
            "truth": tf.reshape(targets_te.edges, (-1, )).numpy(),
            "I": hits_id_nsecs,
            "pid": hits_pid_nsecs,
            "x": inputs_te.nodes, 
        }

        outputs_te_list.append(array['score'])
        targets_te_list.append(array['truth'])


        output = os.path.join(outdir, "{}.npz".format(evtid))
        if not os.path.exists(output) or args.overwrite:
            np.savez(output, **array)

        print("{:,} nodes".format(array['x'].shape[0]))
        print("{:,} edges".format(array['senders'].shape[0]))
        ievt += 1

        if args.inspect:
            y_test = array['truth']
            threshold = 0.8
            for i in range(num_processing_steps_tr):
                print("running {} message passing".format(i))
                score = tf.reshape(outputs_te[i].edges, (-1, )).numpy()
                plot_metrics(
                    score, y_test,
                    outname=os.path.join(outdir, "event{}_roc_{}.pdf".format(evtid, i)),
                    off_interactive=True
                )
                nx_filename = os.path.join(outdir, "event{}_nx_{}.pkl".format(evtid, i))
                if os.path.exists(nx_filename):
                    G = nx.read_gpickle(nx_filename)
                else:
                    G = np_to_nx(array)
                    nx.write_gpickle(G, nx_filename)
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G, weight_name='weight', threshold=threshold, ax=ax)
                plt.savefig(os.path.join(outdir, "event{}_display_all_{}.pdf".format(evtid, i)))
                plt.clf()

                # do truth
                G1 = nx.Graph()
                G1.add_nodes_from(G.nodes(data=True))
                G1.add_edges_from([edge for edge in G.edges(data=True) if edge[2]['solution'] == 1])
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G1, weight_name='weight', threshold=threshold, ax=ax, cmaps=plt.get_cmap("gray"))
                plt.savefig(os.path.join(outdir, "event{}_display_truth_{}.pdf".format(evtid, i)))
                plt.clf()

                # do fake
                G2 = nx.Graph()
                G2.add_nodes_from(G.nodes(data=True))
                G2.add_edges_from([edge for edge in G.edges(data=True) if edge[2]['solution'] == 0])
                _, ax = plt.subplots(figsize=(8, 8))
                plot_nx_with_edge_cmaps(G2, weight_name='weight', threshold=threshold, ax=ax, cmaps=plt.get_cmap("Greys"))
                plt.savefig(os.path.join(outdir, "event{}_display_fake_{}.pdf".format(evtid, i)))
                plt.clf()

    outplot = os.path.join(outdir, "tot_roc.pdf")
    if os.path.exists(outplot) and not args.overwrite:
       exit(0)

    prediction = np.concatenate(outputs_te_list)
    y_test = np.concatenate(targets_te_list)
    plot_metrics(
        prediction, y_test,
        outname=outplot,
        off_interactive=True
    )
