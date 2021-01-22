#!/usr/bin/env python
"""
Training GNN in HPC using Horovod
"""

import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
try:
    import horovod.tensorflow as hvd
    no_horovod = False
except ModuleNotFoundError:
    logging.warning("No horvod module, cannot perform distributed training")
    no_horovod = True


# tf.config.optimizer.set_jit(True)
# tf.debugging.set_log_device_placement(True)

import os
import sys
import argparse
import glob
import re
import time
import random
import functools
from types import SimpleNamespace

import numpy as np
import sklearn.metrics


from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from exatrkx import graph
from exatrkx import SegmentClassifier
from exatrkx import utils_dir

prog_name = os.path.basename(sys.argv[0])

def init_workers(distributed=False):
    if distributed and not no_horovod:
        hvd.init()
        assert hvd.mpi_threads_supported()
        from mpi4py import MPI
        assert hvd.size() == MPI.COMM_WORLD.Get_size()
        comm = MPI.COMM_WORLD
        print("Rank: {}, Size: {}".format(hvd.rank(), hvd.size()))
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                                local_rank=hvd.local_rank(),
                                local_size=hvd.local_size(), comm=comm)
    else:
        print("not doing distributed")
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)

def train_and_evaluate(args):
    dist = init_workers(args.distributed)

    device = 'CPU'
    global_batch_size = 1
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info("found {} GPUs".format(len(gpus)))

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(gpus) > 0:
        device = "{}GPUs".format(len(gpus))
    if gpus and args.distributed:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    output_dir = utils_dir.gnn_models if args.output_dir is None else args.output_dir
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    num_processing_steps_tr = args.num_iters     ## level of message-passing
    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, global_batch_size))
    logging.info("{} processing steps in the model".format(num_processing_steps_tr))
    logging.info("I am in hvd rank: {} of  total {} ranks".format(dist.rank, dist.size))

    if dist.rank == 0:
        train_input_dir = os.path.join(utils_dir.gnn_inputs, 'train') if args.train_files is None else args.train_files
        val_input_dir = os.path.join(utils_dir.gnn_inputs, 'val') if args.val_files is None else args.val_files
        train_files = tf.io.gfile.glob(os.path.join(train_input_dir, "*"))
        eval_files = tf.io.gfile.glob(os.path.join(val_input_dir, "*"))
        ## split the number of files evenly to all ranks
        train_files = [x.tolist() for x in np.array_split(train_files, dist.size)]
        eval_files = [x.tolist() for x in np.array_split(eval_files, dist.size)]
    else:
        train_files = None
        eval_files = None

    if args.distributed:
        train_files = dist.comm.scatter(train_files, root=0)
        eval_files = dist.comm.scatter(eval_files, root=0)
    else:
        train_files = train_files[0]
        eval_files = eval_files[0]

    logging.info("rank {} has {} training files and {} evaluation files".format(
        dist.rank, len(train_files), len(eval_files)))

    raw_dataset = tf.data.TFRecordDataset(train_files)
    training_dataset = raw_dataset.map(graph.parse_tfrec_function)

    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.prefetch(AUTO)

    with_batch_dim = False
    inputs, targets = next(training_dataset.take(1).as_numpy_iterator())
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    )

    learning_rate = args.learning_rate
    optimizer = snt.optimizers.Adam(learning_rate)
    # optimizer = tf.optimizers.Adam(learning_rate)
    model = SegmentClassifier()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,
                                max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(output_dir))
    status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    # training loss
    real_weight = args.real_edge_weight
    fake_weight = args.fake_edge_weight


    def create_loss_ops(target_op, output_ops):
        weights = target_op.edges * real_weight + (1 - target_op.edges) * fake_weight
        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.edges, tf.squeeze(output_op.edges), weights=weights)
            for output_op in output_ops
        ]
        return tf.stack(loss_ops)

    @functools.partial(tf.function, input_signature=input_signature)
    def train_step(inputs_tr, targets_tr, first_batch):
        print("Tracing update_step")
        print("inputs nodes", inputs_tr.nodes.shape)
        print("inputs edges", inputs_tr.edges.shape)
        print("input n_node", inputs_tr.n_node.shape)
        print(inputs_tr.nodes)
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr, num_processing_steps_tr)
            loss_ops_tr = create_loss_ops(targets_tr, outputs_tr)
            loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

        # Horovod: add Horovod Distributed GradientTape.
        if args.distributed:
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss_op_tr, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        optimizer.apply(gradients, model.trainable_variables)

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if args.distributed and first_batch:
            hvd.broadcast_variables(model.trainable_variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables, root_rank=0)

        return loss_op_tr


    def train_epoch(dataset):
        total_loss = 0.
        num_batches = 0
        for batch, inputs in enumerate(dataset):
            input_tr, target_tr = inputs
            total_loss += train_step(input_tr, target_tr, batch==0)
            num_batches += 1
        logging.info("total batches: {}".format(num_batches))
        return total_loss/num_batches

    out_str  = "Start training " + time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "Epoch, Time [mins], Loss\n"
    log_name = os.path.join(output_dir, "training_log.txt")
    if dist.rank == 0:
        with open(log_name, 'a') as f:
            f.write(out_str)
    now = time.time()

    for epoch in range(n_epochs):
        logging.info("start epoch {} on {}".format(epoch, device))

        loss = train_epoch(training_dataset)
        this_epoch = time.time()

        logging.info("Training {} epoch, {:.2f} mins, Loss := {:.4f}".format(
            epoch, (this_epoch-now)/60., loss/global_batch_size))
        out_str = "{}, {:.2f}, {:.4f}\n".format(epoch, (this_epoch-now)/60., loss/global_batch_size)

        now = this_epoch
        if dist.rank == 0:
            with open(log_name, 'a') as f:
                f.write(out_str)
            ckpt_manager.save()

    if dist.rank == 0:
        out_log = "End @ " + time.strftime('%d %b %Y %H:%M:%S', time.localtime()) + "\n"
        with open(log_name, 'a') as f:
            f.write(out_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("--train-files", help='input TF records for training')
    add_arg("--val-files", help='input TF records for validation')
    add_arg("--output-dir", help="where the model and training info saved")
    add_arg('-d', '--distributed', action='store_true', help='data distributed training')

    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg("--learning-rate", help='learing rate', default=0.0005, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)

    add_arg("--real-edge-weight", help='weights for real edges', default=2., type=float)
    add_arg("--fake-edge-weight", help='weights for fake edges', default=1., type=float)

    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],\
        default="INFO")
    args, _ = parser.parse_known_args()

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train_and_evaluate(args)