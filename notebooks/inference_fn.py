#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# 3rd party
import torch
import tensorflow as tf
import sonnet as snt
from graph_nets import utils_tf
import gc

from trackml.dataset import load_event

# local import
from exatrkx import LayerlessEmbedding
from exatrkx.src import utils_torch
from exatrkx import VanillaFilter
from exatrkx import SegmentClassifier

# for labeling
from exatrkx.scripts.tracks_from_gnn import prepare as prepare_labeling
from exatrkx.scripts.tracks_from_gnn import clustering as dbscan_clustering

def gnn_track_finding(
    x, cell_data,
    embed_ckpt_dir='/global/cfs/cdirs/m3443/data/lightning_models/embedding/checkpoints/epoch=10.ckpt',
    filter_ckpt_dir='/global/cfs/cdirs/m3443/data/lightning_models/filtering/checkpoints/epoch=92.ckpt',
    gnn_ckpt_dir='/global/cfs/cdirs/m3443/data/lightning_models/gnn',
    ckpt_idx=-1, dbscan_epsilon=0.25, dbscan_minsamples=2
    ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ### Setup some hyperparameters and event

    # embed_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/embedding/checkpoints/epoch=10.ckpt'
    # filter_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/filtering/checkpoints/epoch=92.ckpt'
    # gnn_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/gnn'
    # ckpt_idx = -1 # which GNN checkpoint to load
    # dbscan_epsilon, dbscan_minsamples = 0.25, 2 # hyperparameters for DBScan
    # min_hits = 5 # minimum number of hits associated with a particle to define "reconstructable particles"
    # frac_reco_matched, frac_truth_matched = 0.5, 0.5 # parameters for track matching


    data = torch.dataset(x=x, cell_data=cell_data)

    # ### Evaluating Embedding
    # In[9]:
    e_ckpt = torch.load(embed_ckpt_dir, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_config['clustering'] = 'build_edges'
    e_config['knn_val'] = 500
    e_config['r_val'] = 1.7

    e_model = LayerlessEmbedding(e_config).to(device)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.eval()

    # Map each hit to the embedding space, return the embeded parameters for each hit
    with torch.no_grad():
        spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1)) #.to(device)

    # ### From embeddeding space form doublets

    # `r_val = 1.7` and `knn_val = 500` are the hyperparameters to be studied.
    # 
    # * `r_val` defines the radius of the clustering method
    # * `knn_val` defines the number of maximum neighbors in the embedding space

    e_spatial = utils_torch.build_edges(spatial.to(device), e_model.hparams['r_val'], e_model.hparams['knn_val'])


    # Removing edges that point from outer region to inner region, which almost removes half of edges.
    # In[16]:
    R_dist = torch.sqrt(data.x[:,0]**2 + data.x[:,2]**2) # distance away from origin...
    e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

    f_ckpt = torch.load(filter_ckpt_dir, map_location='cpu')
    f_config = f_ckpt['hyper_parameters']
    f_config['train_split'] = [0, 0, 1]
    f_config['filter_cut'] = 0.18

    f_model = VanillaFilter(f_config).to(device)
    f_model.load_state_dict(f_ckpt['state_dict'])
    f_model.eval()

    emb = None # embedding information was not used in the filtering stage.
    chunks = 8
    output_list = []
    for j in range(chunks):
        subset_ind = torch.chunk(torch.arange(e_spatial.shape[1]), chunks)[j]
        with torch.no_grad():
            output = f_model(torch.cat([data.cell_data, data.x], axis=-1), e_spatial[:, subset_ind], emb).squeeze()  #.to(device)
        output_list.append(output)
        del subset_ind
        del output
        gc.collect()
    output = torch.cat(output_list)
    output = torch.sigmoid(output)

    # The filtering network assigns a score to each edge. 
    # In the end, edges with socres > `filter_cut` are selected to construct graphs.
    edge_list = e_spatial[:, output.to('cpu') > f_model.hparams['filter_cut']]

    # ### Form a graph
    # Now moving TensorFlow for GNN inference.


    n_nodes = data.x.shape[0]
    n_edges = edge_list.shape[1]
    nodes = data.x.cpu().numpy().astype(np.float32)
    edges = np.zeros((n_edges, 1), dtype=np.float32)
    senders = edge_list[0].cpu()
    receivers = edge_list[1].cpu()

    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([n_nodes], dtype=np.float32)
    }

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])

    num_processing_steps_tr = 8
    optimizer = snt.optimizers.Adam(0.001)
    model = SegmentClassifier()

    output_dir = gnn_ckpt_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
    status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()

    # clean up GPU memory
    del e_model
    del f_model
    gc.collect()
    torch.cuda.empty_cache()

    outputs_gnn = model(input_graph, num_processing_steps_tr)
    output_graph = outputs_gnn[-1]

    # ### Track labeling
    input_matrix = prepare_labeling(tf.squeeze(output_graph.edges).cpu().numpy(), senders, receivers, n_nodes)
    predict_tracks = dbscan_clustering(data.hid.cpu(), input_matrix, dbscan_epsilon, dbscan_minsamples)
    return predict_tracks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="perform inference")
    add_arg = parser.add_argument
    add_arg("event_file", help="event id")
    add_arg('detector_path', help='detector path')
    args = parser.parse_args()

    # evtid = args.event_id
    # event_file = os.path.join(utils_dir.inputdir, 'event{:09}'.format(evtid))
    event_file = args.event_file
    hits, particles, truth = load_event(event_file, parts=['hits', 'particles', 'truth'])

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    hits = hits.assign(r=r, phi=phi)

    from exatrkx.src.processing.utils.detector_utils import load_detector
    from exatrkx.src.processing.utils.cell_utils import get_one_event
    detector_orig, detector_proc = load_detector(args.detector_path)
    angles = get_one_event(event_file, detector_orig, detector_proc)
    hits = hits.merge(angles, on='hit_id')

    cell_features = ['cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    x = hits[['r', 'phi', 'z']]
    cell_data = hits[cell_features]

    gnn_track_finding(x, cell_data)