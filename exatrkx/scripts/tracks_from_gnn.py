#!/usr/bin/env python
import time
import os

import numpy as np
import networkx as nx
import scipy as sp
from sklearn.cluster import DBSCAN
import pandas as pd

import trackml.dataset
from trackml.score import score_event
from exatrkx.src import utils_dir


def prepare(score, senders, receivers, n_nodes):
    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.sparse.csr_matrix( (score, (senders, receivers)), shape=(n_nodes, n_nodes), dtype=np.float32)
    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1]/2.
    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    # make it symmetric
    e_csr_bi = sp.sparse.coo_matrix((np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]), 
                                    np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),                                                                   
                                                np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])])))
    return e_csr_bi

def clustering(e_csr_bi, epsilon=5, min_samples=1):
    # dbscan clustering
    clustering = DBSCAN(eps=epsilon, metric='precomputed', min_samples=1).fit_predict(e_csr_bi)
    track_labels = np.vstack([np.unique(e_csr_bi.tocoo().row), clustering[np.unique(e_csr_bi.tocoo().row)]])
    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]
    new_hit_id = np.apply_along_axis(lambda x: used_hits[x], 0, track_labels.hit_id.values)
    tracks = pd.DataFrame.from_dict({"hit_id": new_hit_id, "track_id": track_labels.track_id})
    return tracks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="construct tracks from the input created by the evaluate_edge_classifier")
    add_arg = parser.add_argument
    add_arg("--max-evts", help='maximum number of events for testing', type=int, default=1)
    add_arg("--input-dir", help='input directory')
    args = parser.parse_args()

    inputdir = os.path.join(utils_dir.gnn_output, "test") if args.input_dir is None else args.input_dir
    # print("input directory:", inputdir)
    tot_files = os.listdir(inputdir)
    print("total {} testing files".format(len(tot_files)))
    nevts = args.max_evts
    if len(tot_files) < nevts:
        nevts = len(tot_files)

    for evtid in tot_files:
        print("Processing event: {}".format(evtid))
        filedir = os.path.join(inputdir, evtid)
        evtid = int(evtid[:-4])

        array = np.load(filedir)
        prefix = os.path.join(os.path.expandvars(utils_dir.inputdir),
                            'event{:09d}'.format(evtid))
        hits, particles, truth = trackml.dataset.load_event(prefix, parts=['hits', 'particles', 'truth'])
        hits = hits.merge(truth, on='hit_id', how='left')
        hits = hits.merge(particles, on='particle_id', how='left')

        # print(hits.shape, hits.dtypes)
        used_hits = array['I']
        hits = hits[hits.hit_id.isin(used_hits)]
        n_nodes = array['I'].shape[0]
        # print("after filtering", hits.shape)
        # print("edges: {}".format(array['score'].shape[0]))

        hit_id = hits.hit_id.to_numpy()

        pure_edges = array['score'] > 0
        # print(pure_edges.shape, np.sum(pure_edges))
        input_matrix = prepare(array['score'][pure_edges], array['senders'][pure_edges], array['receivers'][pure_edges], n_nodes)
        predicted_tracks = clustering(input_matrix, epsilon=0.25, min_samples=2)
        print(predicted_tracks.shape)

        # compare with the truth tracks that are associated with at least 5 hits
        aa = hits.groupby("particle_id")['hit_id'].count()
        pids = aa[aa > 5].index
        good_hits = hits[hits.particle_id.isin(pids)]

        print("Event {} has track ML score: {:.4f}".format(evtid, score_event(good_hits, predicted_tracks)))