#!/usr/bin/env python
import time
import os
import glob
from multiprocessing import Pool
from functools import partial

import numpy as np
import networkx as nx
import scipy as sp
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

import trackml.dataset
from trackml.score import score_event
from exatrkx import utils_dir


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

def clustering(used_hits, e_csr_bi, epsilon=5, min_samples=1):
    # dbscan clustering
    clustering = DBSCAN(eps=epsilon, metric='precomputed', min_samples=1).fit_predict(e_csr_bi)
    track_labels = np.vstack([np.unique(e_csr_bi.tocoo().row), clustering[np.unique(e_csr_bi.tocoo().row)]])
    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]
    new_hit_id = np.apply_along_axis(lambda x: used_hits[x], 0, track_labels.hit_id.values)
    tracks = pd.DataFrame.from_dict({"hit_id": new_hit_id, "track_id": track_labels.track_id})
    return tracks


def process(filename, edge_score_cut, epsilon, min_samples, min_num_hits, outdir, **kwargs):
    evtid = int(os.path.basename(filename)[:-4])
    array = np.load(filename)

    # infer event id from the filename
    # use it to read the initial ground truth for the event
    prefix = os.path.join(os.path.expandvars(utils_dir.inputdir),
                        'event{:09d}'.format(evtid))
    hits, particles, truth = trackml.dataset.load_event(prefix, parts=['hits', 'particles', 'truth'])
    hits = hits.merge(truth, on='hit_id', how='left')
    hits = hits.merge(particles, on='particle_id', how='left')


    used_hits = array['I']
    hits = hits[hits.hit_id.isin(used_hits)]

    n_nodes = array['I'].shape[0]
    pure_edges = array['score'] > edge_score_cut
    input_matrix = prepare(array['score'][pure_edges], array['senders'][pure_edges], array['receivers'][pure_edges], n_nodes)
    predicted_tracks = clustering(used_hits, input_matrix, epsilon=epsilon, min_samples=min_samples)

    # compare with the truth tracks that are associated with at least 5 hits
    aa = hits.groupby("particle_id")['hit_id'].count()
    pids = aa[aa > min_num_hits].index
    good_hits = hits[hits.particle_id.isin(pids)]
    score = score_event(good_hits, predicted_tracks)

    # save reconstructed tracks into a file
    np.savez(
        os.path.join(outdir, "{}.npz".format(evtid)),
        score=np.array([score]),
        predicts=predicted_tracks,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="construct tracks from the input created by the evaluate_edge_classifier")
    add_arg = parser.add_argument
    # bookkeeping
    add_arg("--max-evts", help='maximum number of events for testing', type=int, default=1)
    add_arg("--input-dir", help='input directory')
    add_arg("--output-dir", help='output file directory for track candidates')
    add_arg("--datatype", help="", default="test", choices=utils_dir.datatypes)
    add_arg("--num-workers", help='number of threads', default=1, type=int)

    # hyperparameters for DB scan
    add_arg("--edge-score-cut", help='edge score cuts', default=0, type=float)
    add_arg("--epsilon", help="epsilon in DBScan", default=0.25, type=float)
    add_arg("--min-samples", help='minimum number of samples in DBScan', default=2, type=int)

    # for tracking ML score
    add_arg("--min-num-hits", help='require minimum number of hits for each track', default=0, type=int)

    args = parser.parse_args()

    inputdir = os.path.join(utils_dir.gnn_output, args.datatype) if args.input_dir is None else args.input_dir
    outdir = os.path.join(utils_dir.trkx_output, args.datatype) if args.output_dir is None else args.output_dir
    os.makedirs(outdir, exist_ok=True)
    min_num_hits = args.min_num_hits

    all_files = glob.glob(os.path.join(inputdir, "*.npz"))
    n_tot_files = len(all_files)
    max_evts = args.max_evts if args.max_evts > 0 and args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers".format(n_tot_files, max_evts, args.num_workers))

    with Pool(args.num_workers) as p:
        process_fnc = partial(process, outdir=outdir, **args.__dict__)
        p.map(process_fnc, all_files[:max_evts])