#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools
from multiprocessing import Pool
from functools import partial

from trackml.dataset import load_event
from trackml.score import _analyze_tracks, score_event

from exatrkx import utils_dir

fontsize=16
minor_size=14
pt_bins = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 1.9, 2.4, 5]
pt_configs = {
    'bins': pt_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}


def get_plot():
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    return fig, ax


def get_ratio(x_vals, y_vals):
    res = [x/y if y!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
    err = [x/y * math.sqrt((x+y)/(x*y)) if y!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
    return res[1:], err[1:]


def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


def process(trk_file, min_hits, frac_reco_matched, frac_truth_matched, **kwargs):
    evtid = int(trk_file[:-4])
    reco_array = np.load(trk_file)
    reco_trkx = reco_array['predicts']
    submission = pd.DataFrame(reco_trkx, columns=['hit_id', 'track_id'])

    # obtain truth information from the original file
    evtdata = os.path.join(utils_dir.inputdir, "event{:09d}".format(evtid))
    hits, particles, truth = load_event(evtdata, parts=['hits', 'particles', 'truth'])
    hits = hits.merge(truth, on='hit_id', how='left')
    hits = hits[hits.particle_id > 0] # remove noise hits
    hits = hits.merge(particles, on='particle_id', how='left')
    hits = hits[hits.nhits >= min_hits]
    particles = particles[particles.nhits >= min_hits]
    par_pt = np.sqrt(particles.px**2 + particles.py**2)

    tracks = _analyze_tracks(hits, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (frac_reco_matched < purity_rec) & (frac_truth_matched < purity_maj)
    matched_pids = tracks[good_track].major_particle_id.values

    n_recotable_trkx = particles.shape[0]
    n_reco_trkx = tracks.shape[0]
    n_good_recos = np.sum(good_track)

    truth_pt_vals, _  = np.histogram(par_pt, bins=pt_bins)
    reco_pt_vals, _ = np.histogram(par_pt[particles.particle_id.isin(matched_pids)], bins=pt_bins)


    return (n_recotable_trkx, n_reco_trkx, n_good_recos, truth_pt_vals, reco_pt_vals)


def plot(reco, truth, outname):
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluating the performance of track reconstruction")
    add_arg = parser.add_argument
    add_arg("--input-dir", help="input directories where the track candidates resides.")
    add_arg("--output-dir", help='output directory that saves outputs')
    add_arg("--datatype", help='which type of dataset', choices=utils_dir.datatypes, default='test')
    add_arg('--max-evts', help='number of events to process', type=int, default=1)
    add_arg("--outname-prefix", help="prefix of output name")
    add_arg("--min-hits", help='minimum number of hits in a truth track', default=0, type=int)
    add_arg("--frac-reco-matched", help='fraction of matched hits over total hits in a reco track',
                default=0.5, type=float)
    add_arg("--frac-truth-matched", help='fraction of matched hits over total hits in a truth track',
                default=0.5, type=float)
    add_arg("--num-workers", help='number of workers', default=1, type=int)
    args = parser.parse_args()

    input_dir = os.path.join(utils_dir.trkx_output, args.datatype) if args.input_dir is None else args.input_dir
    output_dir = os.path.join(utils_dir.trkx_eval, args.datatype) if args.output_dir is None else args.trkx_output_dir
    prefix = "test" if args.outname_prefix is None else args.outname_prefix
    frac_reco_matched = args.frac_reco_matched
    frac_truth_matched = args.frac_truth_matched
    min_hits = args.min_hits


    all_files = glob.glob(os.path.join(input_dir, "*.npz"))
    n_tot_files = len(all_files)
    print("Out of {} events processing {} events".format(n_tot_files, args.max_evts))

    n_reconstructable_trkx = 0
    n_reconstructed_trkx = 0
    n_reconstructed_matched = 0
    truth_vals = []
    reco_vals = []
    with Pool(args.num_workers) as p:
        process_fnc = partial(process, **args.__dict__)
        res = p.map(process_fnc, all_files)

    print(res)
    # outname = os.path.join(output_dir, prefix)
    # plot(reco_vals, truth_vals, outname)