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
import time

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

def add_mean_std(array, x, y, ax, color='k', dy=0.3, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(x, y, "Mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(x, y-dy, "Standard Deviation: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)


def process(trk_file, min_hits, frac_reco_matched, frac_truth_matched, **kwargs):
    evtid = int(os.path.basename(trk_file)[:-4])
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
    score = tracks['major_weight'][good_track].sum()

    n_recotable_trkx = particles.shape[0]
    n_reco_trkx = tracks.shape[0]
    n_good_recos = np.sum(good_track)
    # truth_pt_vals, _  = np.histogram(par_pt, bins=pt_bins)
    # reco_pt_vals, _ = np.histogram(par_pt[particles.particle_id.isin(matched_pids)], bins=pt_bins)


    return (n_recotable_trkx, n_reco_trkx, n_good_recos, par_pt, par_pt[particles.particle_id.isin(matched_pids)], score)


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
    add_arg("--overwrite", help='overwrite existing file', action='store_true')
    args = parser.parse_args()

    input_dir = os.path.join(utils_dir.trkx_output, args.datatype) if args.input_dir is None else args.input_dir
    outdir = os.path.join(utils_dir.trkx_eval, args.datatype) if args.output_dir is None else args.output_dir
    os.makedirs(outdir, exist_ok=True)
    out_prefix = "out" if args.outname_prefix is None else args.outname_prefix
    frac_reco_matched = args.frac_reco_matched
    frac_truth_matched = args.frac_truth_matched
    min_hits = args.min_hits


    all_files = glob.glob(os.path.join(input_dir, "*.npz"))
    n_tot_files = len(all_files)
    max_evts = args.max_evts if args.max_evts > 0 and args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers".format(n_tot_files, max_evts, args.num_workers))

    out_array_name = os.path.join(outdir, "{}_trkx_pt.npz".format(out_prefix))
    if not os.path.exists(out_array_name) or args.overwrite:

        with Pool(args.num_workers) as p:
            process_fnc = partial(process, **args.__dict__)
            res = p.map(process_fnc, all_files[:max_evts])

        n_reconstructable_trkx = sum([x[0] for x in res])
        n_reconstructed_trkx = sum([x[1] for x in res])
        n_reconstructed_matched = sum([x[2] for x in res])
        truth_pt = np.concatenate([np.array(x[3]) for x in res])
        reco_pt = np.concatenate([np.array(x[4]) for x in res])
        scores = np.array([x[5] for x in res])

        outname = os.path.join(outdir, "{}_summary.txt".format(out_prefix))
        ctime = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        with open(outname, 'a') as f:
            out_str  = "Run Info: " + ctime +"\n"
            f.write("Processed {} events from {}\n".format(max_evts, input_dir))
            f.write("Reconstructable tracks:         {}\n".format(n_reconstructable_trkx))
            f.write("Reconstructed tracks:           {}\n".format(n_reconstructed_matched))
            f.write("Reconstructable tracks Matched: {}\n".format(n_reconstructed_matched))
            f.write("Tracking efficiency:            {:.4f}\n".format(n_reconstructed_matched/n_reconstructable_trkx))
            f.write("Tracking purity?:               {:.4f}\n".format(n_reconstructed_matched/n_reconstructed_trkx))

        np.savez(out_array_name, truth_pt=truth_pt, reco_pt=reco_pt, scores=scores)
    else:
        print("Reuse the existing file: {}".format(out_array_name))
        out_array = np.load(out_array_name)
        truth_pt = out_array['truth_pt']
        reco_pt = out_array['reco_pt']
        scores = out_array['scores']

    # plot the efficiency as a function of pT
    _, ax = get_plot()
    reco_vals, _, _ = ax.hist(truth_pt, **pt_configs, label="Reconstructable")
    good_vals, _, _ = ax.hist(reco_pt, **pt_configs, label="Matched")
    ax.set_xlabel("pT [GeV]", fontsize=fontsize)
    ax.set_ylabel("Events", fontsize=fontsize)
    plt.legend()
    plt.savefig(os.path.join(outdir, "{}_pt_matched.pdf".format(out_prefix)))

    _, ax = get_plot()
    matched_ratio, matched_ratio_err = get_ratio(good_vals, reco_vals)
    xvals = [0.5*(x[1]+x[0]) for x in pairwise(pt_bins)][1:]
    xerrs = [0.5*(x[1]-x[0]) for x in pairwise(pt_bins)][1:]
    ax.errorbar(xvals, matched_ratio, yerr=matched_ratio_err, fmt='o', xerr=xerrs, lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel("pT [GeV]")
    ax.set_ylabel("Track efficiency")
    ax.set_yticks(np.arange(0.5, 1.05, step=0.05))
    ax.text(1, 0.8, "pT bins: [{}] GeV".format(", ".join(["{:.1f}".format(x) for x in pt_bins[1:]])))
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "{}_efficiency.pdf".format(out_prefix)))

    _, ax = get_plot()
    ax.hist(scores)    
    add_mean_std(scores, 0.895, 6, ax=ax, dy=0.5, digits=3)
    ax.set_xlabel("trackML score")
    ax.set_ylabel("Events")
    plt.savefig(os.path.join(outdir, "{}_summary_score.pdf".format(out_prefix)))