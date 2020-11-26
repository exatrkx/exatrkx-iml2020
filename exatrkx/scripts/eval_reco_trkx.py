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
eta_bins = np.arange(-4, 4.4, step=0.4)
eta_configs = {
    'bins': eta_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}


def get_plot():
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    return fig, ax


def get_ratio(x_vals, y_vals):
    res = [x/y if y!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
    err = [x/y * math.sqrt((x+y)/(x*y)) if y!=0 and x!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
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

def make_cmp_plot(arrays, legends, configs, xlabel, ylabel, ratio_label, ratio_legends, outname):
    _, ax = get_plot()
    vals_list = []
    for array,legend in zip(arrays, legends):
        vals, bins, _ = ax.hist(array, **configs, label=legend)
        vals_list.append(vals)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    plt.grid(True)
    plt.savefig("{}.pdf".format(outname))

    # make a ratio plot
    _, ax = get_plot()
    xvals = [0.5*(x[1]+x[0]) for x in pairwise(bins)][1:]
    xerrs = [0.5*(x[1]-x[0]) for x in pairwise(bins)][1:]
    # ax.text(1, 0.8, "bins: [{}] GeV".format(", ".join(["{:.1f}".format(x) for x in pt_bins[1:]])))
    for idx in range(1, len(arrays)):
        ratio, ratio_err = get_ratio(vals_list[2], vals_list[idx-1])
        ax.errorbar(xvals, ratio, yerr=ratio_err, fmt='o', xerr=xerrs, lw=2, label=ratio_legends[idx-1])
        

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ratio_label)
    ax.set_yticks(np.arange(0.5, 1.05, step=0.05))
    ax.set_ylim(0.5, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig("{}_ratio.pdf".format(outname))

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
    hits = hits[hits.nhits > min_hits]

    par_pt = np.sqrt(particles.px**2 + particles.py**2)
    momentum = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
    ptheta = np.arccos(particles.pz/momentum)
    peta = -np.log(np.tan(0.5*ptheta))
    reconstructable_pars = particles.nhits > min_hits

    tracks = _analyze_tracks(hits, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (frac_reco_matched < purity_rec) & (frac_truth_matched < purity_maj)

    matched_pids = tracks[good_track].major_particle_id.values
    score = tracks['major_weight'][good_track].sum()

    n_recotable_trkx = particles.shape[0]
    n_reco_trkx = tracks.shape[0]
    n_good_recos = np.sum(good_track)
    matched_idx = particles.particle_id.isin(matched_pids).values

    return (n_recotable_trkx, n_reco_trkx, n_good_recos, par_pt, peta, matched_idx, score, reconstructable_pars)


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
    print("Output directory:", outdir)

    out_array_name = os.path.join(outdir, "{}_trkx_pt_eta.npz".format(out_prefix))
    if not os.path.exists(out_array_name) or args.overwrite:

        with Pool(args.num_workers) as p:
            process_fnc = partial(process, **args.__dict__)
            res = p.map(process_fnc, all_files[:max_evts])

        # merge results from each process
        n_reconstructable_trkx = sum([x[0] for x in res])
        n_reconstructed_trkx = sum([x[1] for x in res])
        n_reconstructed_matched = sum([x[2] for x in res])
        truth_pt = np.concatenate([np.array(x[3]) for x in res])
        truth_eta = np.concatenate([np.array(x[4]) for x in res])
        matched_idx = np.concatenate([np.array(x[5]) for x in res])
        scores = np.array([x[6] for x in res])
        rectable_idx = np.concatenate([np.array(x[7]) for x in res])

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

        np.savez(out_array_name, truth_pt=truth_pt, truth_eta=truth_eta, 
                rectable_idx=rectable_idx, matched_idx=matched_idx, scores=scores)
    else:
        print("Reuse the existing file: {}".format(out_array_name))
        out_array = np.load(out_array_name)
        truth_pt = out_array['truth_pt']
        truth_eta = out_array['truth_eta']
        rectable_idx = out_array['rectable_idx']
        matched_idx = out_array['matched_idx']
        scores = out_array['scores']


    # plot the efficiency as a function of pT, eta
    make_cmp_plot_fn = partial(make_cmp_plot, legends=["Generated", "Reconstructable", "Matched"],
                        ylabel="Events", ratio_label='Track efficiency', ratio_legends=["Physics Eff", "Technical Eff"])
    # fiducial cuts: pT > 1 GeV and |eta| < 4
    all_cuts = [(1, 4), (0.5, 4), (0., 4)]
    for (cut_pt, cut_eta) in all_cuts:
        cuts = (truth_pt > cut_pt) & (np.abs(truth_eta) < cut_eta)
        gen_pt = truth_pt[cuts]
        true_pt = truth_pt[cuts & rectable_idx]
        reco_pt = truth_pt[cuts & rectable_idx & matched_idx]
        make_cmp_plot_fn([gen_pt, true_pt, reco_pt], 
            configs=pt_configs, xlabel="pT [GeV]",
            outname=os.path.join(outdir, "{}_pt-cut{}_{}".format(out_prefix, cut_pt, cut_eta)))

        gen_eta = truth_eta[cuts]
        true_eta = truth_eta[cuts & rectable_idx]
        reco_eta = truth_eta[cuts & rectable_idx & matched_idx]
        make_cmp_plot_fn([gen_eta, true_eta, reco_eta], configs=eta_configs, xlabel=r"$\eta$",
            outname=os.path.join(outdir, "{}_eta-cut{}_{}".format(out_prefix, cut_pt, cut_eta)))

    # pt_ths = [0.5, 1]
    # for pt_th in pt_ths:
    #     reco_eta_pt = reco_eta[reco_pt >= pt_th]
    #     truth_eta_pt = truth_eta[truth_pt >= pt_th]
    #     make_cmp_plot_fn(reco_eta_pt, truth_eta_pt, configs=eta_configs,
    #             xlabel=r"$\eta$ of tracks with pT > {} GeV".format(pt_th),
    #             outname=os.path.join(outdir, "{}_eta_pt_gt{}GeV".format(out_prefix, pt_th)))

    _, ax = get_plot()
    ax.hist(scores)    
    add_mean_std(scores, 0.895, 6, ax=ax, dy=0.5, digits=3)
    ax.set_xlabel("trackML score")
    ax.set_ylabel("Events")
    plt.savefig(os.path.join(outdir, "{}_summary_score.pdf".format(out_prefix)))