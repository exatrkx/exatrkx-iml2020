#!/usr/bin/env python
import os
import torch

from exatrkx import config_dict
from exatrkx import outdir_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count number of nodes and edges at different stages, using training folder")
    add_arg = parser.add_argument
    add_arg("outname", help='output name')
    add_arg("--action", help="which stage", choices=['build', 'embedding', 'filtering'], required=True)
    add_arg("--datatype", default='train', choices=['train', 'val', 'test'], help='which dataset')
    args = parser.parse_args()

    if args.action == 'build':
        datatype = "all"
        input_dir = outdir_dict[args.action]
    else:
        input_dir = os.path.join(outdir_dict[args.action], args.datatype)
        datatype = args.datatype

    all_files = os.listdir(input_dir)
    print("Total {} files".format(len(all_files)))

    n_nodes = []
    n_edges = []
    n_tot_truth = 0
    n_tot_edges = 0
    keyname = {
        "build": "layerless_true_edges", 
        'embedding': "e_radius",
        'filtering': "e_radius",
    }
    for filename in all_files:
        filename = os.path.join(input_dir, filename)
        dd = torch.load(filename, map_location='cpu')
        n_nodes.append(dd.x.shape[0])
        n_edges.append(dd[keyname[args.action]].shape[1])
        if args.action != 'build':
            n_tot_truth += dd['layerless_true_edges'].shape[1]

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="ticks")
    import numpy as np
    n_nodes = np.array(n_nodes)
    n_edges = np.array(n_edges)
    n_tot_edges = np.sum(n_edges)

    _, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    pp = sns.jointplot(x=n_nodes, y=n_edges, kind='hex', color='#4CB391')
    pp.set_axis_labels('number of nodes', 'number of edges')
    if args.action != "build":
        import math
        def ratio_error(a, b, in_percentage=False):
            ratio = a/b
            if in_percentage:
                ratio *= 100
            error = ratio * math.sqrt((a+b)/(a*b))
            return ratio, error

        pp.ax_joint.text(np.min(n_nodes)*1.05, np.max(n_edges)*0.95,
                "edge purity: {0:.2f}$\pm${1:.2f} %".format(*ratio_error(n_tot_truth,n_tot_edges, True)),
                fontsize=14)
        pp.ax_joint.text(np.min(n_nodes)*1.05, np.max(n_edges)*0.90,
            "{} {}".format(args.action, datatype), fontsize=14)
    pp.savefig("{}_{}_{}.pdf".format(args.outname, args.action, datatype))