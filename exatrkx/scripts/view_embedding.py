#/usr/bin/env python
# system import
import os
import pkg_resources
import yaml
import pprint
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# 3rd party
import torch
from torch_geometric.data import Data
from trackml.dataset import load_event


# local import
from heptrkx.dataset import event as master
from exatrkx import config_dict # for accessing predefined configuration files
from exatrkx import outdir_dict # for accessing predefined output directories
from exatrkx.src import utils_dir
from exatrkx.src import utils_torch
from exatrkx import LayerlessEmbedding


def start_view(args):
    outdir = args.outdir
    event = master.Event(utils_dir.inputdir)
    event.read(args.evtid)

    # randomly select N particles with each having at least 6 hits
    pids = event.particles[(event.particles.nhits) > 5]
    np.random.seed(args.seed)
    rnd = np.random.randint(0, pids.shape[0], args.npids)
    sel_pids = pids.particle_id.values[rnd]

    event._hits = event.hits[event.hits.particle_id.isin(sel_pids)]
    hits = event.cluster_info(utils_dir.detector_path)

    # track labeling -- determine true edges...
    hits = hits.assign(R=np.sqrt((hits.x - hits.vx)**2 + (hits.y - hits.vy)**2 + (hits.z - hits.vz)**2))
    hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)
    hit_list = hits.groupby(['particle_id', 'layer'], sort=False)['index'].agg(lambda x: list(x)).groupby(level=0).agg(lambda x: list(x))
    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))
    layerless_true_edges = np.array(e).T

    # input data for embedding 
    data = Data(x=torch.from_numpy(hits[['r', 'phi', 'z']].to_numpy()/np.array([1000, np.pi, 1000])).float(),\
            pid=torch.from_numpy(hits.particle_id.to_numpy()),
            layers=torch.from_numpy(hits.layer.to_numpy()), hid=torch.from_numpy(hits.hit_id.to_numpy()))
    cell_features = ['cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    data.layerless_true_edges = torch.from_numpy(layerless_true_edges)
    data.cell_data = torch.from_numpy(hits[cell_features].values).float()

    action = 'embedding'

    config_file = pkg_resources.resource_filename(
                        "exatrkx",
                        os.path.join('configs', config_dict[action]))
    with open(config_file) as f:
        e_config = yaml.load(f, Loader=yaml.FullLoader)

    e_config['train_split'] = [1, 0, 0]
    e_config['r_val'] = 2.0
    e_model = LayerlessEmbedding(e_config)
    e_model = e_model.load_from_checkpoint(args.embed_ckpt_dir, hparams=e_config)
    e_model.eval()
    spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1))
    spatial_np = spatial.detach().numpy()

    # plot hits in the embedding space
    embedding_dims = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for id1, id2 in embedding_dims:
        fig = plt.figure(figsize=(6,6))
        for pid in sel_pids:
            idx = hits.particle_id == pid
            plt.scatter(spatial_np[idx, id1], spatial_np[idx, id2])
            
        plt.savefig(os.path.join(outdir, "embedding_{}_{}.pdf".format(id1, id2)))
        del fig

    # build edges from the embedding space
    e_spatial = utils_torch.build_edges(spatial, e_model.hparams['r_val'], e_model.hparams['knn_val'])
    e_spatial_np = e_spatial.detach().numpy()

    # view hits with or without edge candidates...
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    for pid in sel_pids:
        ax.scatter(hits[hits.particle_id == pid].x.values, hits[hits.particle_id == pid].y.values,  hits[hits.particle_id == pid].z.values)
    # add edges
    e_spatial_np_t = e_spatial_np.T
    for iedge in range(e_spatial_np.shape[1]):
        ax.plot(hits.iloc[e_spatial_np_t[iedge]].x.values, hits.iloc[e_spatial_np_t[iedge]].y.values, hits.iloc[e_spatial_np_t[iedge]].z.values, color='k', alpha=0.3, lw=1.)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(os.path.join(outdir, "emedding_edges_3d.pdf"))
    del fig
    del ax

    e_spatial_np_t = e_spatial_np.T
    layerless_true_edges_t = layerless_true_edges.T # same as e
    def plot_edges(xname, yname, xlabel, ylabel, outname, with_edges=True, no_axis=False, edges=e_spatial_np_t):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        for pid in sel_pids:
            ax.scatter(hits[hits.particle_id == pid][xname].values, hits[hits.particle_id == pid][yname].values)
        # add edges
        if with_edges:
            for iedge in range(edges.shape[0]):
                ax.plot(hits.iloc[edges[iedge]][xname].values,\
                        hits.iloc[edges[iedge]][yname].values, color='k', alpha=0.3, lw=1.)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        if xname=='z':
            ax.set_xlim(-3000, 3000)
        trans=False
        if no_axis:
            ax.set_axis_off()
            trans=True
            plt.savefig(os.path.join(outdir, "{}.png".format(outname)), transparent=trans)
        plt.savefig(os.path.join(outdir, "{}.pdf".format(outname)), transparent=trans)

    def plot_hits(xname, yname, outname):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(hits[xname].values, hits[yname].values)
        if xname=='z':
            ax.set_xlim(-3000, 3000)
        ax.set_xlabel(xname, fontsize=16)
        ax.set_ylabel(yname, fontsize=16)
        plt.savefig(os.path.join(outdir, "{}.pdf".format(outname)))

    plot_edges("x", 'y', 'x', 'y', 'embedding_edges_x_y')
    plot_edges("z", 'r', 'z', 'r', 'embedding_edges_z_r')
    plot_edges("x", 'y', 'x', 'y', 'embedding_edges_truth_x_y', edges=layerless_true_edges_t)
    plot_edges("z", 'r', 'z', 'r', 'embedding_edges_truth_z_r', edges=layerless_true_edges_t)
    plot_edges("x", 'y', 'x', 'y', 'embedding_hits_truth_x_y', with_edges=False)
    plot_edges("z", 'r', 'z', 'r', 'embedding_hits_truth_z_r', with_edges=False)
    plot_hits("x", 'y', 'embedding_hits_x_y')
    plot_hits("z", 'r', 'embedding_hits_z_r')
    plot_edges("x", 'y', 'x', 'y', 'embedding_front', no_axis=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="view embedding results")
    add_arg = parser.add_argument
    add_arg("embed_ckpt_dir", help="embedding checkpoint")
    add_arg("outdir", help="output directory")
    add_arg("--evtid", default=8000, type=int, help='event id')
    add_arg("--npids", default=10, type=int, help='number of particles')
    add_arg("--seed", default=456, type=int, help='seeding for selecting particles')
    args = parser.parse_args()
    
    start_view(args)