# System imports
import sys
import os
import numpy as np

# 3rd party imports
# import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear

from torch_cluster import radius_graph
from torch_geometric.data import DataLoader

from pytorch_lightning.callbacks import Callback

# Local imports
from exatrkx.src.utils_torch import graph_intersection
from exatrkx.src import utils_torch
from exatrkx.src.embedding.embedding_base import EmbeddingBase

class LayerlessEmbedding(EmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''

        # Construct the MLP architecture
        layers = [Linear(hparams["in_channels"], hparams["emb_hidden"])]
        ln = [Linear(hparams["emb_hidden"], hparams["emb_hidden"]) for _ in range(hparams["nb_layer"]-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"], hparams["emb_dim"])
        self.norm = nn.LayerNorm(hparams["emb_hidden"])
        self.act = nn.Tanh()


    def forward(self, x):
#         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
#         x = self.norm(x) #Option of LayerNorm
        x = self.emb_layer(x)
        return x

class EmbeddingInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_train_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True) for datatype in self.datatypes]

        # Set overwrite setting if it is in config
        if "overwrite" in pl_module.hparams:
            self.overwrite = pl_module.hparams.overwrite

    def on_train_end(self, trainer, pl_module):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {"train": pl_module.trainset, "val": pl_module.valset, "test": pl_module.testset}
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        pl_module.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f'{percent:.01f}% inference complete \r')

                    # print(not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:])))
                    # print(self.overwrite)
                    #
                    # print(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))
                    if (not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))) or self.overwrite:
                        batch = batch.to(pl_module.device) #Is this step necessary??
                        batch = self.construct_downstream(batch, pl_module)
                        self.save_downstream(batch, pl_module, datatype)
                        del batch
                        torch.cuda.empty_cache()

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module):

        if 'ci' in pl_module.hparams["regime"]:
            spatial = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = pl_module(batch.x)

        e_bidir = torch.cat([batch.layerless_true_edges,
                       torch.stack([batch.layerless_true_edges[1], batch.layerless_true_edges[0]], axis=1).T], axis=-1)

        # This step should remove reliance on r_val, 
        clustering = getattr(utils_torch, pl_module.hparams.clustering)
        # and instead compute an r_build based on the EXACT r required to reach target eff/pur
        e_spatial = clustering(spatial, pl_module.hparams.r_val, pl_module.hparams.knn_val)
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # remove edges that point from outter region to inner region
        R_dist = torch.sqrt(batch.x[:,0]**2 + batch.x[:,2]**2) # distance away from origin...
        sel_idx = R_dist[e_spatial[0]] <= R_dist[e_spatial[1]]
        
        batch.e_radius = e_spatial[:, sel_idx]
        batch.y = torch.from_numpy(y_cluster).float()[sel_idx]

        return batch

    def save_downstream(self, batch, pl_module, datatype):

        with open(os.path.join(self.output_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
            torch.save(batch, pickle_file)
