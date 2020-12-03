#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TRKXINPUTDIR'] = '/global/cscratch1/sd/alazar/trackml/data/train_100_events/' # better change to your copy of the dataset.
os.environ['TRKXOUTPUTDIR'] = '../run200' # change to your own directory

# system import
import pkg_resources
import yaml
import pprint
import random
random.seed(1234)
import numpy as np
import pandas as pd
import itertools
import time
import gc
import matplotlib.pyplot as plt
# %matplotlib widget

# 3rd party
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from trackml.dataset import load_event
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# local import
# from heptrkx.dataset import event as master
from exatrkx import config_dict # for accessing predefined configuration files
from exatrkx import outdir_dict # for accessing predefined output directories
from exatrkx.src import utils_dir

# for preprocessing
from exatrkx import FeatureStore
from exatrkx.src import utils_torch

# for embedding
from exatrkx import LayerlessEmbedding
from exatrkx.src import utils_torch

# for filtering
from exatrkx import VanillaFilter

# for GNN
import tensorflow as tf
from graph_nets import utils_tf
from exatrkx import SegmentClassifier
import sonnet as snt

# for labeling
from exatrkx.scripts.tracks_from_gnn import prepare as prepare_labeling
from exatrkx.scripts.tracks_from_gnn import clustering as dbscan_clustering

# track efficiency
from trackml.score import _analyze_tracks
from exatrkx.scripts.eval_reco_trkx import make_cmp_plot, pt_configs, eta_configs
from functools import partial

# In[10]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#print(tf.config.list_physical_devices('GPU'))

# ### Setup some hyperparameters and event

# In[3]:
start_all = time.time()
start_all_cpu = time.process_time()
embed_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/embedding/checkpoints/epoch=10.ckpt'
filter_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/filtering/checkpoints/epoch=92.ckpt'
gnn_ckpt_dir = '/global/cfs/cdirs/m3443/data/lightning_models/gnn'
plots_dir = '../run200' # needs to change...
ckpt_idx = -1 # which GNN checkpoint to load
dbscan_epsilon, dbscan_minsamples = 0.25, 2 # hyperparameters for DBScan
min_hits = 5 # minimum number of hits associated with a particle to define "reconstructable particles"
frac_reco_matched, frac_truth_matched = 0.5, 0.5 # parameters for track matching

# In[4]:
evtid = 1000
event_file = os.path.join(utils_dir.inputdir, 'event{:09}'.format(evtid))

# ### Preprocessing

# In[5]:
action = 'build'

config_file = pkg_resources.resource_filename(
                    "exatrkx",
                    os.path.join('configs', config_dict[action]))
with open(config_file) as f:
    b_config = yaml.load(f, Loader=yaml.FullLoader)
    
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(b_config)

# In[6]:
b_config['pt_min'] = 0
b_config['endcaps'] = True
b_config['n_workers'] = 1
b_config['n_files'] = 1

# In[9]:
# this cell is only needed for the first run to prodcue the dataset
#preprocess_dm = FeatureStore(b_config)
#preprocess_dm.prepare_data()

# ### Read the preprocessed data
# In[8]:
data = torch.load(os.path.join(utils_dir.feature_outdir, str(evtid))).to(device)

# ### Evaluating Embedding
# In[9]:
e_ckpt = torch.load(embed_ckpt_dir, map_location='cpu')
e_config = e_ckpt['hyper_parameters']
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(e_config)

# In[10]:
e_config = e_ckpt['hyper_parameters']
e_config['clustering'] = 'build_edges'
e_config['knn_val'] = 500
e_config['r_val'] = 1.7

# Load the checkpoint and put the model in the evaluation state.
# In[11]:
e_model = LayerlessEmbedding(e_config).to(device)
e_model.load_state_dict(e_ckpt["state_dict"])

# In[12]:
e_model.eval()

# Map each hit to the embedding space, return the embeded parameters for each hit

# In[13]:
start = time.time()
start_cpu = time.process_time()
with torch.no_grad():
    spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1)) #.to(device)
end = time.time()
end_cpu = time.process_time()
wall_time=int(round((end - start) * 1000))
cpu_time=int(round((end_cpu - start_cpu) * 1000))
print("Time for embedding wall: %f cpu: %f" % (wall_time,cpu_time))
# ### From embeddeding space form doublets

# `r_val = 1.7` and `knn_val = 500` are the hyperparameters to be studied.
# 
# * `r_val` defines the radius of the clustering method
# * `knn_val` defines the number of maximum neighbors in the embedding space

# In[14]:
start = time.time()
start_cpu = time.process_time()
e_spatial = utils_torch.build_edges(spatial.to(device), e_model.hparams['r_val'], e_model.hparams['knn_val'])
end = time.time()
end_cpu = time.process_time()
print("Time for build edges: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
# In[15]:
#e_spatial = e_spatial.cpu().numpy()


# Removing edges that point from outer region to inner region, which almost removes half of edges.
# In[16]:
R_dist = torch.sqrt(data.x[:,0]**2 + data.x[:,2]**2) # distance away from origin...
e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

# ### Filtering
# 
# In[17]:
f_ckpt = torch.load(filter_ckpt_dir, map_location='cpu')
f_config = f_ckpt['hyper_parameters']
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(f_config)

# In[18]:
f_config['train_split'] = [0, 0, 1]
f_config['filter_cut'] = 0.18

# In[19]:
f_model = VanillaFilter(f_config).to(device)
# f_model = f_model.load_from_checkpoint(filter_ckpt_dir, hparams=f_config)
f_model.load_state_dict(f_ckpt['state_dict'])

# In[20]:
f_model.eval()

# In[21]:
#%%time
#emb = None # embedding information was not used in the filtering stage.
#output = f_model(torch.cat([data.cell_data, data.x], axis=-1), e_spatial, emb).squeeze()

# In[ ]:
start = time.time()
start_cpu = time.process_time()
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
end = time.time()
end_cpu = time.process_time()
print("Time to run the filtering inference: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))

# In[22]:
output = torch.sigmoid(output)

# In[23]:
output.shape, e_spatial.shape

# In[24]:
# this plot may need some time to load...
#plt.hist(output.detach().to('cpu').numpy(), );

# The filtering network assigns a score to each edge. In the end, edges with socres > `filter_cut` are selected to construct graphs.

# In[25]:
edge_list = e_spatial[:, output.to('cpu') > f_model.hparams['filter_cut']]
#edge_list = e_spatial[:, output > f_model.hparams['filter_cut']]
# In[26]:
edge_list.shape

# ### Form a graph
# Now moving TensorFlow for GNN inference.

# In[27]:
n_nodes = data.x.shape[0]
n_edges = edge_list.shape[1]
#nodes = data.x.numpy().astype(np.float32)
nodes = data.x.cpu().numpy().astype(np.float32)
edges = np.zeros((n_edges, 1), dtype=np.float32)
senders = edge_list[0].cpu()
receivers = edge_list[1].cpu()

# In[28]:
input_datadict = {
    "n_node": n_nodes,
    "n_edge": n_edges,
    "nodes": nodes,
    "edges": edges,
    "senders": senders,
    "receivers": receivers,
    "globals": np.array([n_nodes], dtype=np.float32)
}

# In[29]:
input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])

# ### Apply GNN

# In[30]:
#start = time.time()
#start_cpu = time.process_time()
num_processing_steps_tr = 8
optimizer = snt.optimizers.Adam(0.001)
model = SegmentClassifier()

output_dir = gnn_ckpt_dir
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()
#end = time.time()
#end_cpu = time.process_time()
#print("Time to load the checkpoint: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
#print("Loaded {} checkpoint from {}".format(ckpt_idx, output_dir))
del e_model
del f_model
gc.collect()
torch.cuda.empty_cache()
# In[31]:
start = time.time()
start_cpu = time.process_time()
outputs_gnn = model(input_graph, num_processing_steps_tr)
output_graph = outputs_gnn[-1]
end = time.time()
end_cpu = time.process_time()
print("Time to apply the GNN: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
# ### Track labeling

# In[32]:
start = time.time()
start_cpu = time.process_time()

input_matrix = prepare_labeling(tf.squeeze(output_graph.edges).cpu().numpy(), senders, receivers, n_nodes)
predict_tracks = dbscan_clustering(data.hid.cpu(), input_matrix, dbscan_epsilon, dbscan_minsamples)
end = time.time()
end_cpu = time.process_time()
print("Time to get labels with DBSCAN: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
# ### Track Efficiency

end_all = time.time()
end_all_cpu = time.process_time()
print("Time from begining to end: %f cpu: %f" % ((end_all - start_all),(end_all_cpu - start_all_cpu)))

# In[34]:
start = time.time()
start_cpu = time.process_time()
hits, particles, truth = load_event(event_file, parts=['hits', 'particles', 'truth'])
hits = hits.merge(truth, on='hit_id', how='left')
hits = hits[hits.particle_id > 0] # remove noise hits
hits = hits.merge(particles, on='particle_id', how='left')
hits = hits[hits.nhits >= min_hits]
particles = particles[particles.nhits >= min_hits]
par_pt = np.sqrt(particles.px**2 + particles.py**2)
momentum = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
ptheta = np.arccos(particles.pz/momentum)
peta = -np.log(np.tan(0.5*ptheta))

# In[35]:
tracks = _analyze_tracks(hits, predict_tracks)

# In[36]:
purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
good_track = (frac_reco_matched < purity_rec) & (frac_truth_matched < purity_maj)

matched_pids = tracks[good_track].major_particle_id.values
score = tracks['major_weight'][good_track].sum()

n_recotable_trkx = particles.shape[0]
n_reco_trkx = tracks.shape[0]
n_good_recos = np.sum(good_track)
matched_idx = particles.particle_id.isin(matched_pids).values

# In[37]:
print("Processed {} events from {}".format(evtid, utils_dir.inputdir))
print("Reconstructable tracks:         {}".format(n_recotable_trkx))
print("Reconstructed tracks:           {}".format(n_reco_trkx))
print("Reconstructable tracks Matched: {}".format(n_good_recos))
print("Tracking efficiency:            {:.4f}".format(n_good_recos/n_recotable_trkx))
print("Tracking purity:               {:.4f}".format(n_good_recos/n_reco_trkx))

# In[38]:
#make_cmp_plot_fn = partial(make_cmp_plot, xlegend="Matched", ylegend="Reconstructable",
#                    ylabel="Events", ratio_label='Track efficiency')

# In[39]:
#make_cmp_plot_fn(par_pt[matched_idx], par_pt,
#                 configs=pt_configs,
#                 xlabel="pT [GeV]",
#                 outname=os.path.join(plots_dir, "{}_pt".format(evtid)))

# In[40]:
#make_cmp_plot_fn(peta[matched_idx], peta,
#                 configs=eta_configs,
#                 xlabel=r"$\eta$",
#                 outname=os.path.join(plots_dir, "{}_eta".format(evtid)))
