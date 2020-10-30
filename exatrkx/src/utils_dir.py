import os


try:
    inputdir = os.environ['TRKXINPUTDIR']
except KeyError as e:
    print("Require the directory of tracking ML dataset")
    print("Given by environment variable: TRKXINPUTDIR")
    raise(e)

try:
    output_base = os.environ['TRKXOUTPUTDIR']
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise(e)

# print("Input: {}".format(inputdir))
# print("Output: {}".format(output_base))
detector_path = os.path.join(inputdir, '..', 'detectors.csv')

feature_outdir   = os.path.join(output_base, "feature_store") # store converted input information
embedding_outdir = os.path.join(output_base, "embedding_output") # directory outputs after embedding
filtering_outdir = os.path.join(output_base, "filtering_output") # directory outputs after filtering
gnn_inputs       = os.path.join(output_base, "gnn_inputs")       # directory for converted filtering outputs
gnn_models       = os.path.join(output_base, "gnn_models")       # GNN model outputs
gnn_output       = os.path.join(output_base, "gnn_eval")         # directory for outputs after evalating GNN
trkx_output      = os.path.join(output_base, "trkx_output")      # directory for outputs of track candidates
trkx_eval        = os.path.join(output_base, "trkx_eval")        # directory for evaluating track candidates

outdirs = [feature_outdir, embedding_outdir, filtering_outdir,
        gnn_inputs, gnn_models, gnn_output, trkx_output]

if not os.path.exists(feature_outdir):
    [os.makedirs(x, exist_ok=True) for x in outdirs]

datatypes = ['train', 'val', 'test']

config_dict = {
    "build": 'prepare_feature_store.yaml',
    'embedding': 'train_embedding.yaml', 
    'filtering': 'train_filter.yaml',
}

outdir_dict = {
    "build": feature_outdir,
    'embedding': embedding_outdir,
    'filtering': filtering_outdir,
}