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

feature_outdir   = os.path.join(output_base, "feature_store")
embedding_outdir = os.path.join(output_base, "embedding_output")
filtering_outdir = os.path.join(output_base, "filtering_output")
gnn_inputs       = os.path.join(output_base, "gnn_inputs")
gnn_models       = os.path.join(output_base, "gnn_models")
gnn_output       = os.path.join(output_base, "gnn_output")

outdirs = [feature_outdir, embedding_outdir, filtering_outdir,
        gnn_inputs, gnn_models, gnn_output]

if not os.path.exists(feature_outdir):
    [os.makedirs(x, exist_ok=True) for x in outdirs]


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