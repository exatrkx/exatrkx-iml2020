#!/usr/bin/env python
import os
os.environ['TRKXINPUTDIR'] = '/global/cscratch1/sd/alazar/trackml/data/train_100_events/' # better change to your copy of the dataset.
os.environ['TRKXOUTPUTDIR'] = '../run200' # change to your own directory

import tensorflow as tf
import sonnet as snt
from exatrkx import SegmentClassifier
from exatrkx import graph

gnn_ckpt_dir='/global/cfs/cdirs/m3443/data/lightning_models/gnn'
num_processing_steps_tr = 8
optimizer = snt.optimizers.Adam(0.001)
model = SegmentClassifier()
ckpt_idx = -1

output_dir = gnn_ckpt_dir
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()

in_files = '/global/homes/x/xju/atlas/heptrkx/models/trackml/iml2020_models/1100'
train_files = tf.io.gfile.glob(in_files)
print(train_files)
raw_dataset = tf.data.TFRecordDataset(train_files)
training_dataset = raw_dataset.map(graph.parse_tfrec_function)
inputs, targets = next(training_dataset.take(1).as_numpy_iterator())
with_batch_dim = False
input_signature = [
    graph.specs_from_graphs_tuple(inputs, with_batch_dim),
    #tf.TensorSpec(shape=[], dtype=tf.int32)
]


#@tf.function(input_signature=((GraphsTuple(nodes=tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name=None), edges=tf.TensorSpec(shape=(None, 1), dtype=tf.float32, name=None), receivers=tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None), senders=tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None), globals=tf.TensorSpec(shape=(1, 1), dtype=tf.float32, name=None), n_node=tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None), n_edge=tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None)), tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None))))

out = model(inputs, num_processing_steps_tr)
print(out[-1].edges[0])
print(inputs.n_node)

@tf.function(input_signature=input_signature)
def inference(x):
    return model(x, 8)

#concrete_func = tf.function(inference, input_signature=input_signature)
print(inputs.edges[0])
print(input_signature)
out = inference(inputs)
print(out[-1].edges[0])

model.inference = inference
model.all_variables = list(model.variables)
#out_model_dir = "/global/cfs/cdirs/m3443/data/lightning_models/gnn/saved_model" 
out_model_dir = "/global/homes/x/xju/atlas/heptrkx/models/trackml/iml2020_models/gnn/saved_model"
#tf.saved_model.save(model, out_model_dir, signatures=input_signature)
tf.saved_model.save(model, out_model_dir)
print("model is saved")


# tf2onnx
import tf2onnx
model_proto, _ = tf2onnx.convert.from_function(
    inference,
    input_signature=input_signature, opset=None, custom_ops=None,
    custom_op_handlers=None, custom_rewriter=None,
    inputs_as_nchw=None, extra_opset=None, shape_override=None,
    target=None, large_model=False, output_path="gnn.onnx")
