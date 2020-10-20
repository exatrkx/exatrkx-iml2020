# Tracking-ML-Exa.TrkX
Building input graphs for Graph Neural Network (GNN) is based on the embedding learning and filtering with multilayer perceptrons, both of which are implemented in [Pytorch](https://pytorch.org/get-started/locally/). The GNN is implemented in the TensorFlow with the [graph_nets](https://github.com/deepmind/graph_nets) package.

<!-- [Documentation available here.](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/) -->
### Installation

```bash
conda create --name exatrkx python=3.8

pip install --upgrade pip
```
Dependencies not listed in the `setup.py` are tabulated below. We are referring to their webpage for detailed installation instructions.

* [pytorch](https://pytorch.org/get-started/locally/) for embedding learning and filtering
* [torch-geometric](https://github.com/rusty1s/pytorch_geometric#installation) 
* [tensorflow](https://www.tensorflow.org/install) for GNN
* [horovod](https://github.com/horovod/horovod#install) for distributed training