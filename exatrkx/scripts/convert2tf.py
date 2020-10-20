#!/usr/bin/env python
import os

import tensorflow as tf
from exatrkx import DoubletsDataset
from exatrkx.src import utils_dir

if __name__ == "__main__":
    inputdir = utils_dir.filtering_outdir
    outdir = utils_dir.gnn_inputs
    data = DoubletsDataset()
    datatypes = ['train', 'val', 'test']
    for datatype in datatypes:
        print("processing files in folder: {}".format(datatype))
        inputdir = os.path.join(inputdir, datatype)
        outname = os.path.join(outdir, datatype)
        if not os.path.exists(outname):
            os.makedirs(outname, exist_ok=True)
        data.process(indir=inputdir, outdir=outname)