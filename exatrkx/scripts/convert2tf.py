#!/usr/bin/env python
import os

import tensorflow as tf
from exatrkx import DoubletsDataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="create a graph from filtering output")
    add_arg = parser.add_argument
    add_arg("inputdir", help="input directory")
    add_arg("outname", help='output name')
    
    args = parser.parse_args()
    data = DoubletsDataset()
    datatypes = ['train', 'val', 'test']
    for datatype in datatypes:
        print("processing files in folder: {}".format(datatype))
        inputdir = os.path.join(args.inputdir, datatype)
        outname = os.path.join(args.outname, datatype)
        if not os.path.exists(outname):
            os.makedirs(outname, exist_ok=True)
        data.process(indir=inputdir, outdir=outname)