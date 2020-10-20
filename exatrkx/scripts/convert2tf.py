#!/usr/bin/env python

import tensorflow as tf
from tfgraphs.dataset_base import DoubletsDataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="create a graph from filtering output")
    add_arg = parser.add_argument
    add_arg("inputdir", help="input directory")
    add_arg("outname", help='output name')
    
    args = parser.parse_args()
    data = DoubletsDataset()
    data.process(indir=args.inputdir, outdir=args.outname)