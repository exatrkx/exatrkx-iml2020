#!/usr/bin/env python
import os

import tensorflow as tf
from exatrkx import DoubletsDataset
from exatrkx.src import utils_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert to TF records")
    add_arg = parser.add_argument
    add_arg("--num-workers", help="number of threads", default=1)
    add_arg("--input-dir", help='input directory if not predefined one', default=None)
    add_arg("--output-dir", help='output directory if not predefined one', default=None)
    add_arg("--no-category", help="no subfolders train,val,test", action='store_true')
    args = parser.parse_args()
    
    inputdir = utils_dir.filtering_outdir if args.input_dir is None else args.input_dir
    outdir = utils_dir.gnn_inputs if args.output_dir is None else args.output_dir

    data = DoubletsDataset(num_workers=args.num_workers)
    if args.no_category:
        data.process(indir=inputdir, outdir=outdir)
        exit(0)
    else:    
        datatypes = ['train', 'val', 'test']
        for datatype in datatypes:
            indir = os.path.join(inputdir, datatype)
            outname = os.path.join(outdir, datatype)
            print("processing files in folder: {}".format(indir))
            if not os.path.exists(outname):
                os.makedirs(outname, exist_ok=True)
            data.process(indir=indir, outdir=outname)