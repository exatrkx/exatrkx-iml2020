#!/usr/bin/env python

import tensorflow as tf
from tfgraphs.doublets import DoubletsDataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="create a graph")
    add_arg = parser.add_argument
    add_arg("inputdir", help="input directory")
    add_arg("outname", help='output name')
    add_arg('--evts-per-record', default=10, type=int, help='number of events per output file')
    add_arg("--max-evts", type=int, default=-1, help='maximum number of events to process')
    add_arg("--debug", action='store_true', help='in a debug mode')

    
    args = parser.parse_args()
    data = DoubletsDataset()
    data.process(filename=args.inputdir, save=True, outname=args.outname,\
            n_evts_per_record=args.evts_per_record, debug=args.debug,
            max_evts=args.max_evts)