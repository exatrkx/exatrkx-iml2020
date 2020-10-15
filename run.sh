#!/bin/bash


DATADIR=/global/homes/x/xju/work/heptrkx/iml2020/results/iml2020/tfrec
python hvd_distributed.py \
	--train-files "$DATADIR/train/*_*.tfrec" \
	--eval-files "$DATADIR/val/*_*.tfrec" \
	--job-dir "/global/homes/x/xju/work/heptrkx/iml2020/results/iml2020/gnn_output/run100" \
	--train-batch-size 1 \
	--eval-batch-size 1  \
	--num-epochs 1 
