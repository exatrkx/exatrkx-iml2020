#!/bin/bash
#SBATCH -J doublets
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH -t 04:00:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

DATADIR=/global/homes/x/xju/work/heptrkx/iml2020/results/iml2020/filter_output/run102
srun -l python scripts/hvd_distributed \
	--train-files "$DATADIR/train/*_*.tfrec" \
	--eval-files "$DATADIR/val/*_*.tfrec" \
	--job-dir "/global/homes/x/xju/work/heptrkx/iml2020/results/iml2020/gnn_output" \
	--train-batch-size 1 \
	--eval-batch-size 1  \
	--num-epochs 1 \
	-d 
