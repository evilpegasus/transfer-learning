#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J train_fastsim_mlp
#SBATCH --mail-user=mingfong@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -A m3246

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

conda activate jax

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python3 run.py
