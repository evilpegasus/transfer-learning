#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J train_fullsim_mlp
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
free -h
nvidia-smi

# small testing run command for login node
# python3 run.py \
#   --dataload_method=all --epochs=20 \
#   --num_files=1 --max_train_rows=50000 --max_val_rows=50000 \
#   --learning_rate=0.00001 --seed=8 --dnn_layers=400,400,400,400,400,1 \
#   --train_dir=/pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ \
#   --wandb_project=fullsim

# train fullsim
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python3 run.py \
  --dataload_method=all --epochs=400 \
  --num_files=1 --max_train_rows=16384000 --max_val_rows=5242880 \
  --learning_rate=0.00001 --seed=8 --dnn_layers=400,400,400,400,400,1 \
  --train_dir=/pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ \
  --wandb_project=fullsim
