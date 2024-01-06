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
free -h
nvidia-smi

# train fastsim
# GPU nodes can only go up to --dataload_method=all --num_files=7 before OOM error
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python3 run.py \
  --dataload_method=all --epochs=400 \
  --num_files=7 \
  --learning_rate=0.00001 --seed=2 --dnn_layers=400,400,400,400,400,1 \
  --train_dir=/pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ \
  --wandb_project=delphes_pretrain --wandb_run_name=pretrain \
  --checkpoint_interval=10 \
  --wandb_run_path=mingfong/delphes_pretrain/0dgphck6 --resume_training=True
