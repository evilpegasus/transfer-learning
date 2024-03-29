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
#   --dataload_method=all --epochs=200 \
#   --num_files=1 --max_train_rows=20000 --max_val_rows=20000 \
#   --learning_rate=0.00001 --seed=1 --dnn_layers=400,400,400,400,400,1 \
#   --train_dir=/pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ \
#   --wandb_project=fullsim --wandb_run_name=TESTING \
#   --checkpoint_interval=10 \
#   --wandb_run_path=mingfong/delphes_pretrain/2yn9eeqz --resume_training=False



# train fullsim
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python3 run.py \
  --dataload_method=all --epochs=400 \
  --num_files=1 --max_train_rows=8192000 --max_val_rows=5242880 \
  --learning_rate=0.00001 --seed=5 --dnn_layers=400,400,400,400,400,1 \
  --train_dir=/pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ \
  --wandb_project=fullsim --wandb_run_name="fullsim_only5 4M" \
  --checkpoint_interval=5 \
  # --wandb_run_path=mingfong/fullsim/25ao9xre --resume_training=True

# transfered weights
# row amounts: 2048000 4096000 8192000 16384000
# seeds 1 2 3 4 5
# wandb ids: 2yn9eeqz 0dgphck6 3w14krnd qz0apmr2 xrquaso0
# srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 python3 run.py \
#   --dataload_method=all --epochs=200 \
#   --num_files=1 --max_train_rows=16384000 --max_val_rows=5242880 \
#   --learning_rate=0.00001 --seed=5 --dnn_layers=400,400,400,400,400,1 \
#   --train_dir=/pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ \
#   --wandb_project=fullsim --wandb_run_name="fullsim_transfered5 16M rows" \
#   --checkpoint_interval=10 \
#   --wandb_run_path=mingfong/delphes_pretrain/xrquaso0 --resume_training=False
