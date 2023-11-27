#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J Preprocess
#SBATCH --mail-user=mingfong@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 08:01:01
#SBATCH --account=m3246
#SBATCH --qos=regular

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

conda activate jax

#run the application:
# srun -n 1 -c 256 --cpu_bind=cores python3 /global/homes/m/mingfong/git/transfer-learning/data_utils.py

# fastsim train dataset
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_0.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_1.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_2.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_3.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_4.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_5.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_6.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_7.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_8.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_9.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_10.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_11.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_12.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_13.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_14.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ &


# fastsim test dataset
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_test_set/test_0.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_test_processed/ &
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_test_set/test_1.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_test_processed/ &

# fullsim train
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/train.h5 /pscratch/sd/m/mingfong/transfer-learning/fullsim_train_processed/ &

# fullsim test
# srun --exclusive -n 1 -c 32 --cpu_bind=cores python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/test.h5 /pscratch/sd/m/mingfong/transfer-learning/fullsim_test_processed/ &

wait
