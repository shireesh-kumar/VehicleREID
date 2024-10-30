#!/bin/bash
#SBATCH -J TrainDC
#SBATCH -o TrainDC.o%j
#SBATCH --mail-user=sporalas@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=8 -N 2
#SBATCH -t 12:0:0
#SBATCH --gpus=volta:4
#SBATCH --mem=16GB

# Set the environment variable to manage CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_dc.py

