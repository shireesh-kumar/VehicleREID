#!/bin/bash
#SBATCH -J TrainDC
#SBATCH -o TrainDC.o%j
#SBATCH --mail-user=sporalas@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=4 -N 1
#SBATCH -t 1:0:0
#SBATCH --gpus=volta:1
#SBATCH --mem-per-cpu=16GB

python train_dc.py

