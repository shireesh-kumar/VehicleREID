#!/bin/bash
#SBATCH -J TrainResNet50ColorRGBAtt
#SBATCH -o TrainResNet50TypeColorRGB.o%j
#SBATCH --mail-user=sporalas@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=4 -N 2
#SBATCH -t 1:0:0
#SBATCH --gpus=volta:2
#SBATCH --mem-per-cpu=16GB

python train_color.py

