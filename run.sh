#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=11
#SBATCH --output=pong_job.out
#SBATCH --job-name=sym_large_sync

# Your job
#python3 -m pip3 install --user -r requirements.txt
singularity run --nv /data/containers/msoe-tensorflow.sif python3 reinforcement.py
