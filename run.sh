#!/bin/bash
###########80#COLUMNS#BECAUSE#SOME#PEOPLE#STILL#USE#PUNCHCARDS#I#GUESS##########
#
# Example submit file for batch jobs on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################
#
# command-line options to sbatch can be specified at the top of the batch
# submission file when preceeded by '#SBATCH'. These lines will be
# interpreted by the shell as comments but will be parsed by sbatch.
# These lines must be at the top of the file and may only be preceeded
# by comments and whitespace. See 'man sbatch' for a list of options.
#
# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes. You must use the 'batch' partition
# to submit jobs.
#SBATCH --partition=teaching
# The number of GPUs to request
#SBATCH --gpus=1
# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=8
# Naming
#SBATCH --output=pong_job.out
#SBATCH --job-name=200_200_100_rerun2

# Activate the anaconda environment. Must use this form in scripts.
#. /usr/local/anaconda3/bin/activate
# Your job
pip3 install --user -r requirements.txt
python3 pg.py
# Deactivate the anaconda environment
# conda deactivate
