#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J sat_v_10           # Job name
#SBATCH -o sat_v_10%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 1.5 hours


# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.

module load cuda
source activate py36
source $WORK/.bashrc 
python create_input_files.py
python train.py
