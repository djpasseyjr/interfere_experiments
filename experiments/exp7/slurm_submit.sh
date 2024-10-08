#!/bin/sh

# Expected number of hours to run one simulation (always overestimate so that slurm doesnt kill the sim)
HOURS_PER_SIM=120
GIGS=32

# Arguments for sbatch. Sets the appropriate time limit and directory
FLAGS="--ntasks=1 --mem=${GIGS}G  --cpus-per-task=1 --time=$HOURS_PER_SIM:00:00"
# Total number of jobs
sbatch $FLAGS /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp7/run.sh
