#!/bin/sh

# Expected number of hours to run one simulation (always overestimate so that slurm doesnt kill the sim)
HOURS_PER_SIM=72
GIGS=16
DATA_DIR=/nas/longleaf/home/djpassey/InterfereBenchmark1.1.1

EXP_RUNNER=/nas/longleaf/home/djpassey/interfere_experiments/experiments/exp13/run.sh

# Arguments for sbatch. Sets the appropriate time limit and directory
FLAGS="--ntasks=1 --mem=${GIGS}G  --cpus-per-task=1 --time=$HOURS_PER_SIM:00:00"

for file in "$DATA_DIR"/*/*.json
do
    args="$FLAGS  $EXP_RUNNER $file $1"
    sbatch $args
done
