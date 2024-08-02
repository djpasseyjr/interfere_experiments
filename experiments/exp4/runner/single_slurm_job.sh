#!/bin/sh

module purge
module load anaconda
conda activate interfere_exp4

python /nas/longleaf/home/djpassey/interfere/experiments/exp4/runner/run.py $1
