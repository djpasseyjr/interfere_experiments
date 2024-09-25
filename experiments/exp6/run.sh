#!/bin/sh

module purge
module load anaconda
conda activate interfere_exp5

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp6/main.py


