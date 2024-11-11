#!/bin/sh

module purge
module load anaconda
conda activate interfere_exp9

pip list > /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp9/requirements.txt

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp9/main.py