#!/bin/sh
DATA_FILE=$1

module purge
module load anaconda
conda activate interfere_exp12

pip list > /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp11/requirements.txt

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp11/main.py $DATA_FILE