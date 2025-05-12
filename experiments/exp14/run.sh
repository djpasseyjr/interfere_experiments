#!/bin/sh
DATA_FILE=$1
METHODS=$2

module purge
module load anaconda
conda activate InterfereExp14

pip list > /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp14/requirements.txt

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp14/main.py $DATA_FILE $METHODS
