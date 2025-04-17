#!/bin/sh
DATA_FILE=$1
METHODS=$2

module purge
module load anaconda
conda activate InterfereExp13

pip list > /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp13/requirements.txt

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp13/main.py $DATA_FILE $METHODS
