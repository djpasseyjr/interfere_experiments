#!/bin/sh

module purge
module load anaconda
conda activate interfere_exp5

pip uninstall interfere -y
pip install git+https://www.github.com/djpasseyjr/interfere.git@6b600cafefbbc5771efeecc817da5d50fe8e9ca9
pip install git+https://www.github.com/djpasseyjr/interfere_experiments.git

python /nas/longleaf/home/djpassey/interfere_experiments/experiments/exp5/main.py


