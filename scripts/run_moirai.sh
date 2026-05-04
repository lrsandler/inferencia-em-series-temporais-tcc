#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moirai
cd ..
python main.py --model moirai
