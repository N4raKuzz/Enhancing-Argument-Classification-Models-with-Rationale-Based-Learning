#!/bin/bash

#SBATCH --nodes=5               
#SBATCH --mem=20000               
#SBATCH --job-name=PythonTrain   
#SBATCH --time=00:30:00  
#SBATCH --output=run1.out
#SBATCH --error=run1.err    

python train.py
squeue -lu zz22u21
