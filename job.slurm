#!/bin/bash
#SBATCH --job-name=conv
#SBATCH --output=output/conv_%j.out
#SBATCH --error=output/conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

xmake run conv -i data/in_16384_kern_9.in -a data/in_16384_kern_9.ans