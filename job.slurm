#!/bin/bash
#SBATCH --job-name=conv
#SBATCH --output=output/conv_%j.out
#SBATCH --error=output/conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# OMP_NUM_THREADS=64 xmake run conv -i data/in_32768_kern_9.in -a data/in_32768_kern_9.ans

spack load intel-oneapi-vtune

OMP_NUM_THREADS=64 vtune -collect hotspots build/linux/x86_64/release/conv \
    -i data/in_32768_kern_9.in -a data/in_32768_kern_9.ans