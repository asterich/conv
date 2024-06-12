#!/bin/bash
#SBATCH --job-name=conv
#SBATCH --output=output/conv_%j.out
#SBATCH --error=output/conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_NUM_THREADS=64

spack load intel-oneapi-compilers

xmake run conv -i data/in_16384_kern_9.in -a data/in_16384_kern_9.ans

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/asterich/.xmake/packages/o/onednn/v3.4.3/3234cd7eee27411bbcfe7dcb4d24ef2b/lib
# spack load intel-oneapi-vtune
# vtune -collect hotspot ./build/linux/x86_64/release/conv \
#     -i data/in_16384_kern_9.in -a data/in_16384_kern_9.ans