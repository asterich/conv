export OMP_PLACES=cores
export OMP_NUM_THREADS=64
export MKLDNN_JIT_DUMP=1

. /data/spack/share/spack/setup-env.sh
spack load intel-oneapi-compilers@2024.0.2

MKLDNN_JIT_DUMP=1 xmake run conv -i data/in_32768_kern_3.in -a data/in_32768_kern_3.ans

# build/linux/x86_64/release/conv \
#     -i data/in_32768_kern_3.in -a data/in_32768_kern_3.ans