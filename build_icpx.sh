#!/bin/bash

# Build the project
echo "Building the project..."

# import intel compiler with spack
spack load intel-oneapi-compilers
which icx

# compile the code
xmake clean
xmake config --toolchain=intel-oneapi -c -p linux -a x86_64
xmake build --rebuild --verbose