#!/bin/bash

# Build the project
echo "Building the project..."

# import intel compiler with spack
spack load intel-oneapi-compilers
which icx
spack load intel-oneapi-vtune

# compile the code
xmake clean
xmake config --toolchain=intel-oneapi -c -p linux -a x86_64
xmake project -k compile_commands
xmake build --rebuild --verbose