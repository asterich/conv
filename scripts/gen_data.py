# Usage: python gen_data.py --dtype <data_type> --input_size <input_size> --kernel_size <kernel_size> --output_name <output_name> --output_dir <output_dir>
# data_type: int or float
# input_size: size of input matrix
# kernel_size: size of kernel matrix
# output_name: name of output files
# output_dir: directory to output file

# for example, command "python gen_data.py --dtype int --input_size 10 --kernel_size 3 --output_name in_1024_kern_3 --output_dir data"
# will generate 2 files in data directory: in_1024_kern_3.in, in_1024_kern_3.ans
# in_1024_kern_3.in contains input matrix and kernel matrix, in_1024_kern_3.ans contains output matrix


# data format: 1 matrix per line
# e.g.
# 1 2 3 4 5 6 7 8 9
# 0 0 0 1


import argparse
import numpy as np
import os

def gen_data(input_size, kernel_size, dtype):
    input_data = None
    kernel_data = None
    output_data = None

    if dtype == 'int':
        input_data = np.random.randint(0, 10, (input_size, input_size))
        kernel_data = np.random.randint(0, 10, (kernel_size, kernel_size))
        output_data = np.zeros((input_size - kernel_size + 1, input_size - kernel_size + 1), dtype=dtype)
    elif dtype == 'float':
        input_data = np.random.rand(input_size, input_size)
        kernel_data = np.random.rand(kernel_size, kernel_size)
        output_data = np.zeros((input_size - kernel_size + 1, input_size - kernel_size + 1), dtype=dtype)
    else:
        raise ValueError(f'Invalid data type: {dtype}')
    
    print("input matrix and kernel generated")

    for i in range(input_size - kernel_size + 1):
        for j in range(input_size - kernel_size + 1):
            output_data[i, j] = np.sum(input_data[i:i+kernel_size, j:j+kernel_size] * kernel_data)

    print("output matrix generated")

    return input_data, kernel_data, output_data

def output_in_mat_and_kernel_mat(input_data, kernel_data, output_dir, output_name):
    filepath = os.path.join(output_dir, f'{output_name}.in')
    print(f'outputting input matrix and kernel matrix to {filepath}')
    with open(filepath, 'w') as in_file:
        for i in range(input_data.shape[0]):
            print(' '.join(map(str, input_data[i])), file=in_file, end=' ')
        print(file=in_file)
        for i in range(kernel_data.shape[0]):
            print(' '.join(map(str, kernel_data[i])), file=in_file, end=' ')
        print(file=in_file)

def output_output_mat(output_data, output_dir, output_name):
    filepath = os.path.join(output_dir, f'{output_name}.ans')
    print(f'outputting output matrix to {filepath}')
    with open(filepath, 'w') as out_file:
        for i in range(output_data.shape[0]):
            print(' '.join(map(str, output_data[i])), file=out_file, end=' ')
        print(file=out_file)


parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, default='int')
parser.add_argument('--input_size', type=int, required=True)
parser.add_argument('--kernel_size', type=int, required=True)
parser.add_argument('--output_name', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

input_data, kernel_data, output_data = gen_data(args.input_size, args.kernel_size, args.dtype)
assert input_data is not None and kernel_data is not None and output_data is not None
output_in_mat_and_kernel_mat(input_data, kernel_data, args.output_dir, args.output_name)
output_output_mat(output_data, args.output_dir, args.output_name)

    