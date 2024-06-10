## Build

```sh
xmake build
```

## Run

use

```sh
xmake run
```

It will get input/kernel matrix from stdin

or

```sh
xmake run -i <input_file> -a <ans_file>
```

The latter will get input/kernel matrix from \<input_file\> and verify result with \<ans_file\>

## Data generation

Usage: `python scripts/gen_data.py --dtype <data_type> --input_size <input_size> --kernel_size <kernel_size> --output_name <output_name> --output_dir <output_dir>`

- data_type: int or float
- input_size: size of input matrix
- kernel_size: size of kernel matrix
- output_name: name of output files
- output_dir: directory to output file

For example, command `python gen_data.py --dtype int --input_size 1024 --kernel_size 3 --output_name in_1024_kern_3 --output_dir data` will generate 2 files in data directory: in_1024_kern_3.in, in_1024_kern_3.ans

in_1024_kern_3.in contains input matrix and kernel matrix, in_1024_kern_3.ans contains output matrix