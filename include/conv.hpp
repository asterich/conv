#ifndef __CONV_HPP__
#define __CONV_HPP__

#include <iostream>
#include <omp.h>

template <typename T>
void conv2d_naive(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i * out_size + j] = 0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[i * out_size + j] += 
                        input[(i + k) * in_size + (j + l)] * kernel[k * kernel_size + l];
                }
            }
        }
    }
}

template <typename T>
void conv2d_omp(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i * out_size + j] = 0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[i * out_size + j] += 
                        input[(i + k) * in_size + (j + l)] * kernel[k * kernel_size + l];
                }
            }
        }
    }
}

template <typename T>
void im2col_openmp(T *input, T *output, int in_size, int out_size, int kernel_size) {
    #pragma omp parallel for
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[(i * out_size + j) * kernel_size * kernel_size + k * kernel_size + l] = 
                        input[(i + k) * in_size + (j + l)];
                }
            }
        }
    }
}

template <typename T>
void spmv_openmp(T *data, T *output, T *vec, int row_size, int vec_size) {
    #pragma omp parallel for
    for (int i = 0; i < row_size; i++) {
        output[i] = 0;
        for (int j = 0; j < vec_size; j++) {
            output[i] += data[i * vec_size + j] * vec[j];
        }
    }
}

// bad_alloc will happen with large input size
template <typename T>
void conv2d_with_im2col(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    T *im2col_input = new T[out_size * out_size * kernel_size * kernel_size];
    im2col_openmp(input, im2col_input, in_size, out_size, kernel_size);
    spmv_openmp(im2col_input, output, kernel, out_size * out_size, kernel_size * kernel_size);
    delete[] im2col_input;
}

template <typename T>
void conv2d_with_im2col_openmp(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {

}


#endif