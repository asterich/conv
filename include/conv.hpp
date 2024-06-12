#ifndef __CONV_HPP__
#define __CONV_HPP__

#include "oneapi/dnnl/dnnl.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <omp.h>
#include <mkl.h>
#include <dnnl.hpp>

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
void __attribute__((noinline)) conv2d_omp(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
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
    constexpr int stride = 1;
    constexpr int pad = 0;
    
    size_t out_h = (in_size - kernel_size + 2 * pad) / stride + 1;
    size_t out_w = (in_size - kernel_size + 2 * pad) / stride + 1;
    size_t kernel_size_sq = kernel_size * kernel_size;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < out_h; ++i) {    
        for (size_t c = 0; c < kernel_size; ++c) {
            size_t kernel_idx = c * kernel_size;
            size_t input_row = i + c;
            for (size_t j = 0; j < out_w; ++j) {
                for (size_t d = 0; d < kernel_size; ++d) {
                    size_t input_col = j + d;
                    size_t output_idx = (i * out_w + j) * kernel_size_sq + kernel_idx + d;
                    output[output_idx] = input[input_row * in_size + input_col];
                }
            }
        }
    }

}

template <typename T>
void spmv_openmp(T *data, T *output, T *vec, int row_size, int vec_size) {
    #pragma omp parallel for
    for (size_t i = 0; i < row_size; i++) {
        output[i] = 0;
        for (size_t j = 0; j < vec_size; j++) {
            output[i] += data[i * vec_size + j] * vec[j];
        }
    }
}

template <typename T>
void __attribute__((noinline)) conv2d_with_im2col(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    T *im2col_input = (T *)calloc((size_t)out_size * out_size * kernel_size * kernel_size, sizeof(T));
    if (im2col_input == nullptr) {
        std::cout << "bad_alloc" << std::endl;
    }
    im2col_openmp(input, im2col_input, in_size, out_size, kernel_size);
    // spmv_openmp(im2col_input, output, kernel, out_size * out_size, kernel_size * kernel_size);
    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_size * out_size, kernel_size * kernel_size, 1.0, im2col_input, kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
    } else if constexpr (std::is_same<T, double>::value) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, out_size * out_size, kernel_size * kernel_size, 1.0, im2col_input, kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
    } else {
        spmv_openmp(im2col_input, output, kernel, out_size * out_size, kernel_size * kernel_size);
    }
    if (im2col_input != nullptr) {
        free(im2col_input);
    }
}

template <typename T>
void conv2d_mkl_dnn(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    dnnl::memory::dims input_dims = {1, 1, in_size, in_size};
    dnnl::memory::dims kernel_dims = {1, 1, kernel_size, kernel_size};
    dnnl::memory::dims output_dims = {1, 1, out_size, out_size};
    dnnl::memory::dims strides = {1, 1};
    dnnl::memory::dims padding = {0, 0};
    dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    dnnl::memory::desc kernel_md(kernel_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    dnnl::memory::desc output_md(output_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    dnnl::memory input_mem(input_md, eng, input);
    dnnl::memory kernel_mem(kernel_md, eng, kernel);
    dnnl::memory output_mem(output_md, eng, output);
    dnnl::convolution_forward::primitive_desc conv_pd(eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, input_md, kernel_md, output_md, strides, padding, padding);
    dnnl::convolution_forward(conv_pd).execute(s, {{DNNL_ARG_SRC, input_mem}, {DNNL_ARG_WEIGHTS, kernel_mem}, {DNNL_ARG_DST, output_mem}});
    s.wait();
}


#endif