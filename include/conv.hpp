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
#include <immintrin.h>

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
    /* data format: kh * kw * oh * ow */
    constexpr int stride = 1;
    constexpr int pad = 0;
    
    size_t out_h = (in_size - kernel_size + 2 * pad) / stride + 1;
    size_t out_w = (in_size - kernel_size + 2 * pad) / stride + 1;
    size_t kernel_size_sq = kernel_size * kernel_size;

    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < kernel_size_sq; c++) {
        size_t w_offset = c % kernel_size;
        size_t h_offset = (c / kernel_size) % kernel_size;
        size_t c_im = c / kernel_size_sq;
        for (size_t h = 0; h < out_h; h++) {
            for (size_t w = 0; w < out_w; w++) {
                size_t im_row = h_offset + h * stride;
                size_t im_col = w_offset + w * stride;
                output[c * out_h * out_w + h * out_w + w] = input[im_row * in_size + im_col];
            }
        }
    }

}

template <typename T>
void __attribute__((noinline)) conv2d_with_im2col(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    T *im2col_input = (T *)calloc((size_t)kernel_size * kernel_size * out_size * out_size, sizeof(T));
    if (im2col_input == nullptr) {
        std::cout << "bad_alloc" << std::endl;
    }

    im2col_openmp(input, im2col_input, in_size, out_size, kernel_size);

    /* im2col output data format: kh * kw * oh * ow */
    /* kernel data format: kh * kw, no channels */
    /* output data format: oh * ow */
    /* use blas */
    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_size * out_size, 1, kernel_size * kernel_size, 1.0, im2col_input, kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
    } else  if constexpr (std::is_same<T, double>::value) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_size * out_size, 1, kernel_size * kernel_size, 1.0, im2col_input, kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
    } else {
        // fallback
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < out_size; j++) {
                output[i * out_size + j] = 0;
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        output[i * out_size + j] += 
                            im2col_input[(k * kernel_size + l) * out_size * out_size + i * out_size + j] * kernel[k * kernel_size + l];
                    }
                }
            }
        }
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

template <typename T>
T conv2d_direct_omp_blocking_kernel_3x3_avx(T *input1, T *input2, T *input3, T *kernel);

template <>
inline float conv2d_direct_omp_blocking_kernel_3x3_avx<float>(float *input1, float *input2, float *input3, float *kernel) {
    float sum = input1[0] * kernel[0];

    __m256 input_v = _mm256_setr_ps(input1[1], input1[2], input2[0], input2[1], input2[2], input3[0], input3[1], input3[2]);
    __m256 kernel_v = _mm256_loadu_ps(kernel + 1);
    __m256 sum_v = _mm256_mul_ps(input_v, kernel_v);
    sum += sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3] + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];

    return sum;
}

template <>
inline double conv2d_direct_omp_blocking_kernel_3x3_avx<double>(double *input1, double *input2, double *input3, double *kernel) {
    double sum = input1[0] * kernel[0];

    __m512d input_v = _mm512_setr_pd(input1[1], input1[2], input2[0], input2[1], input2[2], input3[0], input3[1], input3[2]);
    __m512d kernel_v = _mm512_loadu_pd(kernel + 1);
    __m512d sum_v = _mm512_mul_pd(input_v, kernel_v);
    
    sum += sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3] + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];

    return sum;
}

template <typename T>
void conv2d_direct_omp_blocking(T *input, T *output, T *kernel, int in_size, int out_size, int kernel_size) {
    int blockSize = 8;
    
    if (kernel_size == 3) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < out_size; i += blockSize) {
            for (int j = 0; j < out_size; j += blockSize) {
                for (int bi = i; bi < std::min(i + blockSize, out_size); ++bi) {
                    for (int bj = j; bj < std::min(j + blockSize, out_size); ++bj) {
                        output[bi * out_size + bj] = conv2d_direct_omp_blocking_kernel_3x3_avx(
                            input + bi * in_size + bj,
                            input + (bi + 1) * in_size + bj,
                            input + (bi + 2) * in_size + bj,
                            kernel
                        );
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < out_size; i += blockSize) {
            for (int j = 0; j < out_size; j += blockSize) {
                for (int bi = i; bi < std::min(i + blockSize, out_size); ++bi) {
                    for (int bj = j; bj < std::min(j + blockSize, out_size); ++bj) {
                        T sum = 0.0;
                        
                        for (int ki = 0; ki < kernel_size; ++ki) {
                            for (int kj = 0; kj < kernel_size; ++kj) {
                                sum += input[(bi + ki) * in_size + (bj + kj)] * kernel[ki * kernel_size + kj];
                            }
                        }
                        output[bi * out_size + bj] = sum;
                    }
                }
            }
        }
    }
    
}




#endif