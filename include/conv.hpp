#ifndef __CONV_HPP__
#define __CONV_HPP__

#include "oneapi/dnnl/dnnl.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dnnl.hpp>
#include <exception>
#include <iostream>
#include <mkl.h>
#include <numeric>
#include <omp.h>

using namespace dnnl;

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
                         std::multiplies<dnnl::memory::dim>());
}

inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle)
    throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
  bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                      eng.get_kind() == dnnl::engine::kind::cpu);
  bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL &&
                      eng.get_kind() == dnnl::engine::kind::gpu);
  if (is_cpu_sycl || is_gpu_sycl) {
    auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
    if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
      auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
      auto dst = buffer.get_host_access();
      uint8_t *dst_ptr = dst.get_pointer();
      if (!dst_ptr)
        throw std::runtime_error("get_pointer returned nullptr.");
      for (size_t i = 0; i < size; ++i)
        dst_ptr[i] = ((uint8_t *)handle)[i];
    } else {
      assert(mkind == dnnl::sycl_interop::memory_kind::usm);
      uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
      if (!dst_ptr)
        throw std::runtime_error("get_data_handle returned nullptr.");
      if (is_cpu_sycl) {
        for (size_t i = 0; i < size; ++i)
          dst_ptr[i] = ((uint8_t *)handle)[i];
      } else {
        auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
        sycl_queue.memcpy(dst_ptr, handle, size).wait();
      }
    }
    return;
  }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    void *mapped_ptr = mem.map_data();
    if (mapped_ptr)
      std::memcpy(mapped_ptr, handle, size);
    mem.unmap_data(mapped_ptr);
    return;
  }
#endif

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst)
      throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *)handle)[i];
    return;
  }

  // assert(!"not expected");
}

template <typename T>
void conv2d_naive(T *input, T *output, T *kernel, int in_size, int out_size,
                  int kernel_size) {
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
void __attribute__((noinline))
conv2d_omp(T *input, T *output, T *kernel, int in_size, int out_size,
           int kernel_size) {
  std::cout << "num threads: " << omp_get_max_threads() << std::endl;

  const int block_size = 16; // 选择一个合适的block size

#pragma omp parallel for collapse(2)
  for (int ii = 0; ii < out_size; ii += block_size) {
    for (int jj = 0; jj < out_size; jj += block_size) {
      for (int i = ii; i < ii + block_size && i < out_size; ++i) {
        for (int j = jj; j < jj + block_size && j < out_size; ++j) {
          T sum = 0;
          for (int k = 0; k < kernel_size; ++k) {
            for (int l = 0; l < kernel_size; ++l) {
              sum += input[(i + k) * in_size + (j + l)] *
                     kernel[k * kernel_size + l];
            }
          }
          output[i * out_size + j] = sum;
        }
      }
    }
  }
}

template <typename T>
void im2col_openmp(T *input, T *output, int in_size, int out_size,
                   int kernel_size) {
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
void __attribute__((noinline))
conv2d_with_im2col(T *input, T *output, T *kernel, int in_size, int out_size,
                   int kernel_size) {
  T *im2col_input = (T *)calloc(
      (size_t)out_size * out_size * kernel_size * kernel_size, sizeof(T));
  if (im2col_input == nullptr) {
    std::cout << "bad_alloc" << std::endl;
  }
  im2col_openmp(input, im2col_input, in_size, out_size, kernel_size);
  // spmv_openmp(im2col_input, output, kernel, out_size * out_size, kernel_size
  // * kernel_size);
  if constexpr (std::is_same<T, float>::value) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_size * out_size,
                kernel_size * kernel_size, 1.0, im2col_input,
                kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
  } else if constexpr (std::is_same<T, double>::value) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, out_size * out_size,
                kernel_size * kernel_size, 1.0, im2col_input,
                kernel_size * kernel_size, kernel, 1, 0.0, output, 1);
  } else {
    spmv_openmp(im2col_input, output, kernel, out_size * out_size,
                kernel_size * kernel_size);
  }
  if (im2col_input != nullptr) {
    free(im2col_input);
  }
}

template <typename T>
void conv2d_mkl_dnn(T *input, T *output, T *kernel, int in_size, int out_size,
                    int kernel_size) {
  engine eng(dnnl::engine::kind::cpu, 0);
  stream s(eng);
  memory::dims input_dims = {1, 1, in_size, in_size};
  memory::dims kernel_dims = {1, 1, kernel_size, kernel_size};
  dnnl::memory::dims output_dims = {1, 1, out_size, out_size};
  dnnl::memory::dims strides = {1, 1};
  dnnl::memory::dims padding = {0, 0};
  dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32,
                              dnnl::memory::format_tag::nchw);
  dnnl::memory::desc kernel_md(kernel_dims, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nchw);
  dnnl::memory::desc output_md(output_dims, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nchw);
  dnnl::memory input_mem(input_md, eng, input);
  dnnl::memory kernel_mem(kernel_md, eng, kernel);
  dnnl::memory output_mem(output_md, eng, output);
  dnnl::convolution_forward::primitive_desc conv_pd(
      eng, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::convolution_direct, input_md, kernel_md, output_md,
      strides, padding, padding);
  dnnl::convolution_forward(conv_pd).execute(s, {{DNNL_ARG_SRC, input_mem},
                                                 {DNNL_ARG_WEIGHTS, kernel_mem},
                                                 {DNNL_ARG_DST, output_mem}});
  s.wait();
}

template <typename T>
void conv2d_mkl_dnn_opt(T *input, T *output, T *kernel, int in_size,
                        int out_size, int kernel_size) {
  engine eng(dnnl::engine::kind::cpu, 0);

  using tag = memory::format_tag;
  using dt = memory::data_type;

  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = 1;

  // dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32,
  //                             dnnl::memory::format_tag::nchw);
  // dnnl::memory::desc kernel_md(kernel_dims, dnnl::memory::data_type::f32,
  //                              dnnl::memory::format_tag::nchw);
  // dnnl::memory::desc output_md(output_dims, dnnl::memory::data_type::f32,
  //                              dnnl::memory::format_tag::nchw);
  // dnnl::memory input_mem(input_md, eng, input);
  // dnnl::memory kernel_mem(kernel_md, eng, kernel);
  // dnnl::memory output_mem(output_md, eng, output);

  memory::dims conv1_src_tz = {1, 1, in_size, in_size};
  memory::dims conv1_weights_tz = {1, 1, kernel_size, kernel_size};
  memory::dims conv1_dst_tz = {1, 1, out_size, out_size};
  memory::dims conv1_strides = {1, 1};
  memory::dims conv1_padding = {0, 0};

  auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
  // write_to_dnnl_memory(input, user_src_memory);
  auto user_weights_memory =
      memory({{conv1_weights_tz}, dt::f32, tag::nchw}, eng);
  // write_to_dnnl_memory(kernel, user_weights_memory);

  auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
  auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
  auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);

  auto conv1_prim_desc = convolution_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm::convolution_direct,
      conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
      conv1_padding, conv1_padding);

  auto conv1_src_memory = user_src_memory;
  if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
    conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, conv1_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, user_src_memory}, {DNNL_ARG_TO, conv1_src_memory}});
  }

  auto conv1_weights_memory = user_weights_memory;
  if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, conv1_weights_memory)
        .execute(s, user_weights_memory, conv1_weights_memory);
  }
  //[Reorder data and weights]

  /// Create a memory primitive for output.
  /// @snippet cnn_inference_f32.cpp Create memory for output
  //[Create memory for output]
  auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
  //[Create memory for output]

  /// Create a convolution primitive and add it to the net.
  /// @snippet cnn_inference_f32.cpp Create memory for output
  //[Create convolution primitive]
  net.push_back(convolution_forward(conv1_prim_desc));
  net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
                      {DNNL_ARG_WEIGHTS, conv1_weights_memory},
                      {DNNL_ARG_DST, conv1_dst_memory}});
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(s, net_args.at(i));

  s.wait();
}
#endif