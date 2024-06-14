#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <istream>
#include <ittnotify.h>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "conv.hpp"

using namespace std;

class Timer {
public:
  Timer() : start_(std::chrono::high_resolution_clock::now()) {}
  Timer(const std::string &name) : Timer() { name_ = name; }

  void reset() { start_ = std::chrono::high_resolution_clock::now(); }

  double elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::high_resolution_clock::now() - start_)
        .count();
  }

  ~Timer() {
    cout << "Timer " << name_ << ":" << endl;
    cout << "Time elapsed: " << elapsed() << "s" << endl;
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};

template <typename T> struct Matrix2D {
  vector<T> data;
  int size;

  Matrix2D(int size) : size(size) { data.resize(size * size); }

  /* input from an istream */
  Matrix2D(istream &in) {
    /*
     * data format:
     * one matrix per line, no size info
     * e.g.
     * 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
     */
    string line;
    getline(in, line);
    T val;
    stringstream ss(line);
    while (ss >> val) {
      data.push_back(val);
    }
    size = round(sqrt(data.size()));
  }

  friend bool operator==(const Matrix2D<T> &lhs, const Matrix2D<T> &rhs) {
    static constexpr double EPS = 1e-6;
    if (lhs.size != rhs.size) {
      return false;
    }
    for (int i = 0; i < lhs.size; i++) {
      for (int j = 0; j < lhs.size; j++) {
        if (fabs(lhs.data[i * lhs.size + j] - rhs.data[i * lhs.size + j]) >
            EPS) {
          return false;
        }
      }
    }
    return true;
  }

  void print() {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cout << data[i * size + j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
};

int main(int argc, char *argv[]) {
  __itt_pause();

  /* Usage: ./conv -i <input_file> -a <ans_file> */
  istream *input_stream = nullptr;
  ifstream ifs;
  ifstream ans_ifs;

  int opt;
  string input_file;
  string ans_file;
  while ((opt = getopt(argc, argv, "i:a:")) != -1) {
    switch (opt) {
    case 'i':
      input_file = string(optarg);
      ifs.open(input_file);
      if (!ifs.is_open()) {
        cerr << "Error: cannot open file " << input_file << endl;
        return 1;
      }
      input_stream = &ifs;
      break;
    case 'a':
      ans_file = string(optarg);
      ans_ifs.open(ans_file);
      if (!ans_ifs.is_open()) {
        cerr << "Error: cannot open file " << ans_file << endl;
        return 1;
      }
      break;
    default:
      cerr << "Usage: ./conv -i <input_file> -a <ans_file>" << endl;
      return 1;
    }
  }

  if (input_stream == nullptr) {
    input_stream = &cin;
  }

  Matrix2D<float> input(*input_stream);
  Matrix2D<float> kernel(*input_stream);
  int out_size = input.size - kernel.size + 1;

  cout << "input size: " << input.size << endl;
  cout << "kernel size: " << kernel.size << endl;
  cout << "output size: " << out_size << endl;

  Matrix2D<float> output(out_size);
  // {
  //     Timer timer("conv2d_naive");
  //     Matrix2D<float> output_naive(out_size);
  //     conv2d_naive(input.data.data(), output.data.data(), kernel.data.data(),
  //     input.size, out_size, kernel.size);
  // }

  //   {
  //     Timer timer("conv2d_omp");
  // #pragma noinline
  //     conv2d_omp(input.data.data(), output.data.data(), kernel.data.data(),
  //                input.size, out_size, kernel.size);
  //   }

  __itt_resume();
  {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    dnnl::memory::dims input_dims = {1, 1, input.size, input.size};
    dnnl::memory::dims kernel_dims = {1, 1, kernel.size, kernel.size};
    dnnl::memory::dims output_dims = {1, 1, out_size, out_size};
    dnnl::memory::dims strides = {1, 1};
    dnnl::memory::dims padding = {0, 0};
    dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::nchw);
    dnnl::memory::desc kernel_md(kernel_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::nchw);
    dnnl::memory::desc output_md(output_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::nchw);
    dnnl::memory input_mem(input_md, eng, input.data.data());
    dnnl::memory kernel_mem(kernel_md, eng, kernel.data.data());
    dnnl::memory output_mem(output_md, eng, output.data.data());
    dnnl::convolution_forward::primitive_desc conv_pd(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct, input_md, kernel_md, output_md,
        strides, padding, padding);

    Timer timer("conv2d_with_mkl_dnn");
    dnnl::convolution_forward(conv_pd).execute(s,
                                               {{DNNL_ARG_SRC, input_mem},
                                                {DNNL_ARG_WEIGHTS, kernel_mem},
                                                {DNNL_ARG_DST, output_mem}});
    s.wait();

    // conv2d_mkl_dnn(input.data.data(), output.data.data(),
    // kernel.data.data(),
    // input.size, out_size, kernel.size);
  }
  __itt_pause();

  __itt_resume();
  {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream s(eng);
    dnnl::memory::dims input_dims = {1, 1, input.size, input.size};
    dnnl::memory::dims kernel_dims = {1, 1, kernel.size, kernel.size};
    dnnl::memory::dims output_dims = {1, 1, out_size, out_size};
    dnnl::memory::dims strides = {1, 1};
    dnnl::memory::dims padding = {0, 0};
    dnnl::memory::desc input_md(input_dims, dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::nchw);
    dnnl::memory::desc kernel_md(kernel_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::nchw);
    dnnl::memory::desc output_md(output_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::nchw);
    dnnl::memory input_mem(input_md, eng, input.data.data());
    dnnl::memory kernel_mem(kernel_md, eng, kernel.data.data());
    dnnl::memory output_mem(output_md, eng, output.data.data());
    dnnl::convolution_forward::primitive_desc conv_pd(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct, input_md, kernel_md, output_md,
        strides, padding, padding);

    Timer timer("conv2d_with_mkl_dnn1");
    dnnl::convolution_forward(conv_pd).execute(s,
                                               {{DNNL_ARG_SRC, input_mem},
                                                {DNNL_ARG_WEIGHTS, kernel_mem},
                                                {DNNL_ARG_DST, output_mem}});
    s.wait();

    // conv2d_mkl_dnn(input.data.data(), output.data.data(),
    // kernel.data.data(),
    // input.size, out_size, kernel.size);
  }
  __itt_pause();

  //   __itt_resume();
  //   {
  //     Timer timer("mkl_opt");
  // #pragma noinline
  //     conv2d_mkl_dnn_opt(input.data.data(), output.data.data(),
  //                        kernel.data.data(), input.size, out_size,
  //                        kernel.size);
  //   }
  //   __itt_pause();

  //   __itt_resume();
  //   {
  //     Timer timer("conv2d_with_im2col_openmp");
  // #pragma noinline
  //     conv2d_with_im2col(input.data.data(), output.data.data(),
  //                        kernel.data.data(), input.size, out_size,
  //                        kernel.size);
  //   }
  //   __itt_pause();

  if (ans_ifs.is_open()) {
    Matrix2D<float> ans(ans_ifs);
    if (output == ans) {
      cout << "Correct!" << endl;
    } else {
      cout << "Wrong!" << endl;
    }
  } else {
    cout << "Output:" << endl;
    output.print();
  }
  return 0;
}
