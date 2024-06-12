#include <cstddef>
#include <iostream>
#include <istream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <unistd.h>
#include <ittnotify.h>


#include "conv.hpp"

using namespace std;

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    Timer(const std::string &name) : Timer() { name_ = name; }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_).count();
    }

    ~Timer() {
        cout << "Timer " << name_ << ":" << endl;
        cout << "Time elapsed: " << elapsed() << "s" << endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::string name_;

};

template <typename T>
struct Matrix2D {
    vector<T> data;
    int size;

    Matrix2D(int size) : size(size) {
        data.resize(size * size);
    }

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
                if (fabs(lhs.data[i * lhs.size + j] - rhs.data[i * lhs.size + j]) > EPS) {
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

    Matrix2D<int> input(*input_stream);
    Matrix2D<int> kernel(*input_stream);
    int out_size = input.size - kernel.size + 1;

    cout << "input size: " << input.size << endl;
    cout << "kernel size: " << kernel.size << endl;
    cout << "output size: " << out_size << endl;

    Matrix2D<int> output(out_size);
    {
        Timer timer("conv2d_naive");
        Matrix2D<int> output_naive(out_size);
        conv2d_naive(input.data.data(), output.data.data(), kernel.data.data(), input.size, out_size, kernel.size);
    }

     
    {
        Timer timer("conv2d_omp");
        #pragma noinline
        conv2d_omp(input.data.data(), output.data.data(), kernel.data.data(), input.size, out_size, kernel.size);
    }
    
    __itt_resume();
    {
        Timer timer("conv2d_with_im2col_openmp");
        #pragma noinline
        conv2d_with_im2col(input.data.data(), output.data.data(), kernel.data.data(), input.size, out_size, kernel.size);
    }
    __itt_pause();

    if (ans_ifs.is_open()) {
        Matrix2D<int> ans(ans_ifs);
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
