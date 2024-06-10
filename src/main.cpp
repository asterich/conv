#include <iostream>
#include <istream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <memory>

#include "conv.hpp"

using namespace std;

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

    friend bool operator==(const Matrix2D &lhs, const Matrix2D &rhs) {
        static constexpr float EPS = 1e-6;
        if (lhs.size != rhs.size) {
            return false;
        }
        for (int i = 0; i < lhs.size; i++) {
            for (int j = 0; j < lhs.size; j++) {
                if (fabsf(lhs.data[i * lhs.size + j] - rhs.data[i * lhs.size + j]) > EPS) {
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
    }
};

int main(int argc, char *argv[]) {
    /* Usage: ./conv [filename] */
    istream *input_stream;
    ifstream ifs;
    switch (argc) {
        case 1:
            input_stream = &cin;
            break;
        case 2:
            ifs.open(argv[1]);
            if (!ifs.is_open()) {
                cerr << "Error: cannot open file " << argv[1] << endl;
                return 1;
            }
            input_stream = &ifs;
            break;
        default:
            cerr << "Usage: " << argv[0] << " [filename]" << endl;
            return 1;
    }

    Matrix2D<int> input(*input_stream);
    Matrix2D<int> kernel(*input_stream);
    int out_size = input.size - kernel.size + 1;
    Matrix2D<int> output(out_size);

    conv2d(input.data.data(), output.data.data(), kernel.data.data(), input.size, out_size, kernel.size);

    output.print();
    return 0;
}
