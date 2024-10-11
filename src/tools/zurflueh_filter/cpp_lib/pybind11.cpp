#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "zfilter.h"

using namespace std;

PYBIND11_MODULE(zfilter, m) {
    m.doc() = string("Module that allows the application of a zurflueh filter.");
    m.def("zfilter_cpp", &zfilter, 
          "Computes the Zurflueh filter of a two-dimensional array with a given kernel.");
}

// MAC :    clang++ -std=c++17 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.12 -m pybind11 --includes` zfilter.cpp pybind11.cpp -o zfilter.so `python3.12-config --ldflags` -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
// LINUX :  g++ -std=c++17 -shared -fPIC -I./pybind11/include/ `python3.12 -m pybind11 --includes` zfilter.cpp pybind11.cpp -o zfilter.so `python3.12-config --ldflags` -fopenmp -lm
