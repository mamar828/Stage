#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "advanced_stats.h"

using namespace std;

PYBIND11_MODULE(advanced_stats, m) {
    m.doc() = string("Module that regroups the necessary statistic and analysis tools to compute autocorrelation and ")
            + string("structure functions.");
    m.def("autocorrelation_function_cpp", &autocorrelation_function, 
          "Compute the one-dimensional autocorrelation function of a two-dimensional array.");
    m.def("structure_function_cpp", &structure_function, 
          "Compute the one-dimensional structure function of a two-dimensional array.");
    m.def("increments_cpp", &increments, 
          "Compute the one-dimensional increment function of a two-dimensional array.");
}

// clang++ -std=c++17 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.12 -m pybind11 --includes` advanced_stats.cpp utils.cpp stats.cpp pybind11.cpp -o advanced_stats.so `python3.12-config --ldflags`
