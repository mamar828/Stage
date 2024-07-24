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
    m.def("acr_func_1d_cpp", &autocorrelation_function_1d, 
          "Compute the one-dimensional autocorrelation function of a two-dimensional array.");
    m.def("acr_func_2d_cpp", &autocorrelation_function_2d, 
          "Compute the two-dimensional autocorrelation function of a two-dimensional array.");
    m.def("str_func_cpp", &structure_function, 
          "Compute the one-dimensional structure function of a two-dimensional array.");
    m.def("increments_cpp", &increments, 
          "Compute the one-dimensional increment function of a two-dimensional array.");
}

// clang++ -std=c++17 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.12 -m pybind11 --includes` advanced_stats.cpp utils.cpp stats.cpp pybind11.cpp -o advanced_stats.so `python3.12-config --ldflags`
// clang++ -std=c++17 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.12 -m pybind11 --includes` advanced_stats.cpp stats.cpp tools.cpp pybind11.cpp -o advanced_stats.so `python3.12-config --ldflags` -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
