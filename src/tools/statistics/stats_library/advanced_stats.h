#include <vector>
#include <array>

#include "stats.h"

std::vector<std::vector<double>> autocorrelation_function_1d(std::vector<std::vector<double>>& input_array);
std::vector<std::vector<double>> autocorrelation_function_2d(std::vector<std::vector<double>>& input_array);
std::vector<std::vector<double>> structure_function(const std::vector<std::vector<double>>& input_array);
std::vector<std::vector<double>> increments(const std::vector<std::vector<double>>& input_array);
