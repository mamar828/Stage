#include <vector>
#include <array>
#include "stats.h"

void regroup_distance(std::vector<dist_and_regrouped_vals>& regrouped_vals, const std::array<double, 2>& dist_and_val);
template <typename T>
std::vector<std::array<double, 2>> apply_vector_2d(const std::vector<std::vector<double>>& input_array,
                                                   const T& function);
std::vector<std::vector<double>> autocorrelation_function(const std::vector<std::vector<double>>& input_array);
std::vector<std::vector<double>> structure_function(const std::vector<std::vector<double>>& input_array);
std::vector<std::vector<double>> increments(const std::vector<std::vector<double>>& input_array);
