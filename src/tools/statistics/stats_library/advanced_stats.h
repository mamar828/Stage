#include "stats.h"

double_unordered_map autocorrelation_function_1d_calculation(vector_2d& input_array);
vector_2d autocorrelation_function_1d_kleiner_dickman(vector_2d& input_array);
vector_2d autocorrelation_function_1d_boily(vector_2d& input_array);
array_unordered_map autocorrelation_function_2d_calculation(vector_2d& input_array);
vector_2d autocorrelation_function_2d_kleiner_dickman(vector_2d& input_array);
vector_2d autocorrelation_function_2d_boily(vector_2d& input_array);
vector_2d structure_function(const vector_2d& input_array);
vector_2d increments(const vector_2d& input_array);
