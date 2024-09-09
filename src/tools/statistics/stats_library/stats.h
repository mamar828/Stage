#include "tools.h"

double mean(const std::vector<double>& vals);
double mean(const vector_2d& vals);
double sum(const std::vector<double>& vals);
double sum(const vector_2d& vals);
double sum_of_squares(const vector_2d& vals);
std::vector<double> pow2(const std::vector<double>& vals);
std::vector<double> log(const std::vector<double>& vals);
double variance(const std::vector<double>& vals);
double variance(const vector_2d& vals);
double standard_deviation(const std::vector<double>& vals);
int count_non_nan(const vector_2d& vals);
void subtract_mean(vector_2d& input_array);
