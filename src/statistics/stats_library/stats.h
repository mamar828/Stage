#include <vector>
#include <array>

double mean(const std::vector<double>& vals);
double mean(const std::vector<std::vector<double>>& vals);
double sum(const std::vector<double>& vals);
double sum(const std::vector<std::vector<double>>& vals);
double sum_of_squares(const std::vector<std::vector<double>>& vals);
std::vector<double> pow2(const std::vector<double>& vals);
std::vector<double> log(const std::vector<double>& vals);
double variance(const std::vector<double>& vals);
double variance(const std::vector<std::vector<double>>& vals);
double standard_deviation(const std::vector<double>& vals);
int count_non_nan(const std::vector<std::vector<double>>& vals);
