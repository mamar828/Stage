#include <vector>
#include <array>

/**
 * \struct dist_and_regrouped_vals
 * \brief Regroup a certain distance to a vector of values that are linked to this distance.
 */
struct dist_and_regrouped_vals
{
    double dist;
    std::vector<double> vals;
};

/**
 * \struct dist_and_regrouped_vals_2d
 * \brief Regroup a certain 2d distance vector to a vector of values that are linked to this distance.
 */
struct dist_and_regrouped_vals_2d
{
    std::array<double, 2> dist;
    std::vector<double> vals;
};

double mean(const std::vector<dist_and_regrouped_vals>& regrouped_vals);
double mean(const std::vector<std::vector<double>>& vals);
double mean(const std::vector<double>& vals);
std::vector<double> pow2(const std::vector<double>& vals);
std::vector<double> log(const std::vector<double>& vals);
double variance(const std::vector<dist_and_regrouped_vals>& regrouped_vals);
double variance(const std::vector<std::vector<double>>& vals);
double variance(const std::vector<double>& vals);
double standard_deviation(const std::vector<double>& values);
