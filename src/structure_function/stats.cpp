#include <numeric>
#include "utils.h"
#include "stats.h"

using namespace std;

/**
 * \brief Compute the mean of a vector of dist_and_regrouped_vals structs.
 */
double mean(const vector<dist_and_regrouped_vals >& regrouped_vals)
{
    int size;
    double total;
    for (const auto& individual_dist_and_val : regrouped_vals)
    {
        size += individual_dist_and_val.vals.size();
        total += reduce(begin(individual_dist_and_val.vals), end(individual_dist_and_val.vals));
    }
    return total / size;
}

/**
 * \brief Compute the mean of a dist_and_regrouped_vals struct.
 */
double mean(const vector<double>& regrouped_vals)
{
    int size = regrouped_vals.size();
    double total = reduce(begin(regrouped_vals), end(regrouped_vals));;
    return total / size;
}


/**
 * \brief Compute the mean of a 2d vector.
 */
double mean(const vector<vector<double> >& vals)
{
    int size;
    double total;
    for (const auto& val_vector : vals)
    {
        size += val_vector.size();
        total += reduce(begin(val_vector), end(val_vector));
    }
    return total / size;
}

/**
 * \brief Calculate the square of a vector.
 */
vector<double> pow2(const vector<double>& input_vals)
{
    vector<double> squared_vals;
    for (auto val: input_vals)
    {
        squared_vals.push_back(val * val);
    }
    return squared_vals;
}

/**
 * \brief Compute the variance of a 2d vector.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The sample variance is the one computed (the denominator is the population size N minus one (N-1).
 */
double variance(const vector<vector<double> >& vals)
{
    double mean_val = mean(vals);
    double numerator;
    int denominator;
    for (const auto& val_vector : vals)
    {
        for (const double& val : val_vector)
        {
            if (val != nan_val)
            {
                numerator += (val - mean_val) * (val - mean_val);
                denominator++;
            }
        }
    }
    return numerator / (denominator - 1);
}

/**
 * \brief Compute the variance of a vector of dist_and_regrouped_vals structs.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The sample variance is the one computed (the denominator is the population size N minus one (N-1).
 */
double variance(const vector<dist_and_regrouped_vals >& regrouped_vals)
{
    double mean_val = mean(regrouped_vals);
    double numerator;
    int denominator;
    for (const auto& individual_dist_and_val : regrouped_vals)
    {
        for (const double& val : individual_dist_and_val.vals)
        {
            numerator += (val - mean_val) * (val - mean_val);
            denominator++;
        }
    }
    return numerator / (denominator - 1);
}
