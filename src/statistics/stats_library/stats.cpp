#include "stats.h"

using namespace std;

/**
 * \brief Computes the mean of a vector of dist_and_regrouped_vals structs.
 */
double mean(const vector<dist_and_regrouped_vals>& regrouped_vals)
{
    int size = 0;
    double total = 0;
    for (const auto& individual_dist_and_val : regrouped_vals)
    {
        size += individual_dist_and_val.vals.size();
        for (double val : individual_dist_and_val.vals)
        if (!isnan(val)) total += val;
        else size--;
    }
    return total / size;
}

/**
 * \brief Computes the mean of a dist_and_regrouped_vals struct.
 */
double mean(const vector<double>& vals)
{
    int size = vals.size();
    double total = 0;
    for (double val : vals)
    {
        if (!isnan(val)) total += val;
        else 
        size--;
    }
    return total / size;
}

/**
 * \brief Computes the mean of a 2d vector.
 */
double mean(const vector<vector<double>>& vals)
{
    int size = 0;
    double total = 0;
    for (const auto& val_vector : vals)
    {
        size += val_vector.size();
        for (double val : val_vector)
        {
            if (!isnan(val)) total += val;
            else size--;
        }
    }
    return total / size;
}

/**
 * \brief Calculate the square of a vector.
 */
vector<double> pow2(const vector<double>& vals)
{
    vector<double> squared_vals;
    for (auto val: vals)
    {
        squared_vals.push_back(val * val);
    }
    return squared_vals;
}

/**
 * \brief Calculate the natural logarithm of a vector.
 */
vector<double> log(const vector<double>& vals)
{
    vector<double> log_vals;
    for (auto val: vals)
    {
        log_vals.push_back(log(val));
    }
    return log_vals;
}

/**
 * \brief Computes the variance of a 2d vector.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector<vector<double>>& vals)
{
    double mean_val = mean(vals);
    double numerator = 0;
    int denominator = 0;
    for (const auto& val_vector : vals)
    {
        for (const double& val : val_vector)
        {
            if (!isnan(val))
            {
                numerator += (val - mean_val) * (val - mean_val);
                denominator++;
            }
        }
    }
    return numerator / (denominator);
}

/**
 * \brief Computes the variance of a vector of dist_and_regrouped_vals structs.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector<dist_and_regrouped_vals>& vals)
{
    double mean_val = mean(vals);
    double numerator = 0;
    int denominator = 0;
    for (const auto& individual_dist_and_val : vals)
    {
        for (const double& val : individual_dist_and_val.vals)
        {
            if (!isnan(val))
            {
                numerator += (val - mean_val) * (val - mean_val);
                denominator++;
            }
        }
    }
    return numerator / (denominator);
}

/**
 * \brief Computes the variance of a vector.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector<double>& vals)
{
    double mean_val = mean(vals);
    double numerator = 0;
    int denominator = 0;
    for (const double& val : vals)
    {
        if (!isnan(val))
        {
            numerator += (val - mean_val) * (val - mean_val);
            denominator++;
        }
    }
    return numerator / (denominator);
}

/**
 * \brief Computes the standard deviation of a vector.
 */
double standard_deviation(const vector<double>& values)
{
    double variance_val = variance(values);
    return sqrt(variance_val);
}
