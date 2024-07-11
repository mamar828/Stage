#include "stats.h"

using namespace std;

/**
 * \brief Compute the mean of a vector of dist_and_regrouped_vals structs.
 */
double mean(const vector<dist_and_regrouped_vals >& regrouped_vals)
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
 * \brief Compute the mean of a dist_and_regrouped_vals struct.
 */
double mean(const vector<double>& regrouped_vals)
{
    int size = regrouped_vals.size();
    double total = 0;
    for (double val : regrouped_vals)
    {
        if (!isnan(val)) total += val;
        else 
        {
            size--;
            // cout << "IT WAS ME" << endl;
            // throw;
        }
    }
    return total / size;
}


/**
 * \brief Compute the mean of a 2d vector.
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
 * \brief Compute the variance of a vector of dist_and_regrouped_vals structs.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector<dist_and_regrouped_vals >& regrouped_vals)
{
    double mean_val = mean(regrouped_vals);
    double numerator = 0;
    int denominator = 0;
    for (const auto& individual_dist_and_val : regrouped_vals)
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
