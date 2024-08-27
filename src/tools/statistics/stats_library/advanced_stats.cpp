#include <iostream>
#include <vector>
#include <array>
#include <omp.h>

#include "advanced_stats.h"
#include "tools.h"

using namespace std;

/**
 * \brief Computes the one-dimensional autocorrelation function of a 2d vector.
 */
vector<vector<double>> autocorrelation_function_1d(vector<vector<double>>& input_array)
{
    const size_t height = input_array.size();
    const size_t width = input_array[0].size();
    double mean_value = mean(input_array);

    #pragma omp parallel
    {
        #pragma omp for collapse(1) schedule(dynamic)
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                input_array[y][x] -= mean_value;
            }
        }
    }

    vector<array<double, 2>> single_dists_and_vals_1d = multiply_elements(input_array);
    single_dists_and_vals_1d.shrink_to_fit();
    
    unordered_map<double, vector<double>> regrouped_vals;
    while (!single_dists_and_vals_1d.empty())
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());

    double denominator = sum_of_squares(input_array);
    double N_t_sqrt = pow(count_non_nan(input_array), 0.5);

    // Find the zero-lag value
    double zero_lag_value = 0;
    for (const auto& [dist, vals] : regrouped_vals)
    {
        if (dist == 0)
        {
            double normalization_factor = N_t_sqrt / pow(vals.size(), 1.5);
            double C_i = normalization_factor * sum(vals) / denominator;
            zero_lag_value = C_i;
            break;
        }
    }

    for (const auto& [dist, vals] : regrouped_vals)
    {
        double normalization_factor = N_t_sqrt / pow(vals.size(), 1.5);
        double C_i = normalization_factor * sum(vals) / denominator;
        output_array.push_back({dist, C_i / zero_lag_value});
    }
    return output_array;
}

/**
 * \brief Computes the two-dimensional autocorrelation function of a 2d vector.
 */
vector<vector<double>> autocorrelation_function_2d(vector<vector<double>>& input_array)
{
    const size_t height = input_array.size();
    const size_t width = input_array[0].size();
    vector<array<double, 3>> single_dists_and_vals_2d;

    // Reserve an approximate size to avoid multiple allocations
    size_t max_possible_size = (height * width * (height * width - 1)) / 2;
    single_dists_and_vals_2d.reserve(max_possible_size);
    double mean_value = mean(input_array);

    #pragma omp parallel
    {
        #pragma omp for collapse(1) schedule(dynamic)
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                input_array[y][x] -= mean_value;
            }
        }
        vector<array<double, 3>> thread_single_dists_and_vals;
        thread_single_dists_and_vals.reserve(max_possible_size / omp_get_num_threads());

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                if (isnan(input_array[y][x])) continue;
                for (size_t j = y; j < height; j++)
                {
                    for (size_t i = (j == y) ? x /*+ 1*/ : 0; i < width; i++) // lag=0 is considered here
                    {
                        if (isnan(input_array[j][i])) continue;
                        double dist_x = int(i - x);
                        double dist_y = int(j - y);
                        double numerator = input_array[y][x] * input_array[j][i];
                        thread_single_dists_and_vals.push_back({dist_x, dist_y, numerator});
                    }
                }
            }
        }
        combine_vectors(single_dists_and_vals_2d, thread_single_dists_and_vals);
    }

    single_dists_and_vals_2d.shrink_to_fit();

    unordered_map<array<double, 2>, vector<double>, DoubleArrayHash> regrouped_vals;
    while (!single_dists_and_vals_2d.empty())
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_2d.back());
        single_dists_and_vals_2d.pop_back();
    }

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());

    double denominator = sum_of_squares(input_array);
    double N_t_sqrt = pow(count_non_nan(input_array), 0.5);

    // Find the zero-lag value
    double zero_lag_value = 0;
    for (const auto& [dist, vals] : regrouped_vals)
    {
        if (dist[0] == 0 and dist[1] == 0)
        {
            double normalization_factor = N_t_sqrt / pow(vals.size(), 1.5);
            double C_i = normalization_factor * sum(vals) / denominator;
            zero_lag_value = C_i;
            break;
        }
    }

    for (const auto& [dist, vals] : regrouped_vals)
    {
        double normalization_factor = N_t_sqrt / pow(vals.size(), 1.5);
        double C_i = normalization_factor * sum(vals) / denominator;
        output_array.push_back({dist[0], dist[1], C_i / zero_lag_value});
    }

    return output_array;
}

/**
 * \brief Computes the structure function of a 2d vector.
 */
vector<vector<double>> structure_function(const vector<vector<double>>& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);

    
    unordered_map<double, vector<double>> regrouped_vals;
    while (single_dists_and_vals_1d.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
        if (dist == 0) continue;
        vector<double> pow2_values = pow2(vals);
        int N = pow2_values.size();
        if (N == 1) continue;

        double mean_val = mean(pow2_values);
        double std_val = standard_deviation(pow2_values);

        double structure = mean_val / (variance_val);
        double structure_uncertainty = std_val / (variance_val * sqrt(N-1));
        output_array.push_back(vector<double> {dist, structure, structure_uncertainty});
    }
    return output_array;
}

vector<vector<double>> increments(const vector<vector<double>>& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);

    
    unordered_map<double, vector<double>> regrouped_vals;
    while (single_dists_and_vals_1d.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
        if (dist == 0) continue;
        vector<double> current_vector {dist};
        current_vector.insert(current_vector.end(), vals.begin(), vals.end());
        output_array.push_back(current_vector);
    }
    return output_array;
}
