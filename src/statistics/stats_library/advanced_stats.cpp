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
vector<vector<double>> autocorrelation_function_1d(const vector<vector<double>>& input_array)
{
    cout << "Looping started" << endl;
    vector<array<double, 2>> single_dists_and_vals_1d = multiply_elements(input_array);
    cout << "Looping finished" << endl;
    
    std::unordered_map<double, std::vector<double>> regrouped_vals;
    cout << "Regroupment started" << endl;
    while (single_dists_and_vals_1d.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }
    cout << "Regroupment finished" << endl;

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
        int N = vals.size();
        if (N == 1) continue;
        
        double mean_val = mean(vals);
        double std_val = standard_deviation(vals);
        
        double acr = mean_val / variance_val;
        double acr_uncertainty = std_val / (variance_val * sqrt(N-1));
        output_array.push_back(vector<double> {dist, acr, acr_uncertainty});
    }
    return output_array;
}

/**
 * \brief Computes the two-dimensional autocorrelation function of a 2d vector.
 */
vector<vector<double>> autocorrelation_function_2d(const vector<vector<double>>& input_array)
{
    cout << "Looping started" << endl;

    const size_t height = input_array.size();
    const size_t width = input_array[0].size();
    vector<array<double, 3>> single_dists_and_vals_2d;

    #pragma omp parallel
    {
        vector<array<double, 3>> thread_single_dists_and_vals;

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                if (isnan(input_array[y][x])) continue;
                for (size_t j = y; j < height; j++)
                {
                    for (size_t i = (j == y) ? x + 1 : 0; i < width; i++)
                    {
                        if (isnan(input_array[j][i])) continue;
                        double dist_x = int(i - x);
                        double dist_y = int(j - y);
                        double val = input_array[y][x] * input_array[j][i];
                        thread_single_dists_and_vals.push_back({dist_x, dist_y, val});
                    }
                }
            }
        }
        #pragma omp critical
        combine_vectors(single_dists_and_vals_2d, thread_single_dists_and_vals);
    }
    cout << "Looping finished" << endl;

    std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash> regrouped_vals;
    cout << "Regroupment started" << endl;
    while (!single_dists_and_vals_2d.empty())
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_2d.back());
        single_dists_and_vals_2d.pop_back();
    }
    cout << "Regroupment finished" << endl;

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
        double mean_val = mean(vals);
        double acr = mean_val / variance_val;
        output_array.push_back({dist[0], dist[1], acr});
    }

    return output_array;
}

/**
 * \brief Computes the structure function of a 2d vector.
 */
vector<vector<double>> structure_function(const vector<vector<double>>& input_array)
{
    cout << "Looping started" << endl;
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);
    cout << "Looping finished" << endl;

    
    std::unordered_map<double, std::vector<double>> regrouped_vals;
    cout << "Regroupment started" << endl;
    while (single_dists_and_vals_1d.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }
    cout << "Regroupment finished" << endl;

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
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
    cout << "Looping started" << endl;
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);
    cout << "Looping finished" << endl;

    
    std::unordered_map<double, std::vector<double>> regrouped_vals;
    cout << "Regroupment started" << endl;
    while (single_dists_and_vals_1d.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_1d.back());
        single_dists_and_vals_1d.pop_back();
    }
    cout << "Regroupment finished" << endl;

    vector<vector<double>> output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    for (const auto& [dist, vals] : regrouped_vals)
    {
        vector<double> current_vector {dist};
        current_vector.insert(current_vector.end(), vals.begin(), vals.end());
        output_array.push_back(current_vector);
    }
    return output_array;
}
