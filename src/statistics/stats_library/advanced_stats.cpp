#include <iostream>
#include <vector>
#include <array>

#include "advanced_stats.h"
#include "tools.h"

using namespace std;

/**
 * \brief Computes the one-dimensional autocorrelation function of a 2d vector.
 */
vector<vector<double>> autocorrelation_function_1d(const vector<vector<double>>& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = multiply_elements(input_array);
    std::unordered_map<double, std::vector<double>> regrouped_vals;
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
    vector<array<double, 3>> single_dists_and_vals_2d;
    single_dists_and_vals_2d.reserve(input_array.size() * input_array[0].size());
    std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash> regrouped_vals;

    for (int y = 0; y < input_array.size(); y++)
    {
        for (int x = 0; x < input_array[0].size(); x++)
        {
            if (isnan(input_array[y][x])) continue;
            for (int j = y; j < input_array.size(); j++)
            {
                int i = (j == y) ? x + 1 : 0;
                for (; i < input_array[0].size(); i++)
                {
                    if (isnan(input_array[j][i])) continue;
                    double dist_x = i - x;
                    double dist_y = j - y;
                    double val = input_array[y][x] * input_array[j][i];
                    single_dists_and_vals_2d.push_back({dist_x, dist_y, val});
                }
            }
        }
    }

    while (!single_dists_and_vals_2d.empty())
    {
        regroup_distance(regrouped_vals, single_dists_and_vals_2d.back());
        single_dists_and_vals_2d.pop_back();
    }

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
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);
    std::unordered_map<double, std::vector<double>> regrouped_vals;
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
    // auto subtract = [](double v1, double v2) {return abs(v1 - v2);};
    // vector<array<double, 2>> single_dists_and_vals = apply_vector_2d(input_array, subtract);
    // vector<array<double, 2>> single_dists_and_vals = subtract_elements(input_array);
    
    // vector<dist_and_regrouped_vals> regrouped_vals;
    // while (single_dists_and_vals.size() > 0)
    // {
    //     // regroup_distance(regrouped_vals, single_dists_and_vals.back());
    //     regroup_distance_1d(single_dists_and_vals.back());
    //     single_dists_and_vals.pop_back();
    // }

    // vector<vector<double>> output_array;
    // double variance_val = variance(input_array);
    // for (const auto& [dist, vals]: regrouped_vals_1d)
    // {
    //     vector<double> pow2_values = pow2(vals);
    //     int N = pow2_values.size();
    //     if (N == 1) continue;

    //     double mean_val = mean(pow2_values);
    //     double std_val = standard_deviation(pow2_values);

    //     double structure = mean_val / (variance_val);
    //     double structure_uncertainty = std_val / (variance_val * sqrt(N-1));
    //     output_array.push_back(vector<double> {dist, structure, structure_uncertainty});
    // }
    // return output_array;
}

vector<vector<double>> increments(const vector<vector<double>>& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);
    std::unordered_map<double, std::vector<double>> regrouped_vals;
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
        vector<double> current_vector {dist};
        current_vector.insert(current_vector.end(), vals.begin(), vals.end());
        output_array.push_back(current_vector);
    }
    return output_array;
    // auto subtract = [](double v1, double v2) {return abs(v1 - v2);};
    // vector<array<double, 2>> single_dists_and_vals = apply_vector_2d(input_array, subtract);
    
    // vector<dist_and_regrouped_vals> regrouped_vals;
    // while (single_dists_and_vals.size() > 0)
    // {
    //     regroup_distance(regrouped_vals, single_dists_and_vals.back());
    //     single_dists_and_vals.pop_back();
    // }

    // vector<vector<double>> output_array;
    // for (const auto& unique_dist_and_val: regrouped_vals)
    // {
    //     vector<double> current_vector {unique_dist_and_val.dist};
    //     current_vector.insert(current_vector.end(), unique_dist_and_val.vals.begin(), unique_dist_and_val.vals.end());
    //     output_array.push_back(current_vector);
    // }
    // return output_array;
}

// BREAK

// /**
//  * \brief Regroups a single distance and value with a given of already existing distances and values.
//  */
// void regroup_distance(vector<dist_and_regrouped_vals>& regrouped_vals, const array<double, 2>& dist_and_val)
// {
//     for (auto& individual_dist_and_val : regrouped_vals)
//     {
//         if (individual_dist_and_val.dist == dist_and_val[0])
//         {
//             individual_dist_and_val.vals.push_back(dist_and_val[1]);
//             return;
//         }
//     }
//     regrouped_vals.push_back(dist_and_regrouped_vals {dist_and_val[0], vector<double>{dist_and_val[1]}});
// }

// /**
//  * \brief Applies an operation between each values of a map and computes the pixel distances.
//  * \param[in] input_array a vector of vectors whose values need to be compared with a specified function and their
//  * distances computed
//  * \param[in] function a function of two parameters to apply between every pair of pixels.
//  * \return a vector of arrays of two elements, which represent a pixel distance and the result of the applied function.
//  */
// template <typename T>
// vector<array<double, 2>> apply_vector_2d(const vector<vector<double>>& input_array, const T& function)
// {
//     vector<array<double, 2>> single_dists_and_vals;
//     for (int y = 0; y < input_array.size(); y++)
//     {
//         for (int x = 0; x < input_array[0].size(); x++)
//         {
//             // The pixel at (x,y) is the one currently being processed
//             if (isnan(input_array[y][x])) continue;
//             // j=y ensures that only rows above the current pixel are processed
//             for (int j = y; j < input_array.size(); j++)
//             {
//                 // The logic below starts the search at x+1 if j = y and at 0 if j > y
//                 int i = (j == y) ? x + 1 : 0;
//                 for (; i < input_array[0].size(); i++)
//                 {
//                     if (isnan(input_array[j][i])) continue;
//                     double dist = sqrt((i-x)*(i-x) + (j-y)*(j-y));
//                     double val = function(input_array[y][x], input_array[j][i]);
//                     single_dists_and_vals.push_back({dist, val});
//                 }
//             }
//         }
//     }
//     return single_dists_and_vals;
// }

// // /**
// //  * \brief Computes the structure function of a 2d vector
// //  */
// // vector<vector<double>> structure_function(const vector<vector<double>>& input_array)
// // {
// //     vector<array<double, 2>> single_dists_and_vals = subtract_elements(input_array);
    
// //     while (single_dists_and_vals.size() > 0)
// //     {
// //         regroup_distance_1d(single_dists_and_vals.back());
// //         single_dists_and_vals.pop_back();
// //     }

// //     vector<vector<double>> output_array;
// //     double variance_val = variance(input_array);
// //     for (const auto& unique_dist_and_val: regrouped_vals)
// //     {
// //         vector<double> pow2_values = pow2(unique_dist_and_val.vals);
// //         int N = pow2_values.size();
// //         if (N == 1) continue;

// //         double mean_val = mean(pow2_values);
// //         double std_val = standard_deviation(pow2_values);

// //         double structure = mean_val / (variance_val);
// //         double structure_uncertainty = std_val / (variance_val * sqrt(N-1));
// //         output_array.push_back(vector<double> {unique_dist_and_val.dist, structure, structure_uncertainty});

// //     }
// //     return output_array;
// // }
