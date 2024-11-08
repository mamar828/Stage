#include <omp.h>

#include "time.h"
#include "advanced_stats.h"

using namespace std;

Time current_time;

/**
 * Calculates and regroups the distances used in computing the one-dimensional autocorrelation function.
 */
double_unordered_map autocorrelation_function_1d_calculation(vector_2d& input_array)
{
    subtract_mean(input_array);

    vector<array<double, 2>> single_dists_and_vals_1d = multiply_elements(input_array);
    single_dists_and_vals_1d.shrink_to_fit();
    double_unordered_map regrouped_vals;
    regroup_distance_thread_local(regrouped_vals, single_dists_and_vals_1d);
    return regrouped_vals;
}

/**
 * \brief Computes the one-dimensional autocorrelation function of two-dimensional data using the technique described in
 * Kleiner, S.C. and Dickman, R.L. 1984.
 */
vector_2d autocorrelation_function_1d_kleiner_dickman(vector_2d& input_array)
{
    double_unordered_map regrouped_vals = autocorrelation_function_1d_calculation(input_array);
    vector_2d output_array;
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
 * \brief Computes the one-dimensional autocorrelation function of two-dimensional data using the technique described in
 * Boily E. 1993.
 */
vector_2d autocorrelation_function_1d_boily(vector_2d& input_array)
{
    double_unordered_map regrouped_vals = autocorrelation_function_1d_calculation(input_array);
    vector_2d output_array;
    output_array.reserve(regrouped_vals.size());

    double denominator = sum_of_squares(input_array) / count_non_nan(input_array);

    // Thread-local storage for results
    vector<vector_2d> thread_local_results(omp_get_max_threads());

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        vector_2d& local_output = thread_local_results[thread_id];

        #pragma omp for
        for (int i = 0; i < regrouped_vals.size(); ++i)
        {
            auto it = next(regrouped_vals.begin(), i); // Access ith element
            const auto& [dist, vals] = *it;

            int N = vals.size();
            if (N == 1) continue;
            local_output.push_back(
                {dist, mean(vals) / denominator, standard_deviation(vals) / (denominator * sqrt(N-1))}
            );
        }
    }

    // Combine results from all threads
    for (const auto& local_result : thread_local_results)
    {
        output_array.insert(output_array.end(), local_result.begin(), local_result.end());
    }

    return output_array;
}

/**
 * Calculates and regroups the distances used in computing the two-dimensional autocorrelation function.
 */
array_unordered_map autocorrelation_function_2d_calculation(vector_2d& input_array)
{
    subtract_mean(input_array);

    const size_t height = input_array.size();
    const size_t width = input_array[0].size();
    vector<array<double, 3>> single_dists_and_vals_2d;

    size_t max_possible_size = (height * width * (height * width - 1)) / 2;

    #pragma omp parallel
    {
        vector<array<double, 3>> thread_single_dists_and_vals;
        // Reserve an approximate size to avoid multiple allocations
        thread_single_dists_and_vals.reserve(max_possible_size / omp_get_num_threads());

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                if (isnan(input_array[y][x])) continue;
                for (size_t j = y; j < height; j++)
                {
                    for (size_t i = (j == y) ? x : 0; i < width; i++) // lag=0 is considered here
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

    array_unordered_map regrouped_vals;
    regroup_distance_thread_local(regrouped_vals, single_dists_and_vals_2d);
    return regrouped_vals;
}

/**
 * \brief Computes the two-dimensional autocorrelation function of two-dimensional data using the technique described in
 * Kleiner, S.C. and Dickman, R.L. 1984.
 */
vector_2d autocorrelation_function_2d_kleiner_dickman(vector_2d& input_array)
{
    array_unordered_map regrouped_vals = autocorrelation_function_2d_calculation(input_array);
    vector_2d output_array;
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
 * \brief Computes the two-dimensional autocorrelation function of two-dimensional data using the technique described in
 * Boily E. 1993.
 */
vector_2d autocorrelation_function_2d_boily(vector_2d& input_array)
{
    array_unordered_map regrouped_vals = autocorrelation_function_2d_calculation(input_array);
    vector_2d output_array;
    output_array.reserve(regrouped_vals.size());

    double denominator = sum_of_squares(input_array) / count_non_nan(input_array);

    // Thread-local storage for results
    vector<vector_2d> thread_local_results(omp_get_max_threads());

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        vector_2d& local_output = thread_local_results[thread_id];

        #pragma omp for
        for (int i = 0; i < regrouped_vals.size(); ++i)
        {
            auto it = next(regrouped_vals.begin(), i); // Access ith element
            const auto& [dist, vals] = *it;

            local_output.push_back({dist[0], dist[1], mean(vals) / denominator});
        }
    }

    // Combine results from all threads
    for (const auto& local_result : thread_local_results)
    {
        output_array.insert(output_array.end(), local_result.begin(), local_result.end());
    }
    return output_array;
}

/**
 * \brief Computes the structure function of two-dimensional data.
 */
vector_2d structure_function(const vector_2d& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);
    
    double_unordered_map regrouped_vals;
    regroup_distance_thread_local(regrouped_vals, single_dists_and_vals_1d);

    vector_2d output_array;
    output_array.reserve(regrouped_vals.size());
    double variance_val = variance(input_array);

    // Thread-local storage for results
    vector<vector_2d> thread_local_results(omp_get_max_threads());

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        vector_2d& local_output = thread_local_results[thread_id];

        #pragma omp for
        for (int i = 0; i < regrouped_vals.size(); ++i)
        {
            auto it = next(regrouped_vals.begin(), i); // Access ith element
            const auto& [dist, vals] = *it;
            if (dist == 0) continue;

            vector<double> pow2_values = pow2(vals);
            int N = pow2_values.size();
            if (N == 1) continue;

            double mean_val = mean(pow2_values);
            double std_val = standard_deviation(pow2_values);
            double structure = mean_val / variance_val;
            double structure_uncertainty = std_val / (variance_val * sqrt(N - 1));

            // Store result in thread-local buffer
            local_output.push_back({dist, structure, structure_uncertainty});
        }
    }

    // Combine results from all threads
    for (const auto& local_result : thread_local_results)
    {
        output_array.insert(output_array.end(), local_result.begin(), local_result.end());
    }

    return output_array;
}

vector_2d increments(const vector_2d& input_array)
{
    vector<array<double, 2>> single_dists_and_vals_1d = subtract_elements(input_array);

    double_unordered_map regrouped_vals;
    regroup_distance_thread_local(regrouped_vals, single_dists_and_vals_1d);

    vector_2d output_array;
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
