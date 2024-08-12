#include <omp.h>

#include "tools.h"

using namespace std;

/**
 * \brief Regroups a distance modulus to its struct.
 */
void regroup_distance(unordered_map<double, vector<double>>& regrouped_vals,
                      const array<double, 2>& dist_and_val)
{
    #pragma omp critical
    regrouped_vals[dist_and_val[0]].push_back(dist_and_val[1]);
}

/**
 * \brief Regroups a distance vector to its struct.
 */
void regroup_distance(unordered_map<array<double, 2>, vector<double>, DoubleArrayHash>& regrouped_vals,
                      const array<double, 3>& dist_and_val)
{
    array<double, 2> dist = {dist_and_val[0], dist_and_val[1]};
    #pragma omp critical
    regrouped_vals[dist].push_back(dist_and_val[2]);
}

/**
 * \brief Combines two vectors of arrays of two elements.
 */
void combine_vectors(vector<array<double,2>>& dest, const vector<array<double,2>>& src)
{
    #pragma omp critical
    dest.insert(dest.end(), src.begin(), src.end());
}

/**
 * \brief Combines two vectors of arrays of three elements.
 */
void combine_vectors(vector<array<double,3>>& dest, const vector<array<double,3>>& src)
{
    #pragma omp critical
    dest.insert(dest.end(), src.begin(), src.end());
}

/**
 * \brief Applies an operation between each values of a map and computes the pixel distances.
 * \param[in] input_array a vector of vectors whose values need to be compared with a specified function and their
 * distances computed.
 * \param[in] function a function of two parameters to apply between every pair of pixels.
 * \return a vector of arrays of two elements, which represent a pixel distance and the result of the applied function.
 */
template <typename T>
vector<array<double, 2>> apply_vector_map(const vector<vector<double>>& input_array, const T& function)
{
    const size_t height = input_array.size();
    const size_t width = input_array[0].size();
    vector<array<double, 2>> single_dists_and_vals;

    // Reserve an approximate size to avoid multiple allocations
    size_t max_possible_size = (height * width * (height * width - 1)) / 2;
    single_dists_and_vals.reserve(max_possible_size);

    #pragma omp parallel
    {
        vector<array<double, 2>> thread_single_dists_and_vals;
        thread_single_dists_and_vals.reserve(max_possible_size / omp_get_num_threads());

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                if (isnan(input_array[y][x])) continue;
                for (size_t j = y; j < height; ++j)
                {
                    for (size_t i = (j == y) ? x /* + 1 */ : 0; i < width; ++i)
                    {
                        if (isnan(input_array[j][i])) continue;
                        double dist = sqrt((i - x) * (i - x) + (j - y) * (j - y));
                        double val = function(input_array[y][x], input_array[j][i]);
                        thread_single_dists_and_vals.push_back({dist, val});
                    }
                }
            }
        }

        // Combine the thread-local results into the global vector
        combine_vectors(single_dists_and_vals, thread_single_dists_and_vals);
    }

    // Optionally shrink to fit if memory usage is a concern
    single_dists_and_vals.shrink_to_fit();

    return single_dists_and_vals;
}

vector<array<double, 2>> multiply_elements(const vector<vector<double>>& input_array)
{
    return apply_vector_map(input_array, [](double a, double b) { return a * b; });
}

vector<array<double, 2>> subtract_elements(const vector<vector<double>>& input_array)
{
    return apply_vector_map(input_array, [](double a, double b) { return abs(a - b); });
}
