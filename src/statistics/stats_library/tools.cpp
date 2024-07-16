#include "tools.h"

using namespace std;

/**
 * \brief Regroups a distance modulus to its struct.
 */
void regroup_distance(std::unordered_map<double, std::vector<double>>& regrouped_vals,
                      const array<double, 2>& dist_and_val)
{
    auto it = regrouped_vals.find(dist_and_val[0]);
    if (it != regrouped_vals.end())
    {
        it->second.push_back(dist_and_val[1]);
    }
    else
    {
        regrouped_vals[dist_and_val[0]] = {dist_and_val[1]};
    }
}

/**
 * \brief Regroups a distance vector to its struct.
 */
void regroup_distance(std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash>& regrouped_vals,
                      const array<double, 3>& dist_and_val)
{
    array<double, 2> dist = {dist_and_val[0], dist_and_val[1]};
    auto it = regrouped_vals.find(dist);
    if (it != regrouped_vals.end())
    {
        it->second.push_back(dist_and_val[2]);
    }
    else
    {
        regrouped_vals[dist] = {dist_and_val[2]};
    }
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
    vector<array<double, 2>> single_dists_and_vals;
    for (int y = 0; y < input_array.size(); y++)
    {
        for (int x = 0; x < input_array[0].size(); x++)
        {
            // The pixel at (x,y) is the one currently being processed
            if (isnan(input_array[y][x])) continue;
            // j=y ensures that only rows above the current pixel are processed
            for (int j = y; j < input_array.size(); j++)
            {
                // The logic below starts the search at x+1 if j = y and at 0 if j > y
                int i = (j == y) ? x + 1 : 0;
                for (; i < input_array[0].size(); i++)
                {
                    if (isnan(input_array[j][i])) continue;
                    double dist = sqrt((i-x)*(i-x) + (j-y)*(j-y));
                    double val = function(input_array[y][x], input_array[j][i]);
                    single_dists_and_vals.push_back({dist, val});
                }
            }
        }
    }
    return single_dists_and_vals;
}

std::vector<std::array<double, 2>> multiply_elements(const std::vector<std::vector<double>>& input_array)
{
    return apply_vector_map(input_array, [](double a, double b) {return a * b;});
}

std::vector<std::array<double, 2>> subtract_elements(const std::vector<std::vector<double>>& input_array)
{
    return apply_vector_map(input_array, [](double a, double b) {return abs(a - b);});
}
