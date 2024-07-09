#include <iostream>
#include <vector>
#include <array>

#include "utils.h"
#include "advanced_stats.h"

using namespace std;

/**
 * \brief Regroup a single distance and value with a given of already existing distances and values.
 */
void regroup_distance(vector<dist_and_regrouped_vals>& regrouped_vals, const array<double, 2>& dist_and_val)
{
    for (auto& individual_dist_and_val : regrouped_vals)
    {
        if (individual_dist_and_val.dist == dist_and_val[0])
        {
            individual_dist_and_val.vals.push_back(dist_and_val[1]);
            return;
        }
    }
    regrouped_vals.push_back(dist_and_regrouped_vals {dist_and_val[0], vector<double>{dist_and_val[1]}});
}

/**
 * \brief Compute the autocorrelation function of a 2d vector.
 */
vector<vector<double> > autocorrelation_function(vector<vector<double> >& input_array)
{
    vector<array<double, 2> > single_dists_and_vals;
    for (int y = 0; y < input_array.size(); y++)
    {
        for (int x = 0; x < input_array[0].size(); x++)
        {
            // The pixel at (x,y) is the one currently being processed
            if (input_array.at(y).at(x) == nan_val) break;
            // j=y ensures that only rows above the current pixel are processed
            for (int j = y; j < input_array.size(); j++)
            {
                // The logic below starts the search at x+1 if j = y and at 0 if j > y
                int i = (j == y) ? x + 1 : 0;
                for (; i < input_array[0].size(); i++)
                {
                    if (input_array.at(i).at(j) == nan_val) break;
                    double dist = sqrt((i-x)*(i-x) + (j-y)*(j-y));
                    double val = input_array[y][x] * input_array[j][i];
                    single_dists_and_vals.push_back({dist, val});
                }
            }
        }
    }
    cout << "Finished calculations" << endl;
    
    vector<dist_and_regrouped_vals > regrouped_vals;
    while (single_dists_and_vals.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals.back());
        single_dists_and_vals.pop_back();
    }
    cout << "Finished regrouping" << endl;

    vector<vector<double> > output_array;
    double variance_val = variance(input_array);
    for (const auto& unique_dist_and_val: regrouped_vals)
    {
        double mean_val = mean(unique_dist_and_val.vals);
        double autocorrelation = mean_val / variance_val;
        output_array.push_back(vector<double> {unique_dist_and_val.dist, autocorrelation});
    }
    return output_array;
}

/**
 * \brief Compute the structure function of a 2d vector
 */
vector<vector<double> > structure_function(vector<vector<double> >& input_array)
{
    vector<array<double, 2> > single_dists_and_vals;
    for (int y = 0; y < input_array.size(); y++)
    {
        for (int x = 0; x < input_array[0].size(); x++)
        {
            // The pixel at (x,y) is the one currently being processed
            if (input_array.at(y).at(x) == nan_val) break;
            // j=y ensures that only rows above the current pixel are processed
            for (int j = y; j < input_array.size(); j++)
            {
                // The logic below starts the search at x+1 if j = y and at 0 if j > y
                int i = (j == y) ? x + 1 : 0;
                for (; i < input_array[0].size(); i++)
                // for (int i = 0; i < input_array[0].size(); i++)
                {
                    if (input_array.at(i).at(j) == nan_val) break;
                    double dist = sqrt((i-x)*(i-x) + (j-y)*(j-y));
                    double val = abs(input_array[y][x] - input_array[j][i]);
                    single_dists_and_vals.push_back({dist, val});
                }
            }
        }
    }
    cout << "Finished calculations" << endl;
    
    vector<dist_and_regrouped_vals> regrouped_vals;
    while (single_dists_and_vals.size() > 0)
    {
        regroup_distance(regrouped_vals, single_dists_and_vals.back());
        single_dists_and_vals.pop_back();
    }
    cout << "Finished regrouping" << endl;

    vector<vector<double> > output_array;
    double variance_val = variance(input_array);
    for (const auto& unique_dist_and_val: regrouped_vals)
    {
        double mean_val = mean(pow2(unique_dist_and_val.vals));
        double structure = mean_val / variance_val;
        output_array.push_back(vector<double> {unique_dist_and_val.dist, structure});
    }
    return output_array;
}

// int main()
// {
//     // Define an array of arrays representing an image
//     vector<vector<double> > input_array(3, vector<double>(3, 1));
//     fill_vector(input_array, vector<double> {0,1,2,3,4,5,6,7,8});
//     // print(input_array);

//     // Define an array to store the autocorrelation results
//     vector<vector<double> > output_array;
//     // print(output_array);

//     // Call the autocorrelation function
//     autocorrelation_function(input_array, output_array);

//     return 0;
// }
