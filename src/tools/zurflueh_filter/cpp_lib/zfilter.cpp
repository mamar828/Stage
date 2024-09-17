#include <iostream>
#include <numeric>
#include <omp.h>

#include "zfilter.h"

using namespace std;

/**
 * \brief Computes the Zurflueh filter of a two-dimensional array.
 */
vector_2d zfilter(const vector_2d& input_array, const vector_2d& filter_)
{
    int height = input_array.size(), width = input_array[0].size();
    vector_2d output_array(height, vector<double>(width, numeric_limits<float>::quiet_NaN()));

    int w = filter_.size();
    int hw = w / 2;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (isnan(input_array[y][x])) continue;
            double filtered_val = 0.0, sum = 0.0;
            for (int j = max(0, hw - y); j < min(w, height - y + hw); j++)
            {
                for (int i = max(0, hw - x); i < min(w, height - x + hw); i++)
                {
                    if (isnan(input_array[y+j - hw][x+i - hw])) continue;
                    filtered_val += filter_[j][i] * input_array[y+j - hw][x+i - hw];
                    sum += filter_[j][i];
                }
            }
            output_array[y][x] = filtered_val / sum;
        }
    }
    return output_array;
}
