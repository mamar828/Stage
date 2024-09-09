#include <numeric>
#include <utility>
#include <algorithm>

#include "stats.h"

using namespace std;

/**
 * \brief Computes the mean of a vector.
 */
double mean(const vector<double>& vals)
{
    double total = std::accumulate(vals.begin(), vals.end(), 0.0, [](double total, double val) {
        return isnan(val) ? total : total + val;
    });
    int size = std::count_if(vals.begin(), vals.end(), [](double val) {
        return !isnan(val);
    });
    return total / size;
}

/**
 * \brief Computes the mean of a 2d vector.
 */
double mean(const vector_2d& vals)
{
    int size = 0;
    double total = 0;
    for (const auto& val_vector : vals)
    {
        pair<double, int> result = accumulate(
            val_vector.begin(), val_vector.end(), make_pair(0.0, 0), []
            (pair<double, int> acc, double val)
            {return isnan(val) ? acc : make_pair(acc.first + val, acc.second + 1);}
        );
        total += result.first;
        size += result.second;
    }
    return total / size;
}

/**
 * \brief Computes the sum of a vector.
 */
double sum(const std::vector<double>& vals)
{
    double total = accumulate(
        vals.begin(), vals.end(), 0.0,
        [](double acc, double val)
        {return acc + val;}
    );
    return total;
}

/**
 * \brief Computes the sum of a 2d vector.
 */
double sum(const std::vector<std::vector<double>>& vals)
{
    double total = accumulate(
        vals.begin(), vals.end(), 0.0,
        [](double acc, const vector<double>& val_vector)
        {
            double inner_result = accumulate(
                val_vector.begin(), val_vector.end(), 0.0, 
                [](double inner_acc, double val)
                {return isnan(val) ? inner_acc : inner_acc + val;}
            );
            return acc + inner_result;
        }
    );
    return total;
}

/**
 * \brief Computes the sum of the squares of a 2d vector.
 */
double sum_of_squares(const vector_2d& vals)
{
    double total = accumulate(
        vals.begin(), vals.end(), 0.0,
        [](double acc, const vector<double>& val_vector)
        {
            double inner_result = accumulate(
                val_vector.begin(), val_vector.end(), 0.0, 
                [](double inner_acc, double val)
                {return isnan(val) ? inner_acc : inner_acc + val * val;}
            );
            return acc + inner_result;
        }
    );
    return total;
}

/**
 * \brief Calculate the square of a vector.
 */
vector<double> pow2(const vector<double>& vals)
{
    vector<double> squared_vals(vals.size());
    transform(vals.begin(), vals.end(), squared_vals.begin(), [](double val){return val*val;});
    return squared_vals;
}

/**
 * \brief Calculate the natural logarithm of a vector.
 */
vector<double> log(const vector<double>& vals)
{
    vector<double> log_vals(vals.size());
    transform(vals.begin(), vals.end(), log_vals.begin(), [](double val){return log(val);});
    return log_vals;
}

/**
 * \brief Computes the variance of a vector.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector<double>& vals)
{
    double mean_val = mean(vals);
    pair<double, int> result = accumulate(
        vals.begin(), vals.end(), make_pair(0.0, 0), [mean_val]
        (pair<double, int> acc, double val)
        {return isnan(val) ? acc : make_pair(acc.first + (val - mean_val) * (val - mean_val), acc.second + 1);}
    );
    return result.first / result.second;
}

/**
 * \brief Computes the variance of a 2d vector.
 * \note For the sake of performance, this function may only be used with real values and not complex ones.
 * \note The population variance is the one computed (the denominator is the population size N).
 */
double variance(const vector_2d& vals)
{
    double mean_val = mean(vals);
    pair<double, int> result = accumulate(
        vals.begin(), vals.end(), make_pair(0.0, 0), 
        [mean_val](pair<double, int> acc, const vector<double>& val_vector)
        {
            pair<double, int> inner_result = accumulate(
                val_vector.begin(), val_vector.end(), make_pair(acc.first, acc.second), [mean_val]
                (pair<double, int> inner_acc, double val)
                {return isnan(val) ? inner_acc : make_pair(inner_acc.first + (val - mean_val) * (val - mean_val), 
                                                           inner_acc.second + 1);}
            );
            return make_pair(inner_result.first, inner_result.second);
        }
    );
    return result.first / result.second;
}

/**
 * \brief Computes the standard deviation of a vector.
 */
double standard_deviation(const vector<double>& vals)
{
    double variance_val = variance(vals);
    return sqrt(variance_val);
}

/**
 * \brief Counts the number of non nan elements in a 2d vector.
 */
int count_non_nan(const std::vector<std::vector<double>>& vals)
{
    int size = accumulate(
        vals.begin(), vals.end(), 0,
        [](int acc, const vector<double>& val_vector)
        {
            int inner_result = accumulate(
                val_vector.begin(), val_vector.end(), 0, 
                [](int inner_acc, double val)
                {return isnan(val) ? inner_acc : inner_acc + 1;}
            );
            return acc + inner_result;
        }
    );
    return size;
}

/**
 * \brief Subtracts the mean value from a 2d vector.
 */
void subtract_mean(vector_2d& input_array)
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
}
