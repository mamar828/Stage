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
double mean(const vector<vector<double>>& vals)
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
double variance(const vector<vector<double>>& vals)
{
    double mean_val = mean(vals);
    pair<double, int> result = accumulate(
        vals.begin(), vals.end(), make_pair(0.0, 0), [mean_val]
        (pair<double, int> acc, const vector<double>& val_vector)
        {
            pair<double, int> inner_result = accumulate(
                val_vector.begin(), val_vector.end(), make_pair(acc.first, acc.second), [mean_val]
                (pair<double, int> inner_acc, double val)
                {return isnan(val) ? inner_acc : make_pair(inner_acc.first + (val - mean_val) * (val - mean_val), 
                                                           inner_acc.second + 1);}
            );
            return make_pair(inner_result.first, inner_result.second);
            // return make_pair(inner_result.first, acc.second + inner_result.second);
        }
    );
    return result.first / result.second;
}

/**
 * \brief Computes the standard deviation of a vector.
 */
double standard_deviation(const vector<double>& values)
{
    double variance_val = variance(values);
    return sqrt(variance_val);
}
