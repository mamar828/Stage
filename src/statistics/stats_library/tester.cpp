#include <iostream>
#include <vector>
#include <limits>

#include "stats.h"

using namespace std;

int main() {
    vector<vector<double>> vec = {
        {1.5, 2.5, 3.5},
        {4.5, 5.5, numeric_limits<double>::quiet_NaN()},
        {7.5, 8.5, 9.5}
    };
    cout << sum(vec) << endl;
    cout << sum_of_squares(vec) << endl;
    cout << count_non_nan(vec) << endl;

    return 0;
}