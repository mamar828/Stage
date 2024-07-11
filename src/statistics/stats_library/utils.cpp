#include "utils.h"
#include <iostream>

using namespace std;

void print(const vector<vector<double>>& input_vector)
{
    for (const auto& row : input_vector) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << "\n";
    }
}

void fill_vector(vector<vector<double>>& input_vector, const vector<double>& values)
{
    for (size_t j = 0; j < input_vector.size(); j++)
    {
        for (size_t i = 0; i < input_vector[0].size(); i++)
        {
            input_vector[j][i] = values[j*input_vector.size() + i];
        }
    }
}
