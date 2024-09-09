#include <array>
#include <vector>
#include <unordered_map>
#include <functional>

/**
 * \struct DoubleArrayHash
 * \brief Enables the use of unordered_map with array keys.
 */
struct DoubleArrayHash
{
    size_t operator()(const std::array<double, 2>& arr) const
    {
        size_t seed = 0;
        for (double val : arr)
        {
            seed ^= std::hash<double>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

typedef std::vector<std::vector<double>> vector_2d;
typedef std::unordered_map<double, std::vector<double>> double_unordered_map;
typedef std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash> array_unordered_map;

void regroup_distance(double_unordered_map& regrouped_vals,
                      const std::array<double, 2>& dist_and_val);
void regroup_distance(std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash>& regrouped_vals,
                      const std::array<double, 3>& dist_and_val);
void combine_vectors(std::vector<std::array<double,2>>& dest, const std::vector<std::array<double,2>>& src);
void combine_vectors(std::vector<std::array<double,3>>& dest, const std::vector<std::array<double,3>>& src);

template <typename T>
std::vector<std::array<double, 2>> apply_vector_map(const vector_2d& input_array, const T& function);
std::vector<std::array<double, 2>> multiply_elements(const vector_2d& input_array);
std::vector<std::array<double, 2>> subtract_elements(const vector_2d& input_array);
