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

void regroup_distance(std::unordered_map<double, std::vector<double>>& regrouped_vals,
                      const std::array<double, 2>& dist_and_val);
void regroup_distance(std::unordered_map<std::array<double, 2>, std::vector<double>, DoubleArrayHash>& regrouped_vals,
                      const std::array<double, 3>& dist_and_val);

template <typename T>
std::vector<std::array<double, 2>> apply_vector_map(const std::vector<std::vector<double>>& input_array,
                                                    const T& function);
std::vector<std::array<double, 2>> multiply_elements(const std::vector<std::vector<double>>& input_array);
std::vector<std::array<double, 2>> subtract_elements(const std::vector<std::vector<double>>& input_array);
