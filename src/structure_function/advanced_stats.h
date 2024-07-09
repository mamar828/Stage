#include <vector>
#include <array>
#include "stats.h"

void regroup_distance(std::vector<dist_and_regrouped_vals>& regrouped_vals, const std::array<double, 2>& dist_and_val);
std::vector<std::vector<double> > autocorrelation_function(std::vector<std::vector<double> >& input_array);
std::vector<std::vector<double> > structure_function(std::vector<std::vector<double> >& input_array);
