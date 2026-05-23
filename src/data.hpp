#pragma once
#include "model.hpp"
#include <cstdint>


// Header file for the synthetic data generation (mimicking the setup from the paper)
Dataset generate_data(std::size_t N, std::uint64_t seed = 12345);
