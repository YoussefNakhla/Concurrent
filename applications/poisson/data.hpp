#pragma once
#include "model.hpp"
#include <cstdint>

// doctor visits data generation
Dataset generate_data(std::size_t N, std::uint64_t seed = 42);
