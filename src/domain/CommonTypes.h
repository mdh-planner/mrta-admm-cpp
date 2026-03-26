#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mrta {

    using Index = int;
    using VecInt = std::vector<int>;
    using VecDouble = std::vector<double>;
    using BoolVec = std::vector<bool>;
    using MatrixDouble = std::vector<std::vector<double>>;
    using MatrixInt = std::vector<std::vector<int>>;
    using MatrixBool = std::vector<std::vector<bool>>;

    enum class DepotPolicy {
        Fixed,
        Free
    };

    struct PrecedenceEdge {
        Index from{};
        Index to{};
    };

    inline std::string toString(DepotPolicy policy) {
        switch (policy) {
        case DepotPolicy::Fixed: return "fixed";
        case DepotPolicy::Free:  return "free";
        default: throw std::runtime_error("Unknown DepotPolicy.");
        }
    }

} // namespace mrta