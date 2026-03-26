#pragma once

#include "CommonTypes.h"

namespace mrta {

    struct SolutionState {
        MatrixDouble z;
        MatrixDouble tau;
        VecDouble theta;

        std::vector<VecInt> ordersPhys;
        std::vector<VecInt> ordersVirt;

        VecInt endDepot;
        double makespan{};

        [[nodiscard]] bool empty() const noexcept {
            return z.empty();
        }
    };

    struct AdmmResult {
        MatrixDouble z;
        VecDouble theta;
        bool converged{ false };
        VecDouble histCard;
        VecDouble histSync;
    };

} // namespace mrta