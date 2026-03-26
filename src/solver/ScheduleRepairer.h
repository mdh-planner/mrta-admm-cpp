#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/InstanceData.h"

#include <optional>

namespace mrta {

    struct RepairOptions {
        int nIters{ 10 };
        bool freezeOrders{ false };

        std::optional<std::vector<VecInt>> ordersPhys0;
        std::optional<std::vector<VecInt>> ordersVirt0;
        std::optional<VecDouble> virtPinnedStart;
    };

    struct RepairResult {
        MatrixDouble tauFeas;
        VecDouble thetaFeas;
        VecDouble tFeas;
        std::vector<VecInt> ordersPhys;
        std::vector<VecInt> ordersVirt;
        VecInt lastPhysNode;
    };

    class ScheduleRepairer {
    public:
        RepairResult repairPushforward(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& thetaInit,
            const VecDouble& tInit,
            const RepairOptions& options = {}) const;

    private:
        static std::vector<VecInt> normalizeOrders(
            const std::optional<std::vector<VecInt>>& orders,
            int m);

        static double safeTravel(const MatrixDouble& W, int from, int to);
        static double maxFiniteInColumn(const MatrixDouble& M, int j);
    };

} // namespace mrta