#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/InstanceData.h"

namespace mrta {

    struct ScheduleScore {
        VecInt endDepot;
        VecDouble arrival;
        double mksp{ 0.0 };
    };

    class ScheduleScorer {
    public:
        ScheduleScore scoreScheduleExact(
            const InstanceData& inst,
            const MatrixDouble& z,
            const MatrixDouble& tau,
            const std::vector<VecInt>& ordP,
            const std::vector<VecInt>& ordV,
            const VecInt& lastPhysNode) const;

    private:
        VecInt chooseEndDepotsAuto(
            const InstanceData& inst,
            const VecInt& lastPhysNode) const;

        ScheduleScore computeMakespanToDepots(
            const InstanceData& inst,
            const MatrixDouble& zHard,
            const MatrixDouble& tau,
            const std::vector<VecInt>& ordersPhys,
            const std::vector<VecInt>& ordersVirt,
            const VecInt& lastPhysNode,
            const VecInt& endDepot) const;
    };

} // namespace mrta