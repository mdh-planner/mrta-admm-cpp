#include "ScheduleScorer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mrta {

    ScheduleScore ScheduleScorer::scoreScheduleExact(
        const InstanceData& inst,
        const MatrixDouble& z,
        const MatrixDouble& tau,
        const std::vector<VecInt>& ordP,
        const std::vector<VecInt>& ordV,
        const VecInt& lastPhysNode) const
    {
        VecInt endDepot;

        switch (inst.depotPolicy) {
        case DepotPolicy::Free:
            endDepot = chooseEndDepotsAuto(inst, lastPhysNode);
            break;

        case DepotPolicy::Fixed:
            endDepot = inst.endDepotFixed;
            if (static_cast<int>(endDepot.size()) != inst.m) {
                throw std::runtime_error("ScheduleScorer::scoreScheduleExact: fixed depot policy requires endDepotFixed of length m.");
            }
            break;

        default:
            throw std::runtime_error("ScheduleScorer::scoreScheduleExact: unknown depot policy.");
        }

        return computeMakespanToDepots(inst, z, tau, ordP, ordV, lastPhysNode, endDepot);
    }

    VecInt ScheduleScorer::chooseEndDepotsAuto(
        const InstanceData& inst,
        const VecInt& lastPhysNode) const
    {
        if (static_cast<int>(lastPhysNode.size()) != inst.m) {
            throw std::runtime_error("ScheduleScorer::chooseEndDepotsAuto: lastPhysNode must have length m.");
        }

        VecInt endDepot(inst.m, 0);

        for (int s = 0; s < inst.m; ++s) {
            const int from = lastPhysNode[s];

            double bestCost = std::numeric_limits<double>::infinity();
            int bestDepot = -1;

            for (int depot : inst.depotNodes) {
                const double cost = inst.W[from][depot];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestDepot = depot;
                }
            }

            if (bestDepot < 0) {
                throw std::runtime_error("ScheduleScorer::chooseEndDepotsAuto: no depot found.");
            }

            endDepot[s] = bestDepot;
        }

        return endDepot;
    }

    ScheduleScore ScheduleScorer::computeMakespanToDepots(
        const InstanceData& inst,
        const MatrixDouble& zHard,
        const MatrixDouble& tau,
        const std::vector<VecInt>& ordersPhys,
        const std::vector<VecInt>& ordersVirt,
        const VecInt& lastPhysNode,
        const VecInt& endDepot) const
    {
        const int m = inst.m;
        const int n = inst.n;

        VecDouble finishPhysical(m, 0.0);
        VecDouble finishVirtual(m, 0.0);

        for (int s = 0; s < m; ++s) {
            double endP = 0.0;
            for (int j : ordersPhys[s]) {
                if (zHard[s][j] < 0.5) {
                    continue;
                }

                const double st = tau[s][j];
                if (std::isnan(st)) {
                    continue;
                }

                double d = inst.svcPhysSJ[s][j];
                if (!std::isfinite(d)) {
                    d = 0.0;
                }

                endP = std::max(endP, st + d);
            }

            double endV = 0.0;
            for (int j : ordersVirt[s]) {
                if (zHard[s][j] < 0.5) {
                    continue;
                }

                const double st = tau[s][j];
                if (std::isnan(st)) {
                    continue;
                }

                double d = inst.svcVirtSJ[s][j];
                if (!std::isfinite(d)) {
                    d = 0.0;
                }

                endV = std::max(endV, st + d);
            }

            for (int j = 0; j < n; ++j) {
                if (zHard[s][j] < 0.5) {
                    continue;
                }

                const double st = tau[s][j];
                if (std::isnan(st)) {
                    continue;
                }

                if (inst.isVirtual[j]) {
                    double d = inst.svcVirtSJ[s][j];
                    if (!std::isfinite(d)) {
                        d = 0.0;
                    }
                    endV = std::max(endV, st + d);
                }
                else {
                    double d = inst.svcPhysSJ[s][j];
                    if (!std::isfinite(d)) {
                        d = 0.0;
                    }
                    endP = std::max(endP, st + d);
                }
            }

            finishPhysical[s] = endP;
            finishVirtual[s] = endV;
        }

        VecDouble arrivalDepot(m, 0.0);
        for (int s = 0; s < m; ++s) {
            const int from = lastPhysNode[s];
            const int to = endDepot[s];
            const double travel = inst.W[from][to];

            arrivalDepot[s] = std::max(finishPhysical[s] + travel, finishVirtual[s]);
        }

        double mksp = 0.0;
        for (double a : arrivalDepot) {
            mksp = std::max(mksp, a);
        }

        return ScheduleScore{ endDepot, arrivalDepot, mksp };
    }

} // namespace mrta