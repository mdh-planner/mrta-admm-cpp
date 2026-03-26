#include "ScheduleRepairer.h"

#include "PrecedenceGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mrta {

    std::vector<VecInt> ScheduleRepairer::normalizeOrders(
        const std::optional<std::vector<VecInt>>& orders,
        int m)
    {
        std::vector<VecInt> out(m);
        if (!orders.has_value()) {
            return out;
        }

        out = *orders;
        if (static_cast<int>(out.size()) < m) {
            out.resize(m);
        }
        else if (static_cast<int>(out.size()) > m) {
            out.resize(m);
        }
        return out;
    }

    double ScheduleRepairer::safeTravel(const MatrixDouble& W, int from, int to) {
        const double v = W[from][to];
        return std::isfinite(v) ? v : 0.0;
    }

    double ScheduleRepairer::maxFiniteInColumn(const MatrixDouble& M, int j) {
        double best = 0.0;
        bool found = false;
        for (const auto& row : M) {
            const double v = row[j];
            if (std::isfinite(v)) {
                best = found ? std::max(best, v) : v;
                found = true;
            }
        }
        return found ? best : 0.0;
    }

    RepairResult ScheduleRepairer::repairPushforward(
        const MatrixDouble& zHard,
        const InstanceData& inst,
        const VecDouble& thetaInit,
        const VecDouble& tInit,
        const RepairOptions& options) const
    {
        const int m = inst.m;
        const int n = inst.n;

        if (static_cast<int>(zHard.size()) != m || (m > 0 && static_cast<int>(zHard.front().size()) != n)) {
            throw std::invalid_argument("ScheduleRepairer::repairPushforward: zHard has wrong dimensions.");
        }
        if (static_cast<int>(thetaInit.size()) != n || static_cast<int>(tInit.size()) != n) {
            throw std::invalid_argument("ScheduleRepairer::repairPushforward: thetaInit/tInit have wrong dimensions.");
        }

        const int nIters = std::max(1, options.nIters);
        const double repairTolTime = 1e-6;
        const int repairMinIters = 2;

        MatrixDouble tauFeas(m, VecDouble(n, std::numeric_limits<double>::quiet_NaN()));
        VecDouble thetaFeas = thetaInit;
        VecDouble tFeas = tInit;

        BoolVec iv = inst.isVirtual;
        BoolVec isMR = inst.isMR;

        VecDouble virtPinnedStart(n, std::numeric_limits<double>::quiet_NaN());
        if (options.virtPinnedStart.has_value()) {
            if (static_cast<int>(options.virtPinnedStart->size()) != n) {
                throw std::invalid_argument("ScheduleRepairer::repairPushforward: virtPinnedStart must have length n.");
            }
            virtPinnedStart = *options.virtPinnedStart;
        }

        // ------------------------------------------------------------
        // Precompute
        // ------------------------------------------------------------
        VecInt taskNode(n, 0);
        for (int j = 0; j < n; ++j) {
            taskNode[j] = m + j;
        }

        VecInt startNode(m, 0);
        for (int s = 0; s < m; ++s) {
            startNode[s] = s;
        }

        MatrixDouble svcDur(m, VecDouble(n, 0.0));
        VecDouble durTask(n, 0.0);

        for (int j = 0; j < n; ++j) {
            if (iv[j]) {
                durTask[j] = maxFiniteInColumn(inst.svcVirtSJ, j);
            }
            else {
                durTask[j] = maxFiniteInColumn(inst.svcPhysSJ, j);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (isMR[j]) {
                for (int s = 0; s < m; ++s) {
                    svcDur[s][j] = durTask[j];
                }
            }
            else {
                for (int s = 0; s < m; ++s) {
                    double d = iv[j] ? inst.svcVirtSJ[s][j] : inst.svcPhysSJ[s][j];
                    if (!std::isfinite(d) || d < 0.0) {
                        d = durTask[j];
                    }
                    if (!std::isfinite(d) || d < 0.0) {
                        d = 0.0;
                    }
                    svcDur[s][j] = d;
                }
            }
        }

        VecDouble robotRelease(m, 0.0);
        if (!inst.Tstart.empty()) {
            // Instance loader currently stores Tstart as m x n
            for (int s = 0; s < m; ++s) {
                double best = std::numeric_limits<double>::infinity();
                for (int j = 0; j < n; ++j) {
                    const double v = inst.Tstart[s][j];
                    if (std::isfinite(v)) {
                        best = std::min(best, v);
                    }
                }
                robotRelease[s] = std::isfinite(best) ? best : 0.0;
            }
        }

        std::vector<VecInt> taskRobots(n);
        for (int j = 0; j < n; ++j) {
            for (int s = 0; s < m; ++s) {
                if (zHard[s][j] > 0.5) {
                    taskRobots[j].push_back(s);
                }
            }
        }

        // ------------------------------------------------------------
        // Initial orders
        // ------------------------------------------------------------
        std::vector<VecInt> ordersPhys(m);
        std::vector<VecInt> ordersVirt(m);

        const bool doFreeze =
            options.freezeOrders &&
            options.ordersPhys0.has_value() &&
            options.ordersVirt0.has_value();

        if (doFreeze) {
            ordersPhys = normalizeOrders(options.ordersPhys0, m);
            ordersVirt = normalizeOrders(options.ordersVirt0, m);
        }
        else {
            for (int s = 0; s < m; ++s) {
                VecInt phys;
                VecInt virt;

                for (int j = 0; j < n; ++j) {
                    if (zHard[s][j] > 0.5) {
                        if (iv[j]) {
                            virt.push_back(j);
                        }
                        else {
                            phys.push_back(j);
                        }
                        tauFeas[s][j] = std::max(0.0, tFeas[j]);
                    }
                }

                std::sort(phys.begin(), phys.end(), [&](int a, int b) {
                    return tFeas[a] < tFeas[b];
                    });
                std::sort(virt.begin(), virt.end(), [&](int a, int b) {
                    return tFeas[a] < tFeas[b];
                    });

                ordersPhys[s] = std::move(phys);
                ordersVirt[s] = std::move(virt);
            }
        }

        PrecedenceGraph graph(n, inst.predPairs);

        // ------------------------------------------------------------
        // Main iteration loop
        // ------------------------------------------------------------
        for (int it = 0; it < nIters; ++it) {
            const MatrixDouble tauPrev = tauFeas;
            const VecDouble thetaPrev = thetaFeas;
            const auto ordersPhysPrev = ordersPhys;
            const auto ordersVirtPrev = ordersVirt;

            // ======================
            // completion times C(j)
            // ======================
            VecDouble C(n, 0.0);
            for (int j = 0; j < n; ++j) {
                const auto& P = taskRobots[j];
                if (P.empty()) {
                    continue;
                }

                if (isMR[j]) {
                    double tj = 0.0;
                    bool found = false;
                    for (int s : P) {
                        const double v = tauFeas[s][j];
                        if (std::isfinite(v)) {
                            tj = found ? std::max(tj, v) : v;
                            found = true;
                        }
                    }
                    if (!found) {
                        tj = 0.0;
                    }
                    C[j] = tj + durTask[j];
                }
                else {
                    const int s = P.front();
                    const double tjs = tauFeas[s][j];
                    if (std::isfinite(tjs)) {
                        C[j] = tjs + svcDur[s][j];
                    }
                }
            }

            // ======================
            // precedence EST
            // ======================
            VecDouble EST(n, 0.0);
            for (const auto& e : inst.predPairs) {
                EST[e.to] = std::max(EST[e.to], C[e.from]);
            }

            // ======================
            // (A) physical lane ASAP-tight
            // ======================
            for (int s = 0; s < m; ++s) {
                const auto& ord = ordersPhys[s];
                if (ord.empty()) {
                    continue;
                }

                const double rel = robotRelease[s];
                const int nPrev0 = startNode[s];

                const int j1 = ord.front();
                const int n1 = taskNode[j1];
                const double tr01 = safeTravel(inst.W, nPrev0, n1);

                const double arrJ1 = rel + tr01;
                tauFeas[s][j1] = std::max(EST[j1], arrJ1);

                for (std::size_t kk = 1; kk < ord.size(); ++kk) {
                    const int jPrev = ord[kk - 1];
                    const int jCurr = ord[kk];

                    double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
                    if (!std::isfinite(tPrevStart)) {
                        tPrevStart = 0.0;
                    }

                    const double svcPrev = isMR[jPrev] ? durTask[jPrev] : svcDur[s][jPrev];
                    const double tPrevEnd = tPrevStart + svcPrev;

                    const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);
                    const double earliest = tPrevEnd + tr;

                    tauFeas[s][jCurr] = std::max(EST[jCurr], earliest);
                }
            }

            // ======================
            // (B) MR sync
            // ======================
            for (int j = 0; j < n; ++j) {
                if (!isMR[j]) {
                    continue;
                }

                const auto& P = taskRobots[j];
                if (P.empty()) {
                    continue;
                }

                double tj = EST[j];
                for (int s : P) {
                    const double v = tauFeas[s][j];
                    if (std::isfinite(v) && v > tj) {
                        tj = v;
                    }
                }

                thetaFeas[j] = tj;
                for (int s : P) {
                    tauFeas[s][j] = tj;
                }
            }

            // ======================
            // (B2) re-push physical lane after MR sync
            // ======================
            for (int s = 0; s < m; ++s) {
                const auto& ord = ordersPhys[s];
                if (ord.empty()) {
                    continue;
                }

                const double rel = robotRelease[s];
                const int nPrev0 = startNode[s];

                const int j1 = ord.front();
                const int n1 = taskNode[j1];
                const double tr01 = safeTravel(inst.W, nPrev0, n1);

                tauFeas[s][j1] = std::max(EST[j1], rel + tr01);
                if (isMR[j1]) {
                    tauFeas[s][j1] = thetaFeas[j1];
                }

                for (std::size_t kk = 1; kk < ord.size(); ++kk) {
                    const int jPrev = ord[kk - 1];
                    const int jCurr = ord[kk];

                    double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
                    if (!std::isfinite(tPrevStart)) {
                        tPrevStart = 0.0;
                    }

                    const double svcPrev = isMR[jPrev] ? durTask[jPrev] : svcDur[s][jPrev];
                    const double tPrevEnd = tPrevStart + svcPrev;

                    const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);
                    const double earliest = tPrevEnd + tr;

                    tauFeas[s][jCurr] = std::max(EST[jCurr], earliest);
                    if (isMR[jCurr]) {
                        tauFeas[s][jCurr] = thetaFeas[jCurr];
                    }
                }
            }

            // ======================
            // (C) Virtual lane scheduling
            // ======================
            for (int s = 0; s < m; ++s) {
                const auto& ordV = ordersVirt[s];
                if (ordV.empty()) {
                    continue;
                }

                const auto& physOrd = ordersPhys[s];
                const int nPhys = static_cast<int>(physOrd.size());

                VecDouble physStart(nPhys, 0.0);
                VecDouble physEnd(nPhys, 0.0);

                for (int pp = 0; pp < nPhys; ++pp) {
                    const int jp = physOrd[pp];
                    double sp = tauFeas[s][jp];
                    if (!std::isfinite(sp)) {
                        sp = 0.0;
                    }
                    physStart[pp] = sp;
                    physEnd[pp] = sp + svcDur[s][jp];
                }

                VecInt scheduledV;
                VecDouble scheduledVStart;
                VecDouble scheduledVEnd;

                for (int jv : ordV) {
                    double tStart = EST[jv];
                    if (std::isfinite(virtPinnedStart[jv])) {
                        tStart = std::max(tStart, virtPinnedStart[jv]);
                    }
                    const double dv = svcDur[s][jv];

                    // Avoid overlap with scheduled virtuals
                    for (std::size_t qq = 0; qq < scheduledV.size(); ++qq) {
                        const int q = scheduledV[qq];
                        if (inst.Rpar[jv][q] == 1.0) {
                            continue;
                        }
                        const double tq = scheduledVStart[qq];
                        const double eq = scheduledVEnd[qq];

                        if (!(tStart + dv <= tq || tStart >= eq)) {
                            tStart = eq;
                        }
                    }

                    // Avoid overlap with physicals
                    bool pushed = true;
                    while (pushed) {
                        pushed = false;
                        for (int pp = 0; pp < nPhys; ++pp) {
                            const int jp = physOrd[pp];
                            if (inst.Rpar[jv][jp] == 1.0) {
                                continue;
                            }

                            const double sp = physStart[pp];
                            const double ep = physEnd[pp];

                            if (!(tStart + dv <= sp || tStart >= ep)) {
                                tStart = ep;
                                pushed = true;
                            }
                        }
                    }

                    if (tStart < 0.0) {
                        tStart = 0.0;
                    }

                    tauFeas[s][jv] = tStart;
                    scheduledV.push_back(jv);
                    scheduledVStart.push_back(tStart);
                    scheduledVEnd.push_back(tStart + dv);
                }
            }

            // ======================
            // (D) update global t
            // ======================
            VecDouble tBar = tFeas;
            for (int j = 0; j < n; ++j) {
                const auto& P = taskRobots[j];
                if (P.empty()) {
                    continue;
                }

                if (isMR[j]) {
                    tBar[j] = thetaFeas[j];
                }
                else {
                    tBar[j] = tauFeas[P.front()][j];
                }
            }
            tFeas = graph.pushForward(tBar, durTask);

            // Reorder if not frozen
            if (!doFreeze) {
                for (int s = 0; s < m; ++s) {
                    auto& phys = ordersPhys[s];
                    std::sort(phys.begin(), phys.end(), [&](int a, int b) {
                        double sa = isMR[a] ? thetaFeas[a] : tauFeas[s][a];
                        double sb = isMR[b] ? thetaFeas[b] : tauFeas[s][b];
                        if (!std::isfinite(sa)) sa = std::numeric_limits<double>::infinity();
                        if (!std::isfinite(sb)) sb = std::numeric_limits<double>::infinity();
                        return sa < sb;
                        });

                    auto& virt = ordersVirt[s];
                    std::sort(virt.begin(), virt.end(), [&](int a, int b) {
                        return tauFeas[s][a] < tauFeas[s][b];
                        });
                }
            }

            // ======================
            // Convergence
            // ======================
            double tauDiff = 0.0;
            bool tauInfMismatch = false;

            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    const bool curFin = std::isfinite(tauFeas[s][j]);
                    const bool prevFin = std::isfinite(tauPrev[s][j]);

                    if (curFin && prevFin) {
                        tauDiff = std::max(tauDiff, std::abs(tauFeas[s][j] - tauPrev[s][j]));
                    }
                    else if (curFin != prevFin) {
                        tauInfMismatch = true;
                    }
                }
            }
            if (tauInfMismatch) {
                tauDiff = std::numeric_limits<double>::infinity();
            }

            double thetaDiff = 0.0;
            for (int j = 0; j < n; ++j) {
                thetaDiff = std::max(thetaDiff, std::abs(thetaFeas[j] - thetaPrev[j]));
            }
            if (!std::isfinite(thetaDiff)) {
                thetaDiff = std::numeric_limits<double>::infinity();
            }

            bool sameOrders = true;
            for (int s = 0; s < m; ++s) {
                if (ordersPhys[s] != ordersPhysPrev[s] || ordersVirt[s] != ordersVirtPrev[s]) {
                    sameOrders = false;
                    break;
                }
            }

            if (it + 1 >= repairMinIters &&
                tauDiff <= repairTolTime &&
                thetaDiff <= repairTolTime &&
                sameOrders) {
                break;
            }
        }

        VecInt lastPhysNode(m, 0);
        for (int s = 0; s < m; ++s) {
            if (ordersPhys[s].empty()) {
                lastPhysNode[s] = s;
            }
            else {
                lastPhysNode[s] = taskNode[ordersPhys[s].back()];
            }
        }

        return RepairResult{
            tauFeas,
            thetaFeas,
            tFeas,
            ordersPhys,
            ordersVirt,
            lastPhysNode
        };
    }

} // namespace mrta