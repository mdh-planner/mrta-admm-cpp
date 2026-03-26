#include "AssignmentRounder.h"

#include "ScheduleRepairer.h"
#include "ScheduleScorer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <limits>

namespace mrta {

    RepairedAssignmentMakespanEvaluator::RepairedAssignmentMakespanEvaluator(
        const ScheduleRepairer& repairer,
        const ScheduleScorer& scorer)
        : repairer_(repairer), scorer_(scorer) {
    }

    double RepairedAssignmentMakespanEvaluator::evaluate(
        const MatrixDouble& zHard,
        const InstanceData& inst,
        const VecDouble& theta0,
        const VecDouble& t0) const
    {
        RepairOptions opts;
        opts.nIters = 5;
        opts.freezeOrders = false;

        const RepairResult rep = repairer_.repairPushforward(
            zHard,
            inst,
            theta0,
            t0,
            opts
        );

        const ScheduleScore score = scorer_.scoreScheduleExact(
            inst,
            zHard,
            rep.tauFeas,
            rep.ordersPhys,
            rep.ordersVirt,
            rep.lastPhysNode
        );

        return score.mksp;
    }

    AssignmentRounder::AssignmentRounder(const IAssignmentMakespanEvaluator& evaluator)
        : evaluator_(evaluator) {
    }

    RoundingResult AssignmentRounder::roundAndClean(
        const MatrixDouble& z,
        const InstanceData& inst,
        const DerivedData& data,
        const VecDouble& theta,
        const VecDouble& t0) const
    {
        const int m = inst.m;
        const int n = inst.n;

        if (static_cast<int>(z.size()) != m || (m > 0 && static_cast<int>(z.front().size()) != n)) {
            throw std::invalid_argument("AssignmentRounder::roundAndClean: z has wrong dimensions.");
        }

        MatrixDouble zHard(m, VecDouble(n, 0.0));
        VecDouble loadH(m, 0.0);
        VecDouble workH(m, 0.0);

        constexpr double betaLoadRound = 0.10;
        constexpr double betaWorkRound = 0.25;
        constexpr double betaScarceRound = 0.35;

        std::vector<int> taskOrder(n);
        std::iota(taskOrder.begin(), taskOrder.end(), 0);

        std::sort(taskOrder.begin(), taskOrder.end(),
            [&](int a, int b) {
                if (data.taskFlex[a] != data.taskFlex[b]) {
                    return data.taskFlex[a] < data.taskFlex[b];
                }

                double maxZa = 0.0;
                double maxZb = 0.0;
                for (int s = 0; s < m; ++s) {
                    maxZa = std::max(maxZa, z[s][a]);
                    maxZb = std::max(maxZb, z[s][b]);
                }
                return maxZa > maxZb;
            });

        for (int j : taskOrder) {
            std::vector<int> idxCap;
            idxCap.reserve(m);

            for (int s = 0; s < m; ++s) {
                if (inst.cap[s][j] > 0.5) {
                    idxCap.push_back(s);
                }
            }

            if (idxCap.empty()) {
                throw std::runtime_error("AssignmentRounder::roundAndClean: task has no capable robots.");
            }

            int kk = 1;
            if (inst.isMR[j]) {
                kk = std::max(1, inst.k[j]);
            }
            if (inst.isVirtual[j]) {
                kk = 1;
            }
            kk = std::min(kk, static_cast<int>(idxCap.size()));

            std::vector<std::pair<double, int>> scored;
            scored.reserve(idxCap.size());

            for (int s : idxCap) {
                const double score =
                    z[s][j]
                    - betaLoadRound * loadH[s]
                    - betaWorkRound * workH[s]
                    + betaScarceRound * data.scarcityRobot[s];

                scored.emplace_back(score, s);
            }

            std::sort(scored.begin(), scored.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first > rhs.first;
                });

            for (int t = 0; t < kk; ++t) {
                const int sChosen = scored[t].second;
                zHard[sChosen][j] = 1.0;
                loadH[sChosen] += 1.0;
                workH[sChosen] += data.svcEst[sChosen][j];
            }
        }

        std::cout << "\nAssigned tasks per robot (after workload-aware rounding):\n";
        for (int s = 0; s < m; ++s) {
            double assigned = 0.0;
            for (int j = 0; j < n; ++j) {
                assigned += zHard[s][j];
            }
            std::cout << assigned << (s + 1 < m ? ' ' : '\n');
        }

        std::cout << "Estimated service workload per robot after rounding:\n";
        for (int s = 0; s < m; ++s) {
            std::cout << workH[s] << (s + 1 < m ? ' ' : '\n');
        }

        zHard = rebalanceSrByMakespan(zHard, inst, theta, t0, 6, 6, 3);
        zHard = rebalanceMrParticipantsByMakespan(zHard, inst, theta, t0, 4, 8, 4);

        return RoundingResult{ zHard, workH };
    }

    MatrixDouble AssignmentRounder::rebalanceSrByMakespan(
        const MatrixDouble& z0,
        const InstanceData& inst,
        const VecDouble& theta0,
        const VecDouble& t0,
        int nPasses,
        int maxTasksPerHeavyRobot,
        int maxDstPerTask) const
    {
        const int m = inst.m;
        const int n = inst.n;

        if (nPasses <= 0) {
            return z0;
        }

        MatrixDouble zBest = z0;
        const MatrixDouble svcEst = buildLocalServiceEstimate(inst);

        double mkspBest = evaluator_.evaluate(zBest, inst, theta0, t0);

        for (int pass = 0; pass < nPasses; ++pass) {
            VecDouble workH(m, 0.0);
            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    workH[s] += zBest[s][j] * svcEst[s][j];
                }
            }

            std::vector<int> heavyOrder(m);
            std::vector<int> lightOrder(m);
            std::iota(heavyOrder.begin(), heavyOrder.end(), 0);
            std::iota(lightOrder.begin(), lightOrder.end(), 0);

            std::sort(heavyOrder.begin(), heavyOrder.end(),
                [&](int a, int b) { return workH[a] > workH[b]; });

            std::sort(lightOrder.begin(), lightOrder.end(),
                [&](int a, int b) { return workH[a] < workH[b]; });

            double bestDeltaPass = 0.0;
            std::vector<int> bestMove;
            MatrixDouble bestZPass = zBest;
            double bestMkspPass = mkspBest;

            for (int src : heavyOrder) {
                std::vector<int> tasksSrc;
                for (int j = 0; j < n; ++j) {
                    if (zBest[src][j] > 0.5 && !inst.isMR[j] && !inst.isVirtual[j]) {
                        tasksSrc.push_back(j);
                    }
                }

                if (tasksSrc.empty()) {
                    continue;
                }

                std::sort(tasksSrc.begin(), tasksSrc.end(),
                    [&](int a, int b) {
                        return svcEst[src][a] > svcEst[src][b];
                    });

                if (static_cast<int>(tasksSrc.size()) > maxTasksPerHeavyRobot) {
                    tasksSrc.resize(maxTasksPerHeavyRobot);
                }

                for (int j : tasksSrc) {
                    std::vector<int> dstList;
                    for (int dst : lightOrder) {
                        if (dst != src && inst.cap[dst][j] > 0.5) {
                            dstList.push_back(dst);
                        }
                    }

                    if (dstList.empty()) {
                        continue;
                    }

                    if (static_cast<int>(dstList.size()) > maxDstPerTask) {
                        dstList.resize(maxDstPerTask);
                    }

                    for (int dst : dstList) {
                        MatrixDouble zTry = zBest;
                        zTry[src][j] = 0.0;
                        zTry[dst][j] = 1.0;

                        const double mkspTry = evaluator_.evaluate(zTry, inst, theta0, t0);
                        const double delta = mkspBest - mkspTry;

                        if (delta > bestDeltaPass + 1e-9) {
                            bestDeltaPass = delta;
                            bestMove = { src, dst, j };
                            bestZPass = std::move(zTry);
                            bestMkspPass = mkspTry;
                        }
                    }
                }
            }

            if (bestDeltaPass > 1e-9) {
                zBest = std::move(bestZPass);
                mkspBest = bestMkspPass;

                std::cout
                    << "  SR rebalance improved: R" << (bestMove[0] + 1)
                    << " -> R" << (bestMove[1] + 1)
                    << " task " << (bestMove[2] + 1)
                    << " | mksp=" << mkspBest
                    << '\n';
            }
            else {
                break;
            }
        }

        return zBest;
    }

    MatrixDouble AssignmentRounder::rebalanceMrParticipantsByMakespan(
        const MatrixDouble& z0,
        const InstanceData& inst,
        const VecDouble& theta0,
        const VecDouble& t0,
        int nPasses,
        int maxMrTasksPerPass,
        int maxReplacementPairs) const
    {
        const int m = inst.m;
        const int n = inst.n;
        (void)m;

        if (nPasses <= 0) {
            return z0;
        }

        MatrixDouble zBest = z0;
        const MatrixDouble svcEst = buildLocalServiceEstimate(inst);
        double mkspBest = evaluator_.evaluate(zBest, inst, theta0, t0);

        for (int pass = 0; pass < nPasses; ++pass) {
            VecDouble workH(inst.m, 0.0);
            for (int s = 0; s < inst.m; ++s) {
                for (int j = 0; j < n; ++j) {
                    workH[s] += zBest[s][j] * svcEst[s][j];
                }
            }

            std::vector<int> mrTasks;
            for (int j = 0; j < n; ++j) {
                if (inst.isMR[j]) {
                    mrTasks.push_back(j);
                }
            }

            if (mrTasks.empty()) {
                break;
            }

            std::vector<std::pair<double, int>> prioritized;
            prioritized.reserve(mrTasks.size());

            for (int j : mrTasks) {
                double prio = -std::numeric_limits<double>::infinity();
                bool found = false;

                for (int s = 0; s < inst.m; ++s) {
                    if (zBest[s][j] > 0.5) {
                        prio = found ? std::max(prio, workH[s]) : workH[s];
                        found = true;
                    }
                }

                if (!found) {
                    prio = -std::numeric_limits<double>::infinity();
                }

                prioritized.emplace_back(prio, j);
            }

            std::sort(prioritized.begin(), prioritized.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

            mrTasks.clear();
            for (const auto& p : prioritized) {
                mrTasks.push_back(p.second);
            }

            if (static_cast<int>(mrTasks.size()) > maxMrTasksPerPass) {
                mrTasks.resize(maxMrTasksPerPass);
            }

            double bestDeltaPass = 0.0;
            std::vector<int> bestMove;
            MatrixDouble bestZPass = zBest;
            double bestMkspPass = mkspBest;

            for (int j : mrTasks) {
                std::vector<int> pCur;
                std::vector<int> q;

                for (int s = 0; s < inst.m; ++s) {
                    if (zBest[s][j] > 0.5) {
                        pCur.push_back(s);
                    }
                    else if (inst.cap[s][j] > 0.5) {
                        q.push_back(s);
                    }
                }

                if (pCur.empty() || q.empty()) {
                    continue;
                }

                std::sort(pCur.begin(), pCur.end(),
                    [&](int a, int b) {
                        return workH[a] > workH[b];
                    });

                std::sort(q.begin(), q.end(),
                    [&](int a, int b) {
                        return workH[a] < workH[b];
                    });

                int pairCount = 0;
                bool stopTask = false;

                for (int src : pCur) {
                    for (int dst : q) {
                        ++pairCount;
                        if (pairCount > maxReplacementPairs) {
                            stopTask = true;
                            break;
                        }

                        MatrixDouble zTry = zBest;
                        zTry[src][j] = 0.0;
                        zTry[dst][j] = 1.0;

                        const double mkspTry = evaluator_.evaluate(zTry, inst, theta0, t0);
                        const double delta = mkspBest - mkspTry;

                        if (delta > bestDeltaPass + 1e-9) {
                            bestDeltaPass = delta;
                            bestMove = { j, src, dst };
                            bestZPass = std::move(zTry);
                            bestMkspPass = mkspTry;
                        }
                    }

                    if (stopTask) {
                        break;
                    }
                }
            }

            if (bestDeltaPass > 1e-9) {
                zBest = std::move(bestZPass);
                mkspBest = bestMkspPass;

                std::cout
                    << "  MR participant replacement improved: task " << (bestMove[0] + 1)
                    << " | out R" << (bestMove[1] + 1)
                    << " in R" << (bestMove[2] + 1)
                    << " | mksp=" << mkspBest
                    << '\n';
            }
            else {
                break;
            }
        }

        return zBest;
    }

    MatrixDouble AssignmentRounder::buildLocalServiceEstimate(
        const InstanceData& inst)
    {
        const int m = inst.m;
        const int n = inst.n;

        MatrixDouble svcEst(m, VecDouble(n, 0.0));

        for (int s = 0; s < m; ++s) {
            for (int j = 0; j < n; ++j) {
                if (inst.cap[s][j] < 0.5) {
                    svcEst[s][j] = 0.0;
                    continue;
                }

                if (inst.isVirtual[j]) {
                    double d = inst.svcVirtSJ[s][j];
                    if (!std::isfinite(d) || d < 0.0) {
                        d = 0.0;
                    }
                    svcEst[s][j] = d;
                }
                else if (inst.isMR[j]) {
                    double best = 0.0;
                    bool found = false;
                    for (int r = 0; r < m; ++r) {
                        const double d = inst.svcPhysSJ[r][j];
                        if (std::isfinite(d) && d >= 0.0) {
                            best = found ? std::max(best, d) : d;
                            found = true;
                        }
                    }
                    svcEst[s][j] = found ? best : 0.0;
                }
                else {
                    double d = inst.svcPhysSJ[s][j];
                    if (!std::isfinite(d) || d < 0.0) {
                        d = 0.0;
                    }
                    svcEst[s][j] = d;
                }
            }
        }

        return svcEst;
    }

} // namespace mrta