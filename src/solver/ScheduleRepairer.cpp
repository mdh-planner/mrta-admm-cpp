#include "ScheduleRepairer.h"
#include "PrecedenceGraph.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace mrta {

    // ============================================================================
    // Shared static helpers
    // ============================================================================

    std::vector<VecInt> ScheduleRepairer::normalizeOrders(
        const std::optional<std::vector<VecInt>>& orders, int m)
    {
        std::vector<VecInt> out(m);
        if (!orders.has_value()) { return out; }
        out = *orders;
        if (static_cast<int>(out.size()) < m) { out.resize(m); }
        else if (static_cast<int>(out.size()) > m) { out.resize(m); }
        return out;
    }

    double ScheduleRepairer::safeTravel(const MatrixDouble& W, int from, int to)
    {
        const double v = W[from][to];
        return std::isfinite(v) ? v : 0.0;
    }

    double ScheduleRepairer::maxFiniteInColumn(const MatrixDouble& M, int j)
    {
        double best = 0.0;
        bool   found = false;
        for (const auto& row : M) {
            const double v = row[j];
            if (std::isfinite(v)) {
                best = found ? std::max(best, v) : v;
                found = true;
            }
        }
        return found ? best : 0.0;
    }

    // v2-only helper — mirrors push_past_pinned_virtuals() in pushforward2.m
    double ScheduleRepairer::pushPastPinnedVirtuals(
        int                 s,
        int                 j,
        double              t,
        const MatrixDouble& tauFeas,
        const VecInt& ordVirt,
        const VecDouble& virtPinnedStart,
        const MatrixDouble& svcDur,
        const MatrixDouble& Rpar)
    {
        const double dj = svcDur[s][j];
        bool moved = true;
        while (moved) {
            moved = false;
            for (int jv : ordVirt) {
                if (!std::isfinite(virtPinnedStart[jv])) { continue; }
                const double tv = tauFeas[s][jv];
                if (!std::isfinite(tv)) { continue; }
                if (Rpar[jv][j] == 1.0) { continue; }
                const double ev = tv + svcDur[s][jv];
                if (!(t + dj <= tv || t >= ev)) {
                    t = ev;
                    moved = true;
                }
            }
        }
        return t;
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

        const int    nIters = std::max(1, options.nIters);
        const bool   RANDOMIZE_TIES = true;
        const double TIE_EPS = 1e-6;
        const int REPAIR_MIN_ITERS = 3;

        const BoolVec& iv = inst.isVirtual;
        const BoolVec& isMR = inst.isMR;

        VecDouble virtPinnedStart(n, std::numeric_limits<double>::quiet_NaN());
        if (options.virtPinnedStart.has_value()) {
            if (static_cast<int>(options.virtPinnedStart->size()) != n) {
                throw std::invalid_argument(
                    "repairPushforward2: virtPinnedStart must have length n.");
            }
            virtPinnedStart = *options.virtPinnedStart;
        }

        MatrixDouble tauFeas(m, VecDouble(n, std::numeric_limits<double>::quiet_NaN()));
        VecDouble    thetaFeas = thetaInit;
        VecDouble tFeas = (options.tFeasHint.has_value())
            ? *options.tFeasHint
            : tInit;

        // TEMP DIAGNOSTIC
        if (options.tFeasHint.has_value()) {
            std::cout << "[Repair] tFeasHint active, values: ";
            for (int j = 0; j < n; ++j) std::cout << tFeas[j] << " ";
            std::cout << "\n";
        }

        // ------------------------------------------------------------------
        // startNode, durTask, svcDur  (same structure as v1)
        // ------------------------------------------------------------------
        VecInt taskNode(n);
        for (int j = 0; j < n; ++j) { taskNode[j] = m + j; }

        VecInt startNode(m);
        for (int s = 0; s < m; ++s) { startNode[s] = s; }

        VecDouble durTask(n, 0.0);
        for (int j = 0; j < n; ++j) {
            durTask[j] = iv[j] ? maxFiniteInColumn(inst.svcVirtSJ, j)
                : maxFiniteInColumn(inst.svcPhysSJ, j);
        }

        // service_dur(s,j)  — v2 uses a nested function; replicate inline
        // Same logic as v1 except accessed via lambda to mirror MATLAB's nested fn
        auto serviceDur = [&](int s, int j) -> double {
            double d;
            if (isMR[j]) {
                d = durTask[j];
            }
            else {
                d = iv[j] ? inst.svcVirtSJ[s][j] : inst.svcPhysSJ[s][j];
                if (!std::isfinite(d) || d < 0.0) { d = durTask[j]; }
            }
            if (!std::isfinite(d) || d < 0.0) { d = 0.0; }
            return d;
            };

        // Build svcDur matrix using serviceDur (for pushPastPinnedVirtuals)
        MatrixDouble svcDur(m, VecDouble(n, 0.0));
        for (int s = 0; s < m; ++s) {
            for (int j = 0; j < n; ++j) { svcDur[s][j] = serviceDur(s, j); }
        }

        // getRobotRelease(s) — v2 uses same logic as v1
        auto getRobotRelease = [&](int s) -> double {
            if (inst.Tstart.empty()) { return 0.0; }
            if (inst.Tstart.size() == 1 && inst.Tstart[0].size() == 1) {
                const double v = inst.Tstart[0][0];
                return std::isfinite(v) ? v : 0.0;
            }
            double best = std::numeric_limits<double>::infinity();
            for (const double v : inst.Tstart[s]) {
                if (std::isfinite(v)) { best = std::min(best, v); }
            }
            return std::isfinite(best) ? best : 0.0;
            };

        // taskRobots
        std::vector<VecInt> taskRobots(n);
        for (int j = 0; j < n; ++j) {
            for (int s = 0; s < m; ++s) {
                if (zHard[s][j] > 0.5) { taskRobots[j].push_back(s); }
            }
        }

        // ------------------------------------------------------------------
        // Initial orders
        // ------------------------------------------------------------------
        std::vector<VecInt> ordersPhys(m);
        std::vector<VecInt> ordersVirt(m);

        const bool doFreeze =
            options.freezeOrders &&
            options.ordersPhys0.has_value() &&
            options.ordersVirt0.has_value();

        const bool useHint =        // ← NEW: warm-start without freezing
            !options.freezeOrders &&
            options.ordersPhys0.has_value() &&
            options.ordersVirt0.has_value();

        if (doFreeze) {
            ordersPhys = normalizeOrders(options.ordersPhys0, m);
            ordersVirt = normalizeOrders(options.ordersVirt0, m);
        }
        else if (useHint) {         // ← NEW: use hint as starting order, allow re-sorting
            ordersPhys = normalizeOrders(options.ordersPhys0, m);
            ordersVirt = normalizeOrders(options.ordersVirt0, m);
            // Initialize tauFeas from tFeas just like the unfrozen path
            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    if (zHard[s][j] > 0.5) {
                        tauFeas[s][j] = std::max(0.0, tFeas[j]);
                    }
                }
            }
            // Don't sort — the hint order is the starting point.
            // The main iteration loop will re-sort after computing timing.
        }
        else {
            for (int s = 0; s < m; ++s) {
                VecInt phys, virt;
                for (int j = 0; j < n; ++j) {
                    if (zHard[s][j] > 0.5) {
                        if (iv[j]) { virt.push_back(j); }
                        else { phys.push_back(j); }
                        tauFeas[s][j] = std::max(0.0, tFeas[j]);
                    }
                }
                std::sort(phys.begin(), phys.end(),
                    [&](int a, int b) { return tFeas[a] < tFeas[b]; });
                std::sort(virt.begin(), virt.end(),
                    [&](int a, int b) { return tFeas[a] < tFeas[b]; });
                ordersPhys[s] = std::move(phys);
                ordersVirt[s] = std::move(virt);
            }
        }

        PrecedenceGraph graph(n, inst.predPairs);

        // Random engine for tie-breaking (RANDOMIZE_TIES = true)
        std::mt19937 rng(std::random_device{}());

        // ------------------------------------------------------------------
        // Main iteration loop
        // ------------------------------------------------------------------

        double mkspPrev_ = std::numeric_limits<double>::infinity();

        for (int it = 0; it < nIters; ++it) {

      /*      const MatrixDouble tauPrev = tauFeas;
            const VecDouble    thetaPrev = thetaFeas;
            const auto         ordPhysPrev = ordersPhys;
            const auto         ordVirtPrev = ordersVirt;*/

            // ==================== C(j) ====================
            VecDouble C(n, 0.0);
            for (int j = 0; j < n; ++j) {
                const auto& P = taskRobots[j];
                if (P.empty()) { continue; }
                if (isMR[j]) {
                    double tj = 0.0; bool found = false;
                    for (int s : P) {
                        const double v = tauFeas[s][j];
                        if (std::isfinite(v)) { tj = found ? std::max(tj, v) : v; found = true; }
                    }
                    if (!found) { tj = 0.0; }
                    C[j] = tj + durTask[j];
                }
                else {
                    const int s = P.front();
                    if (std::isfinite(tauFeas[s][j])) {
                        C[j] = tauFeas[s][j] + serviceDur(s, j);
                    }
                }
            }

            // ==================== EST ====================
            VecDouble EST(n, 0.0);
            for (const auto& e : inst.predPairs) {
                EST[e.to] = std::max(EST[e.to], C[e.from]);
            }

            // ==================== Pre-set pinned virtual times (v2 only) ====================
            // MATLAB: tau_feas(s,jv) = max(EST(jv), virtPinnedStart(jv))
            for (int s = 0; s < m; ++s) {
                for (int jv : ordersVirt[s]) {
                    if (std::isfinite(virtPinnedStart[jv])) {
                        tauFeas[s][jv] = std::max(EST[jv], virtPinnedStart[jv]);
                    }
                }
            }

            // ==================== (A) physical lane ASAP + push_past_pinned ====================
            for (int s = 0; s < m; ++s) {
                const auto& ord = ordersPhys[s];
                if (ord.empty()) { continue; }

                const double rel = getRobotRelease(s);
                const int    j1 = ord[0];
                const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);

                double earliest = std::max(EST[j1], rel + tr01);
                earliest = pushPastPinnedVirtuals(
                    s, j1, earliest, tauFeas, ordersVirt[s],
                    virtPinnedStart, svcDur, inst.Rpar);
                tauFeas[s][j1] = earliest;

                for (std::size_t kk = 1; kk < ord.size(); ++kk) {
                    const int jPrev = ord[kk - 1];
                    const int jCurr = ord[kk];

                    double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
                    if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
                    const double svcPrev = serviceDur(s, jPrev);
                    const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);

                    earliest = std::max(EST[jCurr], tPrevStart + svcPrev + tr);
                    earliest = pushPastPinnedVirtuals(
                        s, jCurr, earliest, tauFeas, ordersVirt[s],
                        virtPinnedStart, svcDur, inst.Rpar);
                    tauFeas[s][jCurr] = earliest;
                }
            }

            // ==================== (B) MR sync ====================
            for (int j = 0; j < n; ++j) {
                if (!isMR[j]) { continue; }
                const auto& P = taskRobots[j];
                if (P.empty()) { continue; }

                // tj = max([EST(j); tau_feas(P,j)], [], 'omitnan')
                double tj = EST[j];
                bool   found = false;
                for (int s : P) {
                    const double v = tauFeas[s][j];
                    if (std::isfinite(v)) { tj = found ? std::max(tj, v) : std::max(EST[j], v); found = true; }
                }
                if (!found) { tj = EST[j]; }

                thetaFeas[j] = tj;
                for (int s : P) { tauFeas[s][j] = tj; }
            }

            // ==================== (B2) re-push + push_past_pinned + MR lock ====================
            for (int s = 0; s < m; ++s) {
                const auto& ord = ordersPhys[s];
                if (ord.empty()) { continue; }

                const double rel = getRobotRelease(s);
                const int    j1 = ord[0];
                const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);

                double earliest = std::max(EST[j1], rel + tr01);
                earliest = pushPastPinnedVirtuals(
                    s, j1, earliest, tauFeas, ordersVirt[s],
                    virtPinnedStart, svcDur, inst.Rpar);
                tauFeas[s][j1] = earliest;
                if (isMR[j1]) {
                    // lock MR to synchronized time, but still respect pinned virtuals
                    tauFeas[s][j1] = std::max(
                        thetaFeas[j1],
                        pushPastPinnedVirtuals(
                            s, j1, thetaFeas[j1], tauFeas, ordersVirt[s],
                            virtPinnedStart, svcDur, inst.Rpar));
                }

                for (std::size_t kk = 1; kk < ord.size(); ++kk) {
                    const int jPrev = ord[kk - 1];
                    const int jCurr = ord[kk];

                    double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
                    if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
                    const double svcPrev = serviceDur(s, jPrev);
                    const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);

                    earliest = std::max(EST[jCurr], tPrevStart + svcPrev + tr);
                    earliest = pushPastPinnedVirtuals(
                        s, jCurr, earliest, tauFeas, ordersVirt[s],
                        virtPinnedStart, svcDur, inst.Rpar);
                    tauFeas[s][jCurr] = earliest;
                    if (isMR[jCurr]) {
                        tauFeas[s][jCurr] = std::max(
                            thetaFeas[jCurr],
                            pushPastPinnedVirtuals(
                                s, jCurr, thetaFeas[jCurr], tauFeas, ordersVirt[s],
                                virtPinnedStart, svcDur, inst.Rpar));
                    }
                }
            }

            // ==================== (C) Virtual lane — fake_Rpar = Rpar ====================
            for (int s = 0; s < m; ++s) {
                const auto& ordV = ordersVirt[s];
                if (ordV.empty()) { continue; }

                const auto& physOrd = ordersPhys[s];
                const int   nPhys = static_cast<int>(physOrd.size());

                VecDouble physStart(nPhys, 0.0);
                VecDouble physEnd(nPhys, 0.0);
                for (int pp = 0; pp < nPhys; ++pp) {
                    const int jp = physOrd[pp];
                    double sp = tauFeas[s][jp];
                    if (!std::isfinite(sp)) { sp = 0.0; }
                    physStart[pp] = sp;
                    physEnd[pp] = sp + serviceDur(s, jp);
                }

                VecInt scheduledV;

                for (int jv : ordV) {
                    // Pinned start or EST
                    double t0 = std::isfinite(virtPinnedStart[jv])
                        ? std::max(EST[jv], virtPinnedStart[jv])
                        : EST[jv];
                    const double dv = serviceDur(s, jv);

                    // 1) virtual-vs-virtual  (fake_Rpar = Rpar)
                    for (int q : scheduledV) {
                        if (inst.Rpar[jv][q] == 1.0) { continue; }
                        const double tq = tauFeas[s][q];
                        if (!std::isfinite(tq)) { continue; }
                        const double dq = serviceDur(s, q);
                        if (!(t0 + dv <= tq || t0 >= tq + dq)) { t0 = tq + dq; }
                    }

                    // 2) virtual-vs-physical  (fake_Rpar = Rpar)
                    bool pushed = true;
                    while (pushed) {
                        pushed = false;
                        for (int pp = 0; pp < nPhys; ++pp) {
                            const int jp = physOrd[pp];
                            if (inst.Rpar[jv][jp] == 1.0) { continue; }
                            const double sp = physStart[pp];
                            const double ep = physEnd[pp];
                            if (!std::isfinite(sp)) { continue; }
                            if (!(t0 + dv <= sp || t0 >= ep)) { t0 = ep; pushed = true; }
                        }
                    }

                    // 3) Commit
                    if (t0 < 0.0) { t0 = 0.0; }
                    tauFeas[s][jv] = t0;
                    scheduledV.push_back(jv);
                }
            }

            // ==================== (D) update global t ====================
            // (D) update global t
            VecDouble tBar = tFeas;
            for (int j = 0; j < n; ++j) {
                const auto& P = taskRobots[j];
                if (P.empty()) { continue; }
                tBar[j] = isMR[j] ? thetaFeas[j] : tauFeas[P.front()][j];
            }
            if (!inst.predPairs.empty()) {
                tFeas = graph.pushForward(tBar, durTask);
            }
            else {
                tFeas = std::move(tBar);
            }

            // Reorder if not frozen — with RANDOMIZE_TIES
            if (!doFreeze) {
                for (int s = 0; s < m; ++s) {
                    // Physical tasks
                    auto& phys = ordersPhys[s];
                    if (!phys.empty()) {
                        const int nP = static_cast<int>(phys.size());
                        VecDouble st(nP);
                        for (int kk = 0; kk < nP; ++kk) {
                            const int j = phys[kk];
                            double v = isMR[j] ? thetaFeas[j] : tauFeas[s][j];
                            if (!std::isfinite(v)) { v = std::numeric_limits<double>::infinity(); }
                            st[kk] = v;
                        }
                        // Sort by st
                        VecInt idx(nP);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(),
                            [&](int a, int b) { return st[a] < st[b]; });
                        VecInt    physSorted(nP);
                        VecDouble stSorted(nP);
                        for (int kk = 0; kk < nP; ++kk) {
                            physSorted[kk] = phys[idx[kk]];
                            stSorted[kk] = st[idx[kk]];
                        }
                        // Randomise ties
                        if (RANDOMIZE_TIES) {
                            int k = 0;
                            while (k < nP) {
                                int r = k;
                                while (r + 1 < nP &&
                                    std::abs(stSorted[r + 1] - stSorted[k]) <= TIE_EPS)
                                {
                                    ++r;
                                }
                                if (r > k) {
                                    std::shuffle(
                                        physSorted.begin() + k,
                                        physSorted.begin() + r + 1,
                                        rng);
                                }
                                k = r + 1;
                            }
                        }
                        ordersPhys[s] = std::move(physSorted);
                    }

                    // Virtual tasks
                    auto& virt = ordersVirt[s];
                    if (!virt.empty()) {
                        const int nV = static_cast<int>(virt.size());
                        VecDouble stv(nV);
                        for (int kk = 0; kk < nV; ++kk) { stv[kk] = tauFeas[s][virt[kk]]; }
                        VecInt idx(nV);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(),
                            [&](int a, int b) { return stv[a] < stv[b]; });
                        VecInt    virtSorted(nV);
                        VecDouble stvSorted(nV);
                        for (int kk = 0; kk < nV; ++kk) {
                            virtSorted[kk] = virt[idx[kk]];
                            stvSorted[kk] = stv[idx[kk]];
                        }
                        if (RANDOMIZE_TIES) {
                            int k = 0;
                            while (k < nV) {
                                int r = k;
                                while (r + 1 < nV &&
                                    std::abs(stvSorted[r + 1] - stvSorted[k]) <= TIE_EPS)
                                {
                                    ++r;
                                }
                                if (r > k) {
                                    std::shuffle(
                                        virtSorted.begin() + k,
                                        virtSorted.begin() + r + 1,
                                        rng);
                                }
                                k = r + 1;
                            }
                        }
                        ordersVirt[s] = std::move(virtSorted);
                    }
                }
            }

            // ==================== Convergence ====================
            //double tauDiff = 0.0;
            //bool   infMismatch = false;
            //for (int s = 0; s < m; ++s) {
            //    for (int j = 0; j < n; ++j) {
            //        const bool cf = std::isfinite(tauFeas[s][j]);
            //        const bool pf = std::isfinite(tauPrev[s][j]);
            //        if (cf && pf) { tauDiff = std::max(tauDiff, std::abs(tauFeas[s][j] - tauPrev[s][j])); }
            //        else if (cf != pf) { infMismatch = true; }
            //    }
            //}
            //if (infMismatch) { tauDiff = std::numeric_limits<double>::infinity(); }

            //double thetaDiff = 0.0;
            //for (int j = 0; j < n; ++j) {
            //    thetaDiff = std::max(thetaDiff, std::abs(thetaFeas[j] - thetaPrev[j]));
            //}
            //if (!std::isfinite(thetaDiff)) { thetaDiff = std::numeric_limits<double>::infinity(); }

            //bool sameOrders = true;
            //for (int s = 0; s < m && sameOrders; ++s) {
            //    if (ordersPhys[s] != ordPhysPrev[s] || ordersVirt[s] != ordVirtPrev[s]) {
            //        sameOrders = false;
            //    }
            //}

            //if (it + 1 >= 2 &&
            //    tauDiff <= 1e-6 &&
            //    thetaDiff <= 1e-6 &&
            //    sameOrders) {
            //    break;
            //}
            // 
            // ==================== Convergence ====================
            // Fast scalar check: if makespan hasn't changed, schedule is converged.
            // Compute current makespan from tauFeas and thetaFeas.
            double mkspCurr = 0.0;
            for (int j = 0; j < n; ++j) {
                const auto& P = taskRobots[j];
                if (P.empty()) continue;
                double tjEnd;
                if (isMR[j]) {
                    tjEnd = thetaFeas[j] + durTask[j];
                }
                else {
                    const int s = P.front();
                    tjEnd = std::isfinite(tauFeas[s][j])
                        ? tauFeas[s][j] + svcDur[s][j]
                        : 0.0;
                }
                mkspCurr = std::max(mkspCurr, tjEnd);
            }

            if (it + 1 >= REPAIR_MIN_ITERS &&
                std::abs(mkspCurr - mkspPrev_) <= 1e-6) {
                break;
            }
            mkspPrev_ = mkspCurr;
        }

        VecInt lastPhysNode(m, 0);
        for (int s = 0; s < m; ++s) {
            lastPhysNode[s] = ordersPhys[s].empty()
                ? s
                : taskNode[ordersPhys[s].back()];
        }
        return RepairResult{ tauFeas, thetaFeas, tFeas, ordersPhys, ordersVirt, lastPhysNode };
    }
    // ============================================================================
    // v2 — exact translation of repair_schedule_mt_pushforward2.m
    //
    // Key differences from v1:
    //   nIters honoured (not overridden)
    //   fake_Rpar = Rpar  (real Rpar used in Step C)
    //   push_past_pinned_virtuals in Steps A and B2
    //   pinned virtual times pre-set before Step A
    //   pinned tStart = max(EST, virtPinnedStart) in Step C
    //   RANDOMIZE_TIES = true in reorder step
    // ============================================================================

    //RepairResult ScheduleRepairer::repairPushforward2(
    //    const MatrixDouble& zHard,
    //    const InstanceData& inst,
    //    const VecDouble& thetaInit,
    //    const VecDouble& tInit,
    //    const RepairOptions& options) const
    //{
    //    const int m = inst.m;
    //    const int n = inst.n;

    //    const int    nIters = std::max(1, options.nIters);
    //    const bool   RANDOMIZE_TIES = true;
    //    const double TIE_EPS = 1e-6;

    //    const BoolVec& iv = inst.isVirtual;
    //    const BoolVec& isMR = inst.isMR;

    //    VecDouble virtPinnedStart(n, std::numeric_limits<double>::quiet_NaN());
    //    if (options.virtPinnedStart.has_value()) {
    //        if (static_cast<int>(options.virtPinnedStart->size()) != n) {
    //            throw std::invalid_argument(
    //                "repairPushforward2: virtPinnedStart must have length n.");
    //        }
    //        virtPinnedStart = *options.virtPinnedStart;
    //    }

    //    MatrixDouble tauFeas(m, VecDouble(n, std::numeric_limits<double>::quiet_NaN()));
    //    VecDouble    thetaFeas = thetaInit;
    //    VecDouble    tFeas = tInit;

    //    // ------------------------------------------------------------------
    //    // startNode, durTask, svcDur  (same structure as v1)
    //    // ------------------------------------------------------------------
    //    VecInt taskNode(n);
    //    for (int j = 0; j < n; ++j) { taskNode[j] = m + j; }

    //    VecInt startNode(m);
    //    for (int s = 0; s < m; ++s) { startNode[s] = s; }

    //    VecDouble durTask(n, 0.0);
    //    for (int j = 0; j < n; ++j) {
    //        durTask[j] = iv[j] ? maxFiniteInColumn(inst.svcVirtSJ, j)
    //            : maxFiniteInColumn(inst.svcPhysSJ, j);
    //    }

    //    // service_dur(s,j)  — v2 uses a nested function; replicate inline
    //    // Same logic as v1 except accessed via lambda to mirror MATLAB's nested fn
    //    auto serviceDur = [&](int s, int j) -> double {
    //        double d;
    //        if (isMR[j]) {
    //            d = durTask[j];
    //        }
    //        else {
    //            d = iv[j] ? inst.svcVirtSJ[s][j] : inst.svcPhysSJ[s][j];
    //            if (!std::isfinite(d) || d < 0.0) { d = durTask[j]; }
    //        }
    //        if (!std::isfinite(d) || d < 0.0) { d = 0.0; }
    //        return d;
    //        };

    //    // Build svcDur matrix using serviceDur (for pushPastPinnedVirtuals)
    //    MatrixDouble svcDur(m, VecDouble(n, 0.0));
    //    for (int s = 0; s < m; ++s) {
    //        for (int j = 0; j < n; ++j) { svcDur[s][j] = serviceDur(s, j); }
    //    }

    //    // getRobotRelease(s) — v2 uses same logic as v1
    //    auto getRobotRelease = [&](int s) -> double {
    //        if (inst.Tstart.empty()) { return 0.0; }
    //        if (inst.Tstart.size() == 1 && inst.Tstart[0].size() == 1) {
    //            const double v = inst.Tstart[0][0];
    //            return std::isfinite(v) ? v : 0.0;
    //        }
    //        double best = std::numeric_limits<double>::infinity();
    //        for (const double v : inst.Tstart[s]) {
    //            if (std::isfinite(v)) { best = std::min(best, v); }
    //        }
    //        return std::isfinite(best) ? best : 0.0;
    //        };

    //    // taskRobots
    //    std::vector<VecInt> taskRobots(n);
    //    for (int j = 0; j < n; ++j) {
    //        for (int s = 0; s < m; ++s) {
    //            if (zHard[s][j] > 0.5) { taskRobots[j].push_back(s); }
    //        }
    //    }

    //    // ------------------------------------------------------------------
    //    // Initial orders
    //    // ------------------------------------------------------------------
    //    std::vector<VecInt> ordersPhys(m);
    //    std::vector<VecInt> ordersVirt(m);

    //    const bool doFreeze =
    //        options.freezeOrders &&
    //        options.ordersPhys0.has_value() &&
    //        options.ordersVirt0.has_value();

    //    if (doFreeze) {
    //        ordersPhys = normalizeOrders(options.ordersPhys0, m);
    //        ordersVirt = normalizeOrders(options.ordersVirt0, m);
    //    }
    //    else {
    //        for (int s = 0; s < m; ++s) {
    //            VecInt phys, virt;
    //            for (int j = 0; j < n; ++j) {
    //                if (zHard[s][j] > 0.5) {
    //                    if (iv[j]) { virt.push_back(j); }
    //                    else { phys.push_back(j); }
    //                    tauFeas[s][j] = std::max(0.0, tFeas[j]);
    //                }
    //            }
    //            std::sort(phys.begin(), phys.end(),
    //                [&](int a, int b) { return tFeas[a] < tFeas[b]; });
    //            std::sort(virt.begin(), virt.end(),
    //                [&](int a, int b) { return tFeas[a] < tFeas[b]; });
    //            ordersPhys[s] = std::move(phys);
    //            ordersVirt[s] = std::move(virt);
    //        }
    //    }

    //    PrecedenceGraph graph(n, inst.predPairs);

    //    // Random engine for tie-breaking (RANDOMIZE_TIES = true)
    //    std::mt19937 rng(std::random_device{}());

    //    // ------------------------------------------------------------------
    //    // Main iteration loop
    //    // ------------------------------------------------------------------
    //    for (int it = 0; it < nIters; ++it) {

    //        const MatrixDouble tauPrev = tauFeas;
    //        const VecDouble    thetaPrev = thetaFeas;
    //        const auto         ordPhysPrev = ordersPhys;
    //        const auto         ordVirtPrev = ordersVirt;

    //        // ==================== C(j) ====================
    //        VecDouble C(n, 0.0);
    //        for (int j = 0; j < n; ++j) {
    //            const auto& P = taskRobots[j];
    //            if (P.empty()) { continue; }
    //            if (isMR[j]) {
    //                double tj = 0.0; bool found = false;
    //                for (int s : P) {
    //                    const double v = tauFeas[s][j];
    //                    if (std::isfinite(v)) { tj = found ? std::max(tj, v) : v; found = true; }
    //                }
    //                if (!found) { tj = 0.0; }
    //                C[j] = tj + durTask[j];
    //            }
    //            else {
    //                const int s = P.front();
    //                if (std::isfinite(tauFeas[s][j])) {
    //                    C[j] = tauFeas[s][j] + serviceDur(s, j);
    //                }
    //            }
    //        }

    //        // ==================== EST ====================
    //        VecDouble EST(n, 0.0);
    //        for (const auto& e : inst.predPairs) {
    //            EST[e.to] = std::max(EST[e.to], C[e.from]);
    //        }

    //        // ==================== Pre-set pinned virtual times (v2 only) ====================
    //        // MATLAB: tau_feas(s,jv) = max(EST(jv), virtPinnedStart(jv))
    //        for (int s = 0; s < m; ++s) {
    //            for (int jv : ordersVirt[s]) {
    //                if (std::isfinite(virtPinnedStart[jv])) {
    //                    tauFeas[s][jv] = std::max(EST[jv], virtPinnedStart[jv]);
    //                }
    //            }
    //        }

    //        // ==================== (A) physical lane ASAP + push_past_pinned ====================
    //        for (int s = 0; s < m; ++s) {
    //            const auto& ord = ordersPhys[s];
    //            if (ord.empty()) { continue; }

    //            const double rel = getRobotRelease(s);
    //            const int    j1 = ord[0];
    //            const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);

    //            double earliest = std::max(EST[j1], rel + tr01);
    //            earliest = pushPastPinnedVirtuals(
    //                s, j1, earliest, tauFeas, ordersVirt[s],
    //                virtPinnedStart, svcDur, inst.Rpar);
    //            tauFeas[s][j1] = earliest;

    //            for (std::size_t kk = 1; kk < ord.size(); ++kk) {
    //                const int jPrev = ord[kk - 1];
    //                const int jCurr = ord[kk];

    //                double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
    //                if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
    //                const double svcPrev = serviceDur(s, jPrev);
    //                const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);

    //                earliest = std::max(EST[jCurr], tPrevStart + svcPrev + tr);
    //                earliest = pushPastPinnedVirtuals(
    //                    s, jCurr, earliest, tauFeas, ordersVirt[s],
    //                    virtPinnedStart, svcDur, inst.Rpar);
    //                tauFeas[s][jCurr] = earliest;
    //            }
    //        }

    //        // ==================== (B) MR sync ====================
    //        for (int j = 0; j < n; ++j) {
    //            if (!isMR[j]) { continue; }
    //            const auto& P = taskRobots[j];
    //            if (P.empty()) { continue; }

    //            // tj = max([EST(j); tau_feas(P,j)], [], 'omitnan')
    //            double tj = EST[j];
    //            bool   found = false;
    //            for (int s : P) {
    //                const double v = tauFeas[s][j];
    //                if (std::isfinite(v)) { tj = found ? std::max(tj, v) : std::max(EST[j], v); found = true; }
    //            }
    //            if (!found) { tj = EST[j]; }

    //            thetaFeas[j] = tj;
    //            for (int s : P) { tauFeas[s][j] = tj; }
    //        }

    //        // ==================== (B2) re-push + push_past_pinned + MR lock ====================
    //        for (int s = 0; s < m; ++s) {
    //            const auto& ord = ordersPhys[s];
    //            if (ord.empty()) { continue; }

    //            const double rel = getRobotRelease(s);
    //            const int    j1 = ord[0];
    //            const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);

    //            double earliest = std::max(EST[j1], rel + tr01);
    //            earliest = pushPastPinnedVirtuals(
    //                s, j1, earliest, tauFeas, ordersVirt[s],
    //                virtPinnedStart, svcDur, inst.Rpar);
    //            tauFeas[s][j1] = earliest;
    //            if (isMR[j1]) {
    //                // lock MR to synchronized time, but still respect pinned virtuals
    //                tauFeas[s][j1] = std::max(
    //                    thetaFeas[j1],
    //                    pushPastPinnedVirtuals(
    //                        s, j1, thetaFeas[j1], tauFeas, ordersVirt[s],
    //                        virtPinnedStart, svcDur, inst.Rpar));
    //            }

    //            for (std::size_t kk = 1; kk < ord.size(); ++kk) {
    //                const int jPrev = ord[kk - 1];
    //                const int jCurr = ord[kk];

    //                double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
    //                if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
    //                const double svcPrev = serviceDur(s, jPrev);
    //                const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);

    //                earliest = std::max(EST[jCurr], tPrevStart + svcPrev + tr);
    //                earliest = pushPastPinnedVirtuals(
    //                    s, jCurr, earliest, tauFeas, ordersVirt[s],
    //                    virtPinnedStart, svcDur, inst.Rpar);
    //                tauFeas[s][jCurr] = earliest;
    //                if (isMR[jCurr]) {
    //                    tauFeas[s][jCurr] = std::max(
    //                        thetaFeas[jCurr],
    //                        pushPastPinnedVirtuals(
    //                            s, jCurr, thetaFeas[jCurr], tauFeas, ordersVirt[s],
    //                            virtPinnedStart, svcDur, inst.Rpar));
    //                }
    //            }
    //        }

    //        // ==================== (C) Virtual lane — fake_Rpar = Rpar ====================
    //        for (int s = 0; s < m; ++s) {
    //            const auto& ordV = ordersVirt[s];
    //            if (ordV.empty()) { continue; }

    //            const auto& physOrd = ordersPhys[s];
    //            const int   nPhys = static_cast<int>(physOrd.size());

    //            VecDouble physStart(nPhys, 0.0);
    //            VecDouble physEnd(nPhys, 0.0);
    //            for (int pp = 0; pp < nPhys; ++pp) {
    //                const int jp = physOrd[pp];
    //                double sp = tauFeas[s][jp];
    //                if (!std::isfinite(sp)) { sp = 0.0; }
    //                physStart[pp] = sp;
    //                physEnd[pp] = sp + serviceDur(s, jp);
    //            }

    //            VecInt scheduledV;

    //            for (int jv : ordV) {
    //                // Pinned start or EST
    //                double t0 = std::isfinite(virtPinnedStart[jv])
    //                    ? std::max(EST[jv], virtPinnedStart[jv])
    //                    : EST[jv];
    //                const double dv = serviceDur(s, jv);

    //                // 1) virtual-vs-virtual  (fake_Rpar = Rpar)
    //                for (int q : scheduledV) {
    //                    if (inst.Rpar[jv][q] == 1.0) { continue; }
    //                    const double tq = tauFeas[s][q];
    //                    if (!std::isfinite(tq)) { continue; }
    //                    const double dq = serviceDur(s, q);
    //                    if (!(t0 + dv <= tq || t0 >= tq + dq)) { t0 = tq + dq; }
    //                }

    //                // 2) virtual-vs-physical  (fake_Rpar = Rpar)
    //                bool pushed = true;
    //                while (pushed) {
    //                    pushed = false;
    //                    for (int pp = 0; pp < nPhys; ++pp) {
    //                        const int jp = physOrd[pp];
    //                        if (inst.Rpar[jv][jp] == 1.0) { continue; }
    //                        const double sp = physStart[pp];
    //                        const double ep = physEnd[pp];
    //                        if (!std::isfinite(sp)) { continue; }
    //                        if (!(t0 + dv <= sp || t0 >= ep)) { t0 = ep; pushed = true; }
    //                    }
    //                }

    //                // 3) Commit
    //                if (t0 < 0.0) { t0 = 0.0; }
    //                tauFeas[s][jv] = t0;
    //                scheduledV.push_back(jv);
    //            }
    //        }

    //        // ==================== (D) update global t ====================
    //        VecDouble tBar = tFeas;
    //        for (int j = 0; j < n; ++j) {
    //            const auto& P = taskRobots[j];
    //            if (P.empty()) { continue; }
    //            tBar[j] = isMR[j] ? thetaFeas[j] : tauFeas[P.front()][j];
    //        }
    //        tFeas = graph.pushForward(tBar, durTask);

    //        // Reorder if not frozen — with RANDOMIZE_TIES
    //        if (!doFreeze) {
    //            for (int s = 0; s < m; ++s) {
    //                // Physical tasks
    //                auto& phys = ordersPhys[s];
    //                if (!phys.empty()) {
    //                    const int nP = static_cast<int>(phys.size());
    //                    VecDouble st(nP);
    //                    for (int kk = 0; kk < nP; ++kk) {
    //                        const int j = phys[kk];
    //                        double v = isMR[j] ? thetaFeas[j] : tauFeas[s][j];
    //                        if (!std::isfinite(v)) { v = std::numeric_limits<double>::infinity(); }
    //                        st[kk] = v;
    //                    }
    //                    // Sort by st
    //                    VecInt idx(nP);
    //                    std::iota(idx.begin(), idx.end(), 0);
    //                    std::sort(idx.begin(), idx.end(),
    //                        [&](int a, int b) { return st[a] < st[b]; });
    //                    VecInt    physSorted(nP);
    //                    VecDouble stSorted(nP);
    //                    for (int kk = 0; kk < nP; ++kk) {
    //                        physSorted[kk] = phys[idx[kk]];
    //                        stSorted[kk] = st[idx[kk]];
    //                    }
    //                    // Randomise ties
    //                    if (RANDOMIZE_TIES) {
    //                        int k = 0;
    //                        while (k < nP) {
    //                            int r = k;
    //                            while (r + 1 < nP &&
    //                                std::abs(stSorted[r + 1] - stSorted[k]) <= TIE_EPS)
    //                            {
    //                                ++r;
    //                            }
    //                            if (r > k) {
    //                                std::shuffle(
    //                                    physSorted.begin() + k,
    //                                    physSorted.begin() + r + 1,
    //                                    rng);
    //                            }
    //                            k = r + 1;
    //                        }
    //                    }
    //                    ordersPhys[s] = std::move(physSorted);
    //                }

    //                // Virtual tasks
    //                auto& virt = ordersVirt[s];
    //                if (!virt.empty()) {
    //                    const int nV = static_cast<int>(virt.size());
    //                    VecDouble stv(nV);
    //                    for (int kk = 0; kk < nV; ++kk) { stv[kk] = tauFeas[s][virt[kk]]; }
    //                    VecInt idx(nV);
    //                    std::iota(idx.begin(), idx.end(), 0);
    //                    std::sort(idx.begin(), idx.end(),
    //                        [&](int a, int b) { return stv[a] < stv[b]; });
    //                    VecInt    virtSorted(nV);
    //                    VecDouble stvSorted(nV);
    //                    for (int kk = 0; kk < nV; ++kk) {
    //                        virtSorted[kk] = virt[idx[kk]];
    //                        stvSorted[kk] = stv[idx[kk]];
    //                    }
    //                    if (RANDOMIZE_TIES) {
    //                        int k = 0;
    //                        while (k < nV) {
    //                            int r = k;
    //                            while (r + 1 < nV &&
    //                                std::abs(stvSorted[r + 1] - stvSorted[k]) <= TIE_EPS)
    //                            {
    //                                ++r;
    //                            }
    //                            if (r > k) {
    //                                std::shuffle(
    //                                    virtSorted.begin() + k,
    //                                    virtSorted.begin() + r + 1,
    //                                    rng);
    //                            }
    //                            k = r + 1;
    //                        }
    //                    }
    //                    ordersVirt[s] = std::move(virtSorted);
    //                }
    //            }
    //        }

    //        // ==================== Convergence ====================
    //        double tauDiff = 0.0;
    //        bool   infMismatch = false;
    //        for (int s = 0; s < m; ++s) {
    //            for (int j = 0; j < n; ++j) {
    //                const bool cf = std::isfinite(tauFeas[s][j]);
    //                const bool pf = std::isfinite(tauPrev[s][j]);
    //                if (cf && pf) { tauDiff = std::max(tauDiff, std::abs(tauFeas[s][j] - tauPrev[s][j])); }
    //                else if (cf != pf) { infMismatch = true; }
    //            }
    //        }
    //        if (infMismatch) { tauDiff = std::numeric_limits<double>::infinity(); }

    //        double thetaDiff = 0.0;
    //        for (int j = 0; j < n; ++j) {
    //            thetaDiff = std::max(thetaDiff, std::abs(thetaFeas[j] - thetaPrev[j]));
    //        }
    //        if (!std::isfinite(thetaDiff)) { thetaDiff = std::numeric_limits<double>::infinity(); }

    //        bool sameOrders = true;
    //        for (int s = 0; s < m && sameOrders; ++s) {
    //            if (ordersPhys[s] != ordPhysPrev[s] || ordersVirt[s] != ordVirtPrev[s]) {
    //                sameOrders = false;
    //            }
    //        }

    //        if (it + 1 >= 2 &&
    //            tauDiff <= 1e-6 &&
    //            thetaDiff <= 1e-6 &&
    //            sameOrders) {
    //            break;
    //        }
    //    }

    //    VecInt lastPhysNode(m, 0);
    //    for (int s = 0; s < m; ++s) {
    //        lastPhysNode[s] = ordersPhys[s].empty()
    //            ? s
    //            : taskNode[ordersPhys[s].back()];
    //    }
    //    return RepairResult{ tauFeas, thetaFeas, tFeas, ordersPhys, ordersVirt, lastPhysNode };
    //}


    // ============================================================================
 // v1 — exact translation of repair_schedule_mt_pushforward.m
 //
 // Key MATLAB behaviours preserved:
 //   nIters = 10;          — hard override at top of function
 //   fake_Rpar = ones(...) — virtual overlap always skipped (continue)
 //   No push_past_pinned_virtuals
 //   No RANDOMIZE_TIES
 // ============================================================================

 //RepairResult ScheduleRepairer::repairPushforward(
 //    const MatrixDouble& zHard,
 //    const InstanceData& inst,
 //    const VecDouble& thetaInit,
 //    const VecDouble& tInit,
 //    const RepairOptions& options) const
 //{
 //    const int m = inst.m;
 //    const int n = inst.n;

 //    // MATLAB line 1: nIters = 10;  — always overrides whatever was passed
 //    const int    nIters = 10;
 //    const double REPAIR_TOL_TIME = 1e-6;
 //    const int    REPAIR_MIN_ITERS = 2;

 //    const BoolVec& iv = inst.isVirtual;
 //    const BoolVec& isMR = inst.isMR;

 //    // virtPinnedStart (accepted but not used in v1 logic, same as MATLAB)
 //    VecDouble virtPinnedStart(n, std::numeric_limits<double>::quiet_NaN());
 //    if (options.virtPinnedStart.has_value()) {
 //        virtPinnedStart = *options.virtPinnedStart;
 //    }

 //    MatrixDouble tauFeas(m, VecDouble(n, std::numeric_limits<double>::quiet_NaN()));
 //    VecDouble    thetaFeas = thetaInit;
 //    VecDouble    tFeas = tInit;

 //    // ------------------------------------------------------------------
 //    // PRECOMPUTE  (mirrors the "PRECOMPUTE" block in the MATLAB optimised v1)
 //    // ------------------------------------------------------------------

 //    // taskNode(j) = m + j
 //    VecInt taskNode(n);
 //    for (int j = 0; j < n; ++j) { taskNode[j] = m + j; }

 //    // startNode(s) = s  (0-based)
 //    VecInt startNode(m);
 //    for (int s = 0; s < m; ++s) { startNode[s] = s; }

 //    // durTask(j) = max finite value in column j of svcPhys/VirtSJ
 //    VecDouble durTask(n, 0.0);
 //    for (int j = 0; j < n; ++j) {
 //        durTask[j] = iv[j] ? maxFiniteInColumn(inst.svcVirtSJ, j)
 //            : maxFiniteInColumn(inst.svcPhysSJ, j);
 //    }

 //    // svcDur(s,j)
 //    MatrixDouble svcDur(m, VecDouble(n, 0.0));
 //    for (int j = 0; j < n; ++j) {
 //        if (isMR[j]) {
 //            for (int s = 0; s < m; ++s) { svcDur[s][j] = durTask[j]; }
 //        }
 //        else {
 //            for (int s = 0; s < m; ++s) {
 //                double d = iv[j] ? inst.svcVirtSJ[s][j] : inst.svcPhysSJ[s][j];
 //                if (!std::isfinite(d) || d < 0.0) { d = durTask[j]; }
 //                if (!std::isfinite(d) || d < 0.0) { d = 0.0; }
 //                svcDur[s][j] = d;
 //            }
 //        }
 //    }

 //    // robotRelease(s)
 //    VecDouble robotRelease(m, 0.0);
 //    if (!inst.Tstart.empty()) {
 //        if (inst.Tstart.size() == 1 && inst.Tstart[0].size() == 1) {
 //            // scalar
 //            const double v = inst.Tstart[0][0];
 //            for (int s = 0; s < m; ++s) { robotRelease[s] = std::isfinite(v) ? v : 0.0; }
 //        }
 //        else {
 //            for (int s = 0; s < m; ++s) {
 //                double best = std::numeric_limits<double>::infinity();
 //                for (const double v : inst.Tstart[s]) {
 //                    if (std::isfinite(v)) { best = std::min(best, v); }
 //                }
 //                robotRelease[s] = std::isfinite(best) ? best : 0.0;
 //            }
 //        }
 //    }

 //    // taskRobots{j} = robots assigned to task j
 //    std::vector<VecInt> taskRobots(n);
 //    for (int j = 0; j < n; ++j) {
 //        for (int s = 0; s < m; ++s) {
 //            if (zHard[s][j] > 0.5) { taskRobots[j].push_back(s); }
 //        }
 //    }

 //    // ------------------------------------------------------------------
 //    // Initial orders
 //    // ------------------------------------------------------------------
 //    std::vector<VecInt> ordersPhys(m);
 //    std::vector<VecInt> ordersVirt(m);

 //    const bool doFreeze =
 //        options.freezeOrders &&
 //        options.ordersPhys0.has_value() &&
 //        options.ordersVirt0.has_value();

 //    if (doFreeze) {
 //        ordersPhys = normalizeOrders(options.ordersPhys0, m);
 //        ordersVirt = normalizeOrders(options.ordersVirt0, m);
 //    }
 //    else {
 //        for (int s = 0; s < m; ++s) {
 //            VecInt phys, virt;
 //            for (int j = 0; j < n; ++j) {
 //                if (zHard[s][j] > 0.5) {
 //                    if (iv[j]) { virt.push_back(j); }
 //                    else { phys.push_back(j); }
 //                    tauFeas[s][j] = std::max(0.0, tFeas[j]);
 //                }
 //            }
 //            std::sort(phys.begin(), phys.end(),
 //                [&](int a, int b) { return tFeas[a] < tFeas[b]; });
 //            std::sort(virt.begin(), virt.end(),
 //                [&](int a, int b) { return tFeas[a] < tFeas[b]; });
 //            ordersPhys[s] = std::move(phys);
 //            ordersVirt[s] = std::move(virt);
 //        }
 //    }

 //    PrecedenceGraph graph(n, inst.predPairs);

 //    // ------------------------------------------------------------------
 //    // Main iteration loop
 //    // ------------------------------------------------------------------
 //    for (int it = 0; it < nIters; ++it) {

 //        const MatrixDouble tauPrev = tauFeas;
 //        const VecDouble    thetaPrev = thetaFeas;
 //        const auto         ordPhysPrev = ordersPhys;
 //        const auto         ordVirtPrev = ordersVirt;

 //        // ==================== C(j) ====================
 //        VecDouble C(n, 0.0);
 //        for (int j = 0; j < n; ++j) {
 //            const auto& P = taskRobots[j];
 //            if (P.empty()) { continue; }
 //            if (isMR[j]) {
 //                double tj = 0.0; bool found = false;
 //                for (int s : P) {
 //                    const double v = tauFeas[s][j];
 //                    if (std::isfinite(v)) { tj = found ? std::max(tj, v) : v; found = true; }
 //                }
 //                C[j] = tj + durTask[j];
 //            }
 //            else {
 //                const int s = P.front();
 //                if (std::isfinite(tauFeas[s][j])) {
 //                    C[j] = tauFeas[s][j] + svcDur[s][j];
 //                }
 //            }
 //        }

 //        // ==================== EST ====================
 //        VecDouble EST(n, 0.0);
 //        for (const auto& e : inst.predPairs) {
 //            EST[e.to] = std::max(EST[e.to], C[e.from]);
 //        }

 //        // ==================== (A) physical lane ASAP ====================
 //        for (int s = 0; s < m; ++s) {
 //            const auto& ord = ordersPhys[s];
 //            if (ord.empty()) { continue; }

 //            const double rel = robotRelease[s];
 //            const int    j1 = ord[0];
 //            const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);

 //            const double arrJ1 = rel + tr01;
 //            tauFeas[s][j1] = (EST[j1] > arrJ1) ? EST[j1] : arrJ1;

 //            for (std::size_t kk = 1; kk < ord.size(); ++kk) {
 //                const int jPrev = ord[kk - 1];
 //                const int jCurr = ord[kk];

 //                double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
 //                if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
 //                const double svcPrev = isMR[jPrev] ? durTask[jPrev] : svcDur[s][jPrev];
 //                const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);
 //                const double earliest = tPrevStart + svcPrev + tr;

 //                tauFeas[s][jCurr] = (EST[jCurr] > earliest) ? EST[jCurr] : earliest;
 //            }
 //        }

 //        // ==================== (B) MR sync ====================
 //        for (int j = 0; j < n; ++j) {
 //            if (!isMR[j]) { continue; }
 //            const auto& P = taskRobots[j];
 //            if (P.empty()) { continue; }
 //            double tj = EST[j];
 //            for (int s : P) {
 //                const double v = tauFeas[s][j];
 //                if (std::isfinite(v) && v > tj) { tj = v; }
 //            }
 //            thetaFeas[j] = tj;
 //            for (int s : P) { tauFeas[s][j] = tj; }
 //        }

 //        // ==================== (B2) re-push after MR sync ====================
 //        for (int s = 0; s < m; ++s) {
 //            const auto& ord = ordersPhys[s];
 //            if (ord.empty()) { continue; }

 //            const double rel = robotRelease[s];
 //            const int    j1 = ord[0];
 //            const double tr01 = safeTravel(inst.W, startNode[s], taskNode[j1]);
 //            const double arrJ1 = rel + tr01;

 //            tauFeas[s][j1] = (EST[j1] > arrJ1) ? EST[j1] : arrJ1;
 //            if (isMR[j1]) { tauFeas[s][j1] = thetaFeas[j1]; }

 //            for (std::size_t kk = 1; kk < ord.size(); ++kk) {
 //                const int jPrev = ord[kk - 1];
 //                const int jCurr = ord[kk];

 //                double tPrevStart = isMR[jPrev] ? thetaFeas[jPrev] : tauFeas[s][jPrev];
 //                if (!std::isfinite(tPrevStart)) { tPrevStart = 0.0; }
 //                const double svcPrev = isMR[jPrev] ? durTask[jPrev] : svcDur[s][jPrev];
 //                const double tr = safeTravel(inst.W, taskNode[jPrev], taskNode[jCurr]);
 //                const double earliest = tPrevStart + svcPrev + tr;

 //                tauFeas[s][jCurr] = (EST[jCurr] > earliest) ? EST[jCurr] : earliest;
 //                if (isMR[jCurr]) { tauFeas[s][jCurr] = thetaFeas[jCurr]; }
 //            }
 //        }

 //        // ==================== (C) Virtual lane — fake_Rpar = ones ====================
 //        // fake_Rpar = ones(...)  means every  if fake_Rpar(jv,q)==1  fires → continue
 //        // So the overlap body NEVER executes. Virtual tasks sit at EST[jv].
 //        for (int s = 0; s < m; ++s) {
 //            const auto& ordV = ordersVirt[s];
 //            if (ordV.empty()) { continue; }

 //            const auto& physOrd = ordersPhys[s];
 //            const int   nPhys = static_cast<int>(physOrd.size());

 //            // Precompute physical intervals (kept for structural parity;
 //            // never actually used because fake_Rpar always triggers continue)
 //            VecDouble physStart(nPhys, 0.0);
 //            VecDouble physEnd(nPhys, 0.0);
 //            for (int pp = 0; pp < nPhys; ++pp) {
 //                const int jp = physOrd[pp];
 //                double sp = tauFeas[s][jp];
 //                if (!std::isfinite(sp)) { sp = 0.0; }
 //                physStart[pp] = sp;
 //                physEnd[pp] = sp + svcDur[s][jp];
 //            }

 //            VecInt    scheduledV;
 //            VecDouble scheduledVStart;
 //            VecDouble scheduledVEnd;

 //            for (int jv : ordV) {
 //                double       t0 = EST[jv];   // MATLAB: t0 = EST(jv)
 //                const double dv = svcDur[s][jv];

 //                // 1) virtual-vs-virtual  — fake_Rpar(jv,q)==1 always → continue
 //                for (std::size_t qq = 0; qq < scheduledV.size(); ++qq) {
 //                    // if fake_Rpar(jv, q) == 1, continue;  ← always true
 //                    continue;
 //                    // overlap body intentionally unreachable (dead code matching MATLAB)
 //                    const double tq = scheduledVStart[qq];
 //                    const double eq = scheduledVEnd[qq];
 //                    if (!(t0 + dv <= tq || t0 >= eq)) { t0 = eq; }
 //                }

 //                // 2) virtual-vs-physical — fake_Rpar(jv,jp)==1 always → continue
 //                bool pushed = true;
 //                while (pushed) {
 //                    pushed = false;
 //                    for (int pp = 0; pp < nPhys; ++pp) {
 //                        // if fake_Rpar(jv, jp) == 1, continue;  ← always true
 //                        continue;
 //                        // overlap body intentionally unreachable (dead code matching MATLAB)
 //                        const double sp = physStart[pp];
 //                        const double ep = physEnd[pp];
 //                        if (!(t0 + dv <= sp || t0 >= ep)) { t0 = ep; pushed = true; }
 //                    }
 //                }

 //                // 3) Commit
 //                if (t0 < 0.0) { t0 = 0.0; }
 //                tauFeas[s][jv] = t0;
 //                scheduledV.push_back(jv);
 //                scheduledVStart.push_back(t0);
 //                scheduledVEnd.push_back(t0 + dv);
 //            }
 //        }

 //        // ==================== (D) update global t ====================
 //        VecDouble tBar = tFeas;
 //        for (int j = 0; j < n; ++j) {
 //            const auto& P = taskRobots[j];
 //            if (P.empty()) { continue; }
 //            tBar[j] = isMR[j] ? thetaFeas[j] : tauFeas[P.front()][j];
 //        }
 //        tFeas = graph.pushForward(tBar, durTask);

 //        // Reorder if not frozen (deterministic sort, no tie randomisation)
 //        if (!doFreeze) {
 //            for (int s = 0; s < m; ++s) {
 //                auto& phys = ordersPhys[s];
 //                std::sort(phys.begin(), phys.end(), [&](int a, int b) {
 //                    double sa = isMR[a] ? thetaFeas[a] : tauFeas[s][a];
 //                    double sb = isMR[b] ? thetaFeas[b] : tauFeas[s][b];
 //                    if (!std::isfinite(sa)) { sa = std::numeric_limits<double>::infinity(); }
 //                    if (!std::isfinite(sb)) { sb = std::numeric_limits<double>::infinity(); }
 //                    return sa < sb;
 //                    });
 //                auto& virt = ordersVirt[s];
 //                std::sort(virt.begin(), virt.end(), [&](int a, int b) {
 //                    return tauFeas[s][a] < tauFeas[s][b];
 //                    });
 //            }
 //        }

 //        // ==================== Convergence ====================
 //        double tauDiff = 0.0;
 //        bool   infMismatch = false;
 //        for (int s = 0; s < m; ++s) {
 //            for (int j = 0; j < n; ++j) {
 //                const bool cf = std::isfinite(tauFeas[s][j]);
 //                const bool pf = std::isfinite(tauPrev[s][j]);
 //                if (cf && pf) { tauDiff = std::max(tauDiff, std::abs(tauFeas[s][j] - tauPrev[s][j])); }
 //                else if (cf != pf) { infMismatch = true; }
 //            }
 //        }
 //        if (infMismatch) { tauDiff = std::numeric_limits<double>::infinity(); }

 //        double thetaDiff = 0.0;
 //        for (int j = 0; j < n; ++j) {
 //            thetaDiff = std::max(thetaDiff, std::abs(thetaFeas[j] - thetaPrev[j]));
 //        }
 //        if (!std::isfinite(thetaDiff)) { thetaDiff = std::numeric_limits<double>::infinity(); }

 //        bool sameOrders = true;
 //        for (int s = 0; s < m && sameOrders; ++s) {
 //            if (ordersPhys[s] != ordPhysPrev[s] || ordersVirt[s] != ordVirtPrev[s]) {
 //                sameOrders = false;
 //            }
 //        }

 //        if (it + 1 >= REPAIR_MIN_ITERS &&
 //            tauDiff <= REPAIR_TOL_TIME &&
 //            thetaDiff <= REPAIR_TOL_TIME &&
 //            sameOrders) {
 //            break;
 //        }
 //    }

 //    VecInt lastPhysNode(m, 0);
 //    for (int s = 0; s < m; ++s) {
 //        lastPhysNode[s] = ordersPhys[s].empty()
 //            ? s
 //            : taskNode[ordersPhys[s].back()];
 //    }
 //    return RepairResult{ tauFeas, thetaFeas, tFeas, ordersPhys, ordersVirt, lastPhysNode };
 //}

} // namespace mrta