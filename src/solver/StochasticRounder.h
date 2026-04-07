#pragma once
// ============================================================================
// StochasticRounder.h
//
// Randomized multi-sample rounding for the fractional ADMM z-matrix.
//
// Instead of a single deterministic greedy rounding (AssignmentRounder),
// this class generates N candidate binary assignments by sampling from
// the ADMM fractional values, repairs and scores each, and returns the
// best.  The deterministic rounding from AssignmentRounder is also
// included as one of the candidates so we never do worse.
//
// CHANGE vs original:
//   StochasticRoundingResult now also carries  zTopK  — the top-K
//   binary assignments after deep evaluation (not just the single best).
//   This is the only breaking change; all existing code that only reads
//   zBest / workHBest / mkspBest / nEvaluated / bestIdx is unaffected.
//
// ============================================================================

#ifndef MRTA_STOCHASTIC_ROUNDER_H
#define MRTA_STOCHASTIC_ROUNDER_H

#include "AssignmentRounder.h"
#include "ScheduleRepairer.h"
#include "ScheduleScorer.h"
#include "DerivedDataBuilder.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mrta {

    // ── Options ─────────────────────────────────────────────────────────
    struct StochasticRoundingOptions {
        int  nSamples = 128;  // random binary candidates to generate
        int  nRepairIters = 5;    // repair iterations for cheap screening
        int  nRepairItersDeep = 200;  // repair iterations for top-K deep eval
        int  nRebalanceSrPasses = 6;    // SR rebalance passes on final best
        int  nRebalanceMrPasses = 4;    // MR participant rebalance on final best
        int  topKForDeepEval = 10;   // how many cheap-best to evaluate deeply
        bool includeDeterministic = true; // also run your existing AssignmentRounder
        unsigned long seed = 0;    // 0 → seed from high-res clock
        double temperature = 1.0;  // softmax temperature (lower = greedier)
    };

    // ── Result ──────────────────────────────────────────────────────────
    struct StochasticRoundingResult {
        MatrixDouble              zBest;     // best binary assignment  (m × n)
        VecDouble                 workHBest; // estimated workload per robot for zBest
        double                    mkspBest;  // makespan of the best candidate
        int                       nEvaluated;// how many candidates were scored
        int                       bestIdx;   // which candidate was best (-1 = deterministic)

        // Top-K assignments after deep evaluation, ordered best → worst.
        // Always contains at least zBest as element [0].
        // Used by IlsSolver::runMultiStart() to seed independent ILS runs.
        std::vector<MatrixDouble> zTopK;
        std::vector<double>       mkspTopK; // matching makespans for zTopK
    };

    // ── The Rounder ─────────────────────────────────────────────────────
    class StochasticRounder {
    public:
        StochasticRounder(
            const ScheduleRepairer& repairer,
            const ScheduleScorer& scorer)
            : repairer_(repairer), scorer_(scorer) {
        }

        // ================================================================
        // run()  —  main entry point
        // ================================================================
        StochasticRoundingResult run(
            const MatrixDouble& zFrac,
            const InstanceData& inst,
            const DerivedData& derived,
            const VecDouble& theta,
            const VecDouble& t0,
            const StochasticRoundingOptions& opts) const
        {
            const int m = inst.m;
            const int n = inst.n;

            // ── RNG setup ────────────────────────────────────────────────
            const unsigned long actualSeed = (opts.seed != 0)
                ? opts.seed
                : static_cast<unsigned long>(
                    std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());
            std::mt19937 rng(actualSeed);

            // ── Precompute per-task capable robot lists ──────────────────
            std::vector<VecInt> capLists(n);
            for (int j = 0; j < n; ++j)
                for (int s = 0; s < m; ++s)
                    if (inst.cap[s][j] > 0.5)
                        capLists[j].push_back(s);

            // ── Service estimate matrix ──────────────────────────────────
            const MatrixDouble svcEst =
                AssignmentRounder::buildLocalServiceEstimate(inst);

            // ── Internal candidate structure ─────────────────────────────
            struct Candidate {
                MatrixDouble z;
                VecDouble    workH;
                double       mkspCheap = std::numeric_limits<double>::infinity();
                double       mkspDeep = std::numeric_limits<double>::infinity();
                int          idx = -1;  // -1 = deterministic
            };

            std::vector<Candidate> candidates;
            candidates.reserve(opts.nSamples + 1);

            // ────────────────────────────────────────────────────────────
            // (A) Deterministic candidate
            // ────────────────────────────────────────────────────────────
            if (opts.includeDeterministic) {
                RepairedAssignmentMakespanEvaluator eval(repairer_, scorer_);
                AssignmentRounder detRounder(eval);
                const RoundingResult detResult =
                    detRounder.roundAndClean(zFrac, inst, derived, theta, t0);

                Candidate c;
                c.z = detResult.zHard;
                c.workH = detResult.workH;
                c.idx = -1;
                candidates.push_back(std::move(c));
            }

            // ────────────────────────────────────────────────────────────
            // (B) Stochastic candidates
            // ────────────────────────────────────────────────────────────
            for (int sample = 0; sample < opts.nSamples; ++sample) {
                Candidate c;
                c.z.assign(m, VecDouble(n, 0.0));
                c.workH.assign(m, 0.0);
                c.idx = sample;

                VecDouble loadH(m, 0.0);
                VecDouble workH(m, 0.0);

                VecInt taskOrder(n);
                std::iota(taskOrder.begin(), taskOrder.end(), 0);
                std::shuffle(taskOrder.begin(), taskOrder.end(), rng);

                // Scarce tasks first (stable so random shuffle breaks ties)
                std::stable_sort(taskOrder.begin(), taskOrder.end(),
                    [&](int a, int b) {
                        return capLists[a].size() < capLists[b].size();
                    });

                bool feasible = true;

                for (int j : taskOrder) {
                    const VecInt& caps = capLists[j];
                    if (caps.empty()) { feasible = false; break; }

                    int kj = 1;
                    if (inst.isMR[j])      kj = std::max(1, inst.k[j]);
                    if (inst.isVirtual[j]) kj = 1;
                    kj = std::min(kj, static_cast<int>(caps.size()));

                    // ── Build softmax weights ─────────────────────────
                    const int nCaps = static_cast<int>(caps.size());
                    VecDouble logits(nCaps);
                    double maxLogit = -std::numeric_limits<double>::infinity();

                    for (int ci = 0; ci < nCaps; ++ci) {
                        const int s = caps[ci];
                        double w = zFrac[s][j];
                        w -= 0.15 * loadH[s];
                        w -= 0.10 * workH[s] / std::max(1.0, workH[s] + 1.0);
                        w += 0.10 * derived.scarcityRobot[s];

                        logits[ci] = w / std::max(opts.temperature, 0.01);
                        maxLogit = std::max(maxLogit, logits[ci]);
                    }

                    double    sumExp = 0.0;
                    VecDouble probs(nCaps);
                    for (int ci = 0; ci < nCaps; ++ci) {
                        probs[ci] = std::exp(logits[ci] - maxLogit);
                        sumExp += probs[ci];
                    }
                    for (int ci = 0; ci < nCaps; ++ci)
                        probs[ci] /= sumExp;

                    // ── Sample kj robots WITHOUT replacement ──────────
                    VecInt            chosen;
                    chosen.reserve(kj);
                    std::vector<bool> used(nCaps, false);

                    for (int pick = 0; pick < kj; ++pick) {
                        double cumSum = 0.0;
                        for (int ci = 0; ci < nCaps; ++ci)
                            if (!used[ci]) cumSum += probs[ci];

                        if (cumSum <= 1e-30) {
                            for (int ci = 0; ci < nCaps; ++ci) {
                                if (!used[ci]) {
                                    chosen.push_back(caps[ci]);
                                    used[ci] = true;
                                    break;
                                }
                            }
                            continue;
                        }

                        std::uniform_real_distribution<double> dist(0.0, cumSum);
                        const double dart = dist(rng);

                        double running = 0.0;
                        int    pickedCI = -1;
                        for (int ci = 0; ci < nCaps; ++ci) {
                            if (used[ci]) continue;
                            running += probs[ci];
                            if (running >= dart) { pickedCI = ci; break; }
                        }

                        if (pickedCI < 0) {
                            for (int ci = nCaps - 1; ci >= 0; --ci) {
                                if (!used[ci]) { pickedCI = ci; break; }
                            }
                        }

                        if (pickedCI >= 0) {
                            chosen.push_back(caps[pickedCI]);
                            used[pickedCI] = true;
                        }
                    }

                    for (int s : chosen) {
                        c.z[s][j] = 1.0;
                        loadH[s] += 1.0;
                        workH[s] += svcEst[s][j];
                    }
                }

                if (feasible) {
                    c.workH = workH;
                    candidates.push_back(std::move(c));
                }
            }

            std::cout << "\n=== STOCHASTIC ROUNDING ===\n";
            std::cout << "Generated " << candidates.size() << " candidates"
                << " (" << (opts.includeDeterministic ? "incl." : "excl.")
                << " deterministic)\n";

            // ────────────────────────────────────────────────────────────
            // Phase 1: Cheap screening
            // ────────────────────────────────────────────────────────────
            for (auto& c : candidates)
                c.mkspCheap = quickEval(c.z, inst, theta, t0, opts.nRepairIters);

            std::vector<int> sortIdx(candidates.size());
            std::iota(sortIdx.begin(), sortIdx.end(), 0);
            std::sort(sortIdx.begin(), sortIdx.end(),
                [&](int a, int b) {
                    return candidates[a].mkspCheap < candidates[b].mkspCheap;
                });

            const double cheapBest = candidates[sortIdx.front()].mkspCheap;
            const double cheapWorst = candidates[sortIdx.back()].mkspCheap;
            const double cheapMedian = candidates[sortIdx[sortIdx.size() / 2]].mkspCheap;

            std::cout << "Cheap screening:\n"
                << "  best=" << cheapBest
                << "  median=" << cheapMedian
                << "  worst=" << cheapWorst << "\n";

            // ────────────────────────────────────────────────────────────
            // Phase 2: Deep evaluation of top-K
            // ────────────────────────────────────────────────────────────
            const int topK = std::min(opts.topKForDeepEval,
                static_cast<int>(candidates.size()));

            double bestMksp = std::numeric_limits<double>::infinity();
            int    bestCandIdx = sortIdx[0];

            // Deep-eval each of the top-K (sorted by cheap score)
            for (int rank = 0; rank < topK; ++rank) {
                const int ci = sortIdx[rank];
                auto& c = candidates[ci];

                c.mkspDeep = quickEval(c.z, inst, theta, t0, opts.nRepairItersDeep);

                std::cout << "  Deep rank " << rank
                    << " (src=" << (c.idx < 0 ? "det" : "rnd")
                    << " #" << c.idx << "): cheap="
                    << c.mkspCheap << " > deep=" << c.mkspDeep << "\n";

                if (c.mkspDeep < bestMksp) {
                    bestMksp = c.mkspDeep;
                    bestCandIdx = ci;
                }
            }

            // ────────────────────────────────────────────────────────────
            // Phase 3: Rebalance overall best
            // ────────────────────────────────────────────────────────────
            auto& bestCand = candidates[bestCandIdx];

            if (opts.nRebalanceSrPasses > 0 || opts.nRebalanceMrPasses > 0) {
                RepairedAssignmentMakespanEvaluator eval(repairer_, scorer_);
                AssignmentRounder rebalancer(eval);

                std::cout << "Rebalancing best candidate...\n";

                bestCand.z = rebalancer.rebalanceSrByMakespan(
                    bestCand.z, inst, theta, t0,
                    opts.nRebalanceSrPasses, 6, 3);

                bestCand.z = rebalancer.rebalanceMrParticipantsByMakespan(
                    bestCand.z, inst, theta, t0,
                    opts.nRebalanceMrPasses, 8, 4);

                bestCand.mkspDeep = quickEval(
                    bestCand.z, inst, theta, t0, opts.nRepairItersDeep);
            }

            // ── Final workH for best ─────────────────────────────────────
            VecDouble finalWorkH(m, 0.0);
            for (int s = 0; s < m; ++s)
                for (int j = 0; j < n; ++j)
                    if (bestCand.z[s][j] > 0.5)
                        finalWorkH[s] += svcEst[s][j];

            std::cout << "Best candidate: "
                << (bestCand.idx < 0 ? "deterministic" : "stochastic")
                << " #" << bestCand.idx
                << " | mksp=" << bestCand.mkspDeep
                << " (after rebalance)\n";
            std::cout << "=== END STOCHASTIC ROUNDING ===\n\n";

            // ────────────────────────────────────────────────────────────
            // Build zTopK / mkspTopK  (best → worst after deep eval)
            // Re-sort the top-K slice by deep makespan so the caller gets
            // them in quality order.
            // ────────────────────────────────────────────────────────────
            std::vector<int> deepRanked(topK);
            std::iota(deepRanked.begin(), deepRanked.end(), 0);
            std::sort(deepRanked.begin(), deepRanked.end(),
                [&](int a, int b) {
                    return candidates[sortIdx[a]].mkspDeep
                        < candidates[sortIdx[b]].mkspDeep;
                });

            std::vector<MatrixDouble> zTopK;
            std::vector<double>       mkspTopK;
            zTopK.reserve(topK);
            mkspTopK.reserve(topK);

            for (int rank : deepRanked) {
                const auto& c = candidates[sortIdx[rank]];
                // Use the rebalanced z for rank-0 (the overall best).
                if (sortIdx[rank] == bestCandIdx) {
                    zTopK.push_back(bestCand.z);
                    mkspTopK.push_back(bestCand.mkspDeep);
                }
                else {
                    zTopK.push_back(c.z);
                    mkspTopK.push_back(c.mkspDeep);
                }
            }

            return StochasticRoundingResult{
                bestCand.z,
                finalWorkH,
                bestCand.mkspDeep,
                static_cast<int>(candidates.size()),
                bestCand.idx,
                std::move(zTopK),
                std::move(mkspTopK)
            };
        }

    private:
        const ScheduleRepairer& repairer_;
        const ScheduleScorer& scorer_;

        double quickEval(
            const MatrixDouble& z,
            const InstanceData& inst,
            const VecDouble& theta,
            const VecDouble& t0,
            int                 nRepairIters) const
        {
            RepairOptions ropts;
            ropts.nIters = nRepairIters;
            ropts.freezeOrders = false;

            const RepairResult rep = repairer_.repairPushforward(
                z, inst, theta, t0, ropts);

            const ScheduleScore score = scorer_.scoreScheduleExact(
                inst, z, rep.tauFeas, rep.ordersPhys, rep.ordersVirt,
                rep.lastPhysNode);

            return score.mksp;
        }
    };

} // namespace mrta

#endif // MRTA_STOCHASTIC_ROUNDER_H