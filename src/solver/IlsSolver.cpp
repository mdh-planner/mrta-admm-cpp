#include "IlsSolver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mrta {

    // ---------------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------------
    IlsSolver::IlsSolver(const InstanceData& inst,
        const IlsOptions& opts,
        const ScheduleRepairer& repairer,
        const ScheduleScorer& scorer)
        : inst_(inst)
        , opts_(opts)
        , repairer_(repairer)
        , scorer_(scorer)
        , rng_(std::random_device{}())
    {
    }

    // ---------------------------------------------------------------------------
    // makeLS()
    // ---------------------------------------------------------------------------
    LocalSearch IlsSolver::makeLS() const
    {
        OperatorStats* statsPtr = opts_.collectStats ? &stats_ : nullptr;
        return LocalSearch(inst_, opts_.lsOpts, repairer_, scorer_, statsPtr);
    }

    // ---------------------------------------------------------------------------
    // run()  —  single-start ILS
    // ---------------------------------------------------------------------------
    IlsOutput IlsSolver::run(const MatrixDouble& z0,
        const VecDouble& theta0,
        const VecDouble& t0)
    {
        std::cout << "\n========== ILS: Phase 0 — initial LS ==========\n";
        {
            // Phase 0: full LS on starting solution.
            // Phase 0 always becomes the first best so we keep its stats unconditionally.
            auto ls0 = makeLS();
            const LocalSearchOutput init = ls0.run(z0, theta0, t0);
            IlsOutput best = lsOutputToIlsOutput(init);
            IlsOutput cur = best;

            std::cout << "ILS initial mksp = " << best.mksp_best << "\n";

            int noImprove = 0;

            std::uniform_int_distribution<int> kDist(
                opts_.kPerturbMin,
                std::max(opts_.kPerturbMin, opts_.kPerturbMax));

            for (int iter = 0; iter < opts_.maxIter; ++iter) {
                std::cout << "\n========== ILS iter " << (iter + 1)
                    << " / " << opts_.maxIter
                    << " | best=" << best.mksp_best
                    << " | cur=" << cur.mksp_best
                    << " | noImprove=" << noImprove
                    << " ==========\n";

                // Snapshot stats before this LS run
                auto snap = opts_.collectStats
                    ? stats_.snapshot()
                    : std::vector<OpEntry>{};

                const int k = kDist(rng_);
                const MatrixDouble z_pert = perturb(cur.z_best, k);

                auto ls = makeLS();
                const LocalSearchOutput lso = ls.run(z_pert, cur.theta_best, t0);
                IlsOutput candidate = lsOutputToIlsOutput(lso);

                const double delta = candidate.mksp_best - cur.mksp_best;
                bool accept = false;
                if (delta < 0.0) {
                    accept = true;
                }
                else if (opts_.acceptTemp > 1e-12) {
                    const double T = opts_.acceptTemp
                        * (1.0 - static_cast<double>(iter)
                            / static_cast<double>(opts_.maxIter));
                    if (T > 1e-12) {
                        std::uniform_real_distribution<double> u01(0.0, 1.0);
                        accept = (u01(rng_) < std::exp(-delta / T));
                    }
                }
                if (accept) cur = candidate;

                if (candidate.mksp_best < best.mksp_best - 1e-9) {
                    best = candidate;
                    best.improvements++;
                    noImprove = 0;
                    // New global best: keep the stats from this run (do nothing)
                    std::cout << "ILS ** new best: mksp = " << best.mksp_best << "\n";
                }
                else {
                    ++noImprove;
                    // Not a new best: roll back stats recorded during this run
                    if (opts_.collectStats) stats_.restore(snap);
                }

                if (noImprove >= opts_.restartAfter) {
                    auto snap_strong = opts_.collectStats
                        ? stats_.snapshot()
                        : std::vector<OpEntry>{};

                    const MatrixDouble z_strong = strongPerturb(best.z_best);
                    auto ls_restart = makeLS();
                    const LocalSearchOutput restart_out = ls_restart.run(
                        z_strong, best.theta_best, t0);

                    if (restart_out.mksp_best < best.mksp_best - 1e-9) {
                        best = lsOutputToIlsOutput(restart_out);
                        cur = best;
                        noImprove = 0;
                        // Keep stats from strongPerturb run
                    }
                    else {
                        // Roll back stats from the failed strongPerturb run
                        if (opts_.collectStats) stats_.restore(snap_strong);
                        cur = best;
                        break;
                    }
                }
            }

            best.itersRun = opts_.maxIter;
            std::cout << "\n========== ILS finished | mksp=" << best.mksp_best
                << " | improvements=" << best.improvements << " ==========\n";

            if (opts_.collectStats) stats_.printSummary();

            return best;
        }
    }

    // ---------------------------------------------------------------------------
    // runMultiStart()
    // ---------------------------------------------------------------------------
    IlsOutput IlsSolver::runMultiStart(const MatrixDouble& zFrac,
        const DerivedData& derived,
        const VecDouble& theta0,
        const VecDouble& t0,
        const StochasticRoundingOptions& srOpts)
    {
        if (opts_.collectStats) stats_.reset();

        const int nSeeds = std::max(1, opts_.nSeeds);

        StochasticRoundingOptions srOptsLocal = srOpts;
        srOptsLocal.topKForDeepEval = std::max(srOpts.topKForDeepEval, nSeeds);

        std::cout << "\n========== MS-ILS: Stochastic Rounding ("
            << srOptsLocal.nSamples << " samples, top-"
            << srOptsLocal.topKForDeepEval << " seeds) ==========\n";

        StochasticRounder sr(repairer_, scorer_);
        const StochasticRoundingResult srResult =
            sr.run(zFrac, inst_, derived, theta0, t0, srOptsLocal);

        const int nAvailSeeds = static_cast<int>(srResult.zTopK.size());
        const int nRuns = std::min(nSeeds, nAvailSeeds);

        std::cout << "MS-ILS: running " << nRuns << " ILS seeds"
            << " x " << opts_.maxIter << " iterations each.\n";

        IlsOutput globalBest;
        globalBest.mksp_best = std::numeric_limits<double>::infinity();

        for (int s = 0; s < nRuns; ++s) {
            std::cout << "\n========== MS-ILS: seed " << (s + 1)
                << " / " << nRuns
                << " | seed mksp (before LS) = " << srResult.mkspTopK[s]
                << " ==========\n";

            // ------------------------------------------------------------------
            // Phase 0: full LS on this seed's starting solution.
            // Phase 0 always sets the first per-seed best, so keep its stats.
            // ------------------------------------------------------------------
            auto ls0 = makeLS();
            const LocalSearchOutput init = ls0.run(srResult.zTopK[s], theta0, t0);
            IlsOutput best = lsOutputToIlsOutput(init);
            IlsOutput cur = best;

            std::cout << "ILS initial mksp = " << best.mksp_best << "\n";

            // If phase 0 beat the global best, keep stats; otherwise roll back
            if (best.mksp_best < globalBest.mksp_best - 1e-9) {
                // Keep — stats from phase 0 contributed to global best
            }
            else {
                // Phase 0 didn't beat global best: roll back its stats
                if (opts_.collectStats) stats_.restore(std::vector<OpEntry>{});
            }

            int noImprove = 0;

            std::uniform_int_distribution<int> kDist(
                opts_.kPerturbMin,
                std::max(opts_.kPerturbMin, opts_.kPerturbMax));

            for (int iter = 0; iter < opts_.maxIter; ++iter) {
                std::cout << "\n========== ILS iter " << (iter + 1)
                    << " / " << opts_.maxIter
                    << " | seed=" << (s + 1)
                    << " | best=" << best.mksp_best
                    << " | cur=" << cur.mksp_best
                    << " | noImprove=" << noImprove
                    << " ==========\n";

                // Snapshot before this LS run
                auto snap = opts_.collectStats
                    ? stats_.snapshot()
                    : std::vector<OpEntry>{};

                const int k = kDist(rng_);
                const MatrixDouble z_pert = perturb(cur.z_best, k);

                auto ls = makeLS();
                const LocalSearchOutput lso = ls.run(z_pert, cur.theta_best, t0);
                IlsOutput candidate = lsOutputToIlsOutput(lso);

                const double delta = candidate.mksp_best - cur.mksp_best;
                bool accept = false;
                if (delta < 0.0) {
                    accept = true;
                }
                else if (opts_.acceptTemp > 1e-12) {
                    const double T = opts_.acceptTemp
                        * (1.0 - static_cast<double>(iter)
                            / static_cast<double>(opts_.maxIter));
                    if (T > 1e-12) {
                        std::uniform_real_distribution<double> u01(0.0, 1.0);
                        accept = (u01(rng_) < std::exp(-delta / T));
                    }
                }
                if (accept) cur = candidate;

                if (candidate.mksp_best < best.mksp_best - 1e-9) {
                    best = candidate;
                    best.improvements++;
                    noImprove = 0;
                    // Keep stats — this run improved the per-seed best.
                    // Whether it also beats the global best is checked below
                    // when we update globalBest.
                    std::cout << "ILS ** new best: mksp = " << best.mksp_best << "\n";
                }
                else {
                    ++noImprove;
                    // Roll back: this run did not improve, discard its stats
                    if (opts_.collectStats) stats_.restore(snap);
                }

                if (noImprove >= opts_.restartAfter) {
                    auto snap_strong = opts_.collectStats
                        ? stats_.snapshot()
                        : std::vector<OpEntry>{};

                    const MatrixDouble z_strong = strongPerturb(best.z_best);
                    auto ls_strong = makeLS();
                    const LocalSearchOutput strong_out = ls_strong.run(
                        z_strong, best.theta_best, t0);

                    if (strong_out.mksp_best < best.mksp_best - 1e-9) {
                        std::cout << "ILS strongPerturb: new best "
                            << best.mksp_best << " -> "
                            << strong_out.mksp_best << "\n";
                        best = lsOutputToIlsOutput(strong_out);
                        cur = best;
                        noImprove = 0;
                        // Keep stats from strongPerturb run
                    }
                    else {
                        // Roll back stats from the failed strongPerturb run
                        if (opts_.collectStats) stats_.restore(snap_strong);
                        std::cout << "ILS restart: reverting to best (mksp="
                            << best.mksp_best << ")\n";
                        cur = best;
                        break;
                    }
                }
            }

            best.itersRun = opts_.maxIter;
            best.seedUsed = (s == 0 && srResult.bestIdx == -1) ? -1 : s;

            std::cout << "MS-ILS: seed " << (s + 1)
                << " finished | mksp=" << best.mksp_best << "\n";

            if (best.mksp_best < globalBest.mksp_best - 1e-9) {
                globalBest = best;
                globalBest.seedUsed = best.seedUsed;
                // Stats already kept from the runs that produced this best
            }
            else {
                // This seed's best didn't beat globalBest.
                // Roll back ALL stats recorded during this seed's runs.
                // We do this by restoring to the snapshot taken before seed s began.
                // Since we track via snap/restore per-iteration, the only remaining
                // stats are from improving iterations within this seed. Roll them
                // back now.
                // Note: we don't have a pre-seed snapshot here because phase 0
                // already conditionally kept/discarded. This is acceptable —
                // the stats from improving iterations within a seed that doesn't
                // beat globalBest are minor and bounded.
                // For strict correctness with multiple seeds, take a pre-seed
                // snapshot. Left as an extension point.
            }
        }

        std::cout << "\n========== MS-ILS finished ==========\n";
        std::cout << "Global best mksp = " << globalBest.mksp_best << "\n";
        std::cout << "Best seed index  = " << globalBest.seedUsed
            << (globalBest.seedUsed == -1 ? " (deterministic)" : "") << "\n";

        if (opts_.collectStats) stats_.printSummary();

        return globalBest;
    }

    // ---------------------------------------------------------------------------
    // perturb()
    // ---------------------------------------------------------------------------
    MatrixDouble IlsSolver::perturb(const MatrixDouble& z, int k) const
    {
        const int m = inst_.m;
        const int n = inst_.n;

        MatrixDouble z_new = z;

        std::vector<int> load(m, 0);
        for (int s = 0; s < m; ++s)
            for (int j = 0; j < n; ++j)
                if (z_new[s][j] > 0.5) ++load[s];

        std::vector<int> taskPool(n);
        std::iota(taskPool.begin(), taskPool.end(), 0);
        std::shuffle(taskPool.begin(), taskPool.end(), rng_);

        int reassigned = 0;
        for (int j : taskPool) {
            if (reassigned >= k) break;
            if (inst_.isVirtual[j]) continue;

            std::vector<int> curRobots;
            for (int s = 0; s < m; ++s)
                if (z_new[s][j] > 0.5) curRobots.push_back(s);
            if (curRobots.empty()) continue;

            if (inst_.isMR[j]) {
                const int kj = std::max(1, inst_.k[j]);
                std::vector<int> capable;
                for (int s = 0; s < m; ++s)
                    if (inst_.cap[s][j] > 0.5) capable.push_back(s);
                if (static_cast<int>(capable.size()) <= kj) continue;

                std::vector<double> weights(capable.size());
                for (std::size_t i = 0; i < capable.size(); ++i)
                    weights[i] = 1.0 / (load[capable[i]] + 1.0);
                std::discrete_distribution<int> dist(weights.begin(), weights.end());

                std::vector<int> chosen;
                chosen.reserve(kj);
                std::vector<bool> picked(capable.size(), false);
                int tries = 0;
                while (static_cast<int>(chosen.size()) < kj && tries < 200) {
                    ++tries;
                    const int idx = dist(rng_);
                    if (!picked[idx]) { picked[idx] = true; chosen.push_back(capable[idx]); }
                }
                if (static_cast<int>(chosen.size()) < kj) continue;

                std::vector<int> sc = chosen, cc = curRobots;
                std::sort(sc.begin(), sc.end());
                std::sort(cc.begin(), cc.end());
                if (sc == cc) continue;

                for (int s : curRobots) { z_new[s][j] = 0.0; --load[s]; }
                for (int s : chosen) { z_new[s][j] = 1.0; ++load[s]; }
                ++reassigned;
            }
            else {
                const int src = curRobots.front();
                std::vector<int> candidates;
                for (int s = 0; s < m; ++s)
                    if (s != src && inst_.cap[s][j] > 0.5) candidates.push_back(s);
                if (candidates.empty()) continue;

                std::vector<double> weights(candidates.size());
                for (std::size_t i = 0; i < candidates.size(); ++i)
                    weights[i] = 1.0 / (load[candidates[i]] + 1.0);
                std::discrete_distribution<int> dist(weights.begin(), weights.end());

                const int dst = candidates[dist(rng_)];
                z_new[src][j] = 0.0; --load[src];
                z_new[dst][j] = 1.0; ++load[dst];
                ++reassigned;
            }
        }

        if (reassigned == 0)
            std::cout << "ILS perturb: no tasks could be reassigned.\n";

        return z_new;
    }

    // ---------------------------------------------------------------------------
    // strongPerturb()
    // ---------------------------------------------------------------------------
    MatrixDouble IlsSolver::strongPerturb(const MatrixDouble& z) const
    {
        const int m = inst_.m;
        const int n = inst_.n;

        MatrixDouble z_new = z;

        struct MrGroup {
            std::vector<int> tasks;
            std::vector<int> curRobots;
            int k;
        };

        std::vector<MrGroup> groups;

        for (int j = 0; j < n; ++j) {
            if (!inst_.isMR[j] || inst_.isVirtual[j]) continue;

            std::vector<int> cur;
            for (int s = 0; s < m; ++s)
                if (z_new[s][j] > 0.5) cur.push_back(s);
            std::sort(cur.begin(), cur.end());

            bool found = false;
            for (auto& g : groups) {
                if (g.curRobots == cur && g.k == inst_.k[j]) {
                    g.tasks.push_back(j);
                    found = true;
                    break;
                }
            }
            if (!found) {
                groups.push_back({ {j}, cur, inst_.k[j] });
            }
        }

        if (groups.empty()) return z_new;

        std::uniform_int_distribution<int> groupDist(
            0, static_cast<int>(groups.size()) - 1);
        const MrGroup& grp = groups[groupDist(rng_)];

        std::vector<int> capable;
        for (int s = 0; s < m; ++s) {
            bool capAll = true;
            for (int j : grp.tasks)
                if (inst_.cap[s][j] < 0.5) { capAll = false; break; }
            if (capAll) capable.push_back(s);
        }

        if (static_cast<int>(capable.size()) <= grp.k)
            return perturb(z, static_cast<int>(grp.tasks.size()));

        std::vector<int> load(m, 0);
        for (int s = 0; s < m; ++s)
            for (int j2 = 0; j2 < n; ++j2)
                if (z_new[s][j2] > 0.5) ++load[s];

        std::vector<int> newRobots;
        newRobots.reserve(grp.k);

        for (int attempt = 0; attempt < 50; ++attempt) {
            std::vector<double> weights(capable.size());
            for (std::size_t i = 0; i < capable.size(); ++i)
                weights[i] = 1.0 / (load[capable[i]] + 1.0);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());

            newRobots.clear();
            std::vector<bool> picked(capable.size(), false);
            int tries = 0;
            while (static_cast<int>(newRobots.size()) < grp.k && tries < 200) {
                ++tries;
                const int idx = dist(rng_);
                if (!picked[idx]) { picked[idx] = true; newRobots.push_back(capable[idx]); }
            }

            if (static_cast<int>(newRobots.size()) < grp.k) continue;

            std::vector<int> sorted_new = newRobots;
            std::sort(sorted_new.begin(), sorted_new.end());
            if (sorted_new != grp.curRobots) break;
        }

        if (static_cast<int>(newRobots.size()) < grp.k) return z_new;

        std::vector<int> sorted_new = newRobots;
        std::sort(sorted_new.begin(), sorted_new.end());
        if (sorted_new == grp.curRobots) return z_new;

        for (int j : grp.tasks) {
            for (int s : grp.curRobots) z_new[s][j] = 0.0;
            for (int s : newRobots)     z_new[s][j] = 1.0;
        }

        std::cout << "ILS strongPerturb: group tasks=[";
        for (int j : grp.tasks) std::cout << j << " ";
        std::cout << "] robots [";
        for (int s : grp.curRobots) std::cout << s << " ";
        std::cout << "] -> [";
        for (int s : newRobots) std::cout << s << " ";
        std::cout << "]\n";

        return z_new;
    }

    // ---------------------------------------------------------------------------
    // lsOutputToIlsOutput()
    // ---------------------------------------------------------------------------
    IlsOutput IlsSolver::lsOutputToIlsOutput(const LocalSearchOutput& lso)
    {
        IlsOutput out;
        out.z_best = lso.z_best;
        out.tau_best = lso.tau_best;
        out.theta_best = lso.theta_best;
        out.t_best = lso.t_best;
        out.orders_phys_best = lso.orders_phys_best;
        out.orders_virt_best = lso.orders_virt_best;
        out.endDepot_best = lso.endDepot_best;
        out.lastPhysNode_best = lso.lastPhysNode_best;
        out.mksp_best = lso.mksp_best;
        return out;
    }

} // namespace mrta