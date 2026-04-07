#include "ParameterScaler.h"

#include "../domain/InstanceData.h"
#include "../domain/SolverParameters.h"

#include <algorithm>
#include <cmath>

namespace mrta {

    SolverParameters ParameterScaler::build(const InstanceData& instance) const {
        SolverParameters out{};

        int nMR = 0;
        int nSR = 0;
        int nVirt = 0;

        for (int j = 0; j < instance.n; ++j) {
            if (instance.isVirtual[j]) {
                ++nVirt;
            }
            if (instance.isMR[j]) {
                ++nMR;
            }
            else if (!instance.isVirtual[j]) {
                ++nSR;
            }
        }

        const double scale = std::sqrt(static_cast<double>(instance.m * instance.n) / 48.0);
        (void)nVirt;

        const double taskPerRobot =
            static_cast<double>(instance.n) / std::max(instance.m, 1);

        // =========================
        // ADMM parameter selection  (unchanged)
        // =========================
        auto& AP = out.admm;

        AP.rhoCard = 8.0;
        AP.rhoSync = 8.0 / std::sqrt(static_cast<double>(instance.m));
        AP.alphaY = 0.05;
        AP.alphaTau = 2.0;
        AP.shapingPeriod = 10;
        AP.lambdaLoad = 15.0 * taskPerRobot / 5.0;
        AP.lambdaMr = 6.0;
        AP.dualRelax = 0.2;
        AP.thetaRelax = 0.5;
        AP.tolCard = 1e-6;
        AP.tolSync = 1e-4;
        AP.maxIter = 200 + 50 * instance.m;
        AP.nRepair = std::max(30, 10 * instance.m);
        AP.fallbackDualRelax = 0.08;
        AP.fallbackThetaRelax = 0.25;
        AP.fallbackRhoSync = AP.rhoSync * 0.5;
        AP.fallbackTolSync = 1e-3;
        AP.fallbackMaxIter = AP.maxIter + 200;

        // ===============================
        // Local-search parameter selection
        // ===============================
        auto& LSP = out.localSearch;

        // ── Outer loop ────────────────────────────────────────────────────────
        // CHANGED: 5 → 4.
        // ILS provides diversity across seeds so fewer passes per LS run is fine.
        // Freeing budget from outer iterations funds more RR2 trials instead.
        LSP.nOuter = 4;
        LSP.nInnerOrder = std::max(20, static_cast<int>(
            std::lround(60.0 * instance.n / 12.0)));

        // ── MR batch operators ────────────────────────────────────────────────
        // RESTORED after tabu hash bug fix. Stats confirmed:
        //   improveCoupledMrBatchOrder: 27% hit rate
        //   improveCoupledMrOrder:      19% hit rate
        // Using original formulas — they were correct, the tabu was the problem.
        LSP.mrBatchNTrials = std::max(8, 3 * nMR);
        LSP.mrMoveNumBatchTrials = std::max(5, 2 * nMR);
        LSP.mrBatchK = std::min(4, std::max(3, static_cast<int>(
            std::lround(static_cast<double>(nMR) / 3.0))));
        LSP.mrBatchMaxPerms = (nMR <= 8) ? 120 : std::min(24, std::max(2, 5 * nMR));
        LSP.mrMoveMaxTasks = std::max(10, nMR);
        LSP.mrMoveMaxPosTrials = std::max(4, static_cast<int>(
            std::lround(static_cast<double>(instance.m) / 2.0)));

        // ── SR swap ───────────────────────────────────────────────────────────
        // Unchanged — 23% hit rate, working well.
        //LSP.srSwapQuantileCutoff = std::min(0.35, 0.3 + 0.05 * nSR);
        LSP.srSwapQuantileCutoff = std::min(0.35, 0.3 + 0.05 * nSR);
        // ── Gap window ────────────────────────────────────────────────────────
        // BOOSTED: 31–47% hit rate confirmed across all instances.
        // Wider window = more permutations explored per gap.
        LSP.gapWindowBack = std::min(20, std::max(4, static_cast<int>(
            std::lround(static_cast<double>(instance.n) / (1.5 * instance.m)))));
        LSP.gapWindowForward = std::min(15, std::max(3, static_cast<int>(
            std::lround(static_cast<double>(instance.n) / (2.0 * instance.m)))));
        LSP.gapWindowMaxGaps = std::max(30, instance.m + 10);

        // ── MR reallocation ───────────────────────────────────────────────────
        // BOOSTED: improveMrReallocation hit 46% in latest run — strongest
        // top-level operator. Give it more candidates per task and more tasks
        // per outer pass so it finds as many participant swaps as possible.
        //
        // maxMrTasksPerOuter: was max(20, nMR*2) — boosted to nMR*3
        // maxMrCandidatesPerTask: was max(20, m*2) — boosted to m*m
        //   (m*m covers all possible robot pairs for an MR task swap)
        LSP.maxMrTasksPerOuter = std::max(20, nMR * 3);
        LSP.maxMrCandidatesPerTask = std::max(20, instance.m * instance.m);
        LSP.maxTasksPerRobot = std::max(50, instance.n);

        // ── RR (legacy — unchanged) ───────────────────────────────────────────
        LSP.rrNumTrials = std::max(20, static_cast<int>(std::lround(1.5 * instance.n)));
        LSP.rrMaxMrTasks = std::min(14, nMR);
        LSP.rrMaxMrSwapsPerTask = std::min(14, instance.m);
        LSP.rrMaxMrCombos = 20;
        LSP.rrMaxSrStrip = std::min(10, nSR + nVirt);
        LSP.rrMaxPerms = 24;
        LSP.rrMaxSrPos = std::min(16, instance.n);
        LSP.nRepairReloc = 15;
        LSP.rrDiveBacktrack = std::min(13, nSR);
        LSP.rrNumRandomSamples = std::min(
            2000,
            std::max(500, static_cast<int>(std::lround(std::pow(3.0, nSR) / 5.0)))
        );

        // ── x-opt ─────────────────────────────────────────────────────────────
        LSP.xoptMaxSegLen = 6;
        LSP.xoptMaxPairs = std::min(20, instance.m * (instance.m - 1));
        LSP.xoptNearCritFrac = 0.15;

        // ── Repair iterations ─────────────────────────────────────────────────
        // Unchanged — repair quality matters for accurate evaluation.
        LSP.nRepairInit = AP.nRepair;
        LSP.nRepairFrozen = AP.nRepair;

        // ── RR2 ───────────────────────────────────────────────────────────────
        // BOOSTED: improveRuinRecreate 45–57% hit rate confirmed.
        // Its value comes from deepEval running improveIntraOrder (39% hit rate)
        // AND improveCoupledMrBatchOrder (now 11% hit rate in DeepEval context).
        //
        // rr2NumTrials: was 15, then cut to 5 — RESTORED to 10.
        //   More trials = more diverse MR combo candidates entering deepEval.
        //
        // rr2MaxMrCombos: was 10, then cut to 0 — RESTORED to 15.
        //   MR combos now confirmed useful (CoupledMrBatchOrder 27% TopLevel).
        //   Extra combos explore two-robot simultaneous swaps to bridge the
        //   remaining 12-point gap to optimal.
        //
        // rr2NumPerturbations: was 15, then cut to 5 — RESTORED to 10.
        //
        // rr2PolishNInner: unchanged — this drives the 22K+ DeepEval IntraOrder
        //   calls that produce the bulk of improvement. Do not cut.
        //
        // rr2ExhaustLimit / rr2ExhaustTopK: restored to scale-based values
        //   since MR combos are now active and the exhaustive path may be hit.
        LSP.rr2NumTrials = 20;
        LSP.rr2MaxMrCombos = 25;
        LSP.rr2NumPerturbations = 15;
        LSP.rr2PolishNInner = std::max(15, std::min(20, static_cast<int>(
            std::round(10.0 * scale))));
        LSP.rr2ExhaustLimit = std::max(2000, static_cast<int>(
            std::round(1500.0 * scale)));
        LSP.rr2ExhaustTopK = std::max(30, std::min(50, static_cast<int>(
            std::round(25.0 * scale))));

        LSP.rr2PolishMaxDegradation = 60;
        LSP.rr2EvalMaxDegradation = 200;
        LSP.rr2CacheDominatedMaxNew = 15;

        // ── Polish ────────────────────────────────────────────────────────────
        // MR batch polish: scaled from restored mrBatchNTrials.
        // polishMrBatchCriticalOnly=true keeps it focused on the bottleneck robot.
        LSP.polishMrBatchTrials = std::min(8, LSP.mrBatchNTrials);
        LSP.polishMrBatchCriticalOnly = true;
        LSP.polishMrMoveBatchTrials = std::min(10, LSP.mrMoveNumBatchTrials);
        LSP.polishMrMoveMaxPos = std::min(10, LSP.mrMoveMaxPosTrials);

        // ── Relocation ────────────────────────────────────────────────────────
        // Unchanged from original — these fire inside improveSrAssignment and
        // improveMrReallocation where the MR operators do per-candidate polish.
        LSP.relocateMrBatchTrials = 15;
        LSP.relocateMrMoveBatchTrials = 15;
        LSP.relocateMrMoveMaxPos = 10;

        return out;
    }

} // namespace mrta