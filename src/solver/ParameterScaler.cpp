#include "ParameterScaler.h"

#include "../domain/InstanceData.h"
#include "../domain/SolverParameters.h"

#include <algorithm>
#include <cmath>

namespace mrta {

    SolverParameters ParameterScaler::build(const InstanceData& instance) const {
        SolverParameters out{};

        // Number of multi-robot tasks.
        // These tasks require more than one robot to participate.
        int nMR = 0;

        // Number of single-robot physical tasks.
        // These are non-virtual tasks that are not marked as MR.
        int nSR = 0;

        // Number of virtual tasks.
        // Virtual tasks are not tied to a physical service location in the same way.
        int nVirt = 0;

        // Count tasks by category.
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

        // Instance-size scaling factor.
        //
        // Anchor:
        // - if m*n = 48, scale = 1
        //
        // Used to grow some local-search parameters gently with problem size.
        const double scale = std::sqrt(static_cast<double>(instance.m * instance.n) / 48.0);

        // Currently not used directly below.
        // Kept because virtual count is still conceptually important and may be reused later.
        (void)nVirt;

        // Average number of tasks per robot.
        // Used to scale load-balancing pressure in ADMM.
        const double taskPerRobot =
            static_cast<double>(instance.n) / std::max(instance.m, 1);

        // =========================
        // ADMM parameter selection
        // =========================
        auto& AP = out.admm;

        // Penalty for assignment/cardinality consistency.
        // Higher -> stronger enforcement of cardinality agreement, but may become stiff/oscillatory.
        // Lower -> softer enforcement, often more stable but slower to clean up discrete structure.
        AP.rhoCard = 8.0;

        // Penalty for synchronization consistency.
        // Scaled by sqrt(m) so larger teams are not over-penalized.
        // Higher -> stronger synchronization pressure, but can destabilize oscillatory instances.
        // Lower -> weaker sync enforcement, may help stability but can slow convergence.
        AP.rhoSync = 8.0 / std::sqrt(static_cast<double>(instance.m));

        // Dual/assignment update relaxation.
        // Higher -> faster movement, more aggressive, more risk of oscillation.
        // Lower -> safer but slower.
        AP.alphaY = 0.05;

        // Time-related update relaxation.
        // Higher -> timing variables move faster, useful if timing lags behind assignment updates.
        // Lower -> more conservative timing evolution.
        AP.alphaTau = 2.0;

        // Frequency of shaping/stabilization logic inside ADMM.
        // Smaller -> more frequent shaping.
        // Larger -> less frequent shaping, less overhead, but slower correction.
        AP.shapingPeriod = 10;

        // Load-balancing regularization.
        // Scaled by average tasks per robot.
        // Higher -> stronger push toward balanced robot workloads.
        // Lower -> more freedom for uneven but potentially lower-makespan assignments.
        AP.lambdaLoad = 15.0 * taskPerRobot / 5.0;

        // Multi-robot structure regularization.
        // Higher -> stronger bias toward cleaner MR behavior/consensus.
        // Lower -> less structure, potentially more flexibility.
        AP.lambdaMr = 6.0;

        // Relaxation on dual updates in normal ADMM.
        // Higher -> more aggressive dual motion.
        // Lower -> more damping.
        AP.dualRelax = 0.2;

        // Relaxation on theta/time consensus updates in normal ADMM.
        // Higher -> faster time consensus movement.
        // Lower -> more conservative timing behavior.
        AP.thetaRelax = 0.5;

        // Cardinality convergence tolerance.
        // Smaller -> stricter convergence requirement.
        // Larger -> easier convergence declaration, possibly rougher assignment consistency.
        AP.tolCard = 1e-6;

        // Synchronization convergence tolerance.
        // Smaller -> stricter sync requirement.
        // Larger -> easier sync convergence, but potentially rougher timing coordination.
        AP.tolSync = 1e-4;

        // Maximum number of ADMM iterations in the main phase.
        // Higher -> gives ADMM more time to converge.
        // Lower -> faster cutoff, but may stop before a good consensus is reached.
        AP.maxIter = 200 + 50 * instance.m;

        // Number of repair iterations used during ADMM feasibility projection.
        // Higher -> better repaired schedules, but more expensive.
        // Lower -> faster but rougher feasibility recovery.
        AP.nRepair = std::max(30, 10 * instance.m);

        // Fallback dual relaxation used if main ADMM fails to converge.
        // More conservative than the normal setting.
        // Lower -> safer in oscillatory cases.
        AP.fallbackDualRelax = 0.08;

        // Fallback theta/time relaxation.
        // Lower -> more stable but slower.
        AP.fallbackThetaRelax = 0.25;

        // Fallback synchronization penalty.
        // Lower -> softer sync pressure during fallback, may reduce oscillation.
        // If you changed this elsewhere in your project, keep it aligned there too.
        AP.fallbackRhoSync = AP.rhoSync * 0.5;

        // Fallback sync convergence tolerance.
        // Larger -> easier fallback convergence.
        AP.fallbackTolSync = 1e-3;

        // Extra fallback iterations beyond the main ADMM budget.
        // Higher -> gives fallback more chance to recover.
        AP.fallbackMaxIter = AP.maxIter + 200;

        // ===============================
        // Local-search parameter selection
        // ===============================
        auto& LSP = out.localSearch;

        // Number of outer LS passes.
        // Higher -> more neighborhood passes, potentially better solutions, slower runtime.
        // Lower -> faster but less exploration.
        LSP.nOuter = 20 + 5 * instance.m;

        // Number of inner order-improvement attempts.
        // Higher -> stronger route/order polishing, slower.
        // Lower -> faster but less order refinement.
        LSP.nInnerOrder = std::max(20, static_cast<int>(std::lround(60.0 * instance.n / 12.0)));

        // Number of MR batch-order trials.
        // Higher -> more exploration of MR sequence permutations.
        // Lower -> faster but narrower MR order search.
        LSP.mrBatchNTrials = std::max(8, 3 * nMR);

        // Number of coupled/checkpointed MR order batch trials.
        // Higher -> deeper MR order exploration.
        // Lower -> faster but less chance to escape local MR-order traps.
        LSP.mrMoveNumBatchTrials = std::max(5, 2 * nMR);

        // Size of the MR batch reordered at once.
        //
        // Small instances:
        // - allow using all MR tasks
        //
        // Larger instances:
        // - cap to keep factorial growth manageable
        //
        // Higher -> richer but much more expensive permutations.
        // Lower -> cheaper but less expressive batch moves.
        LSP.mrBatchK = std::min(4, std::max(2, static_cast<int>(std::lround(
            static_cast<double>(nMR) / 3.0))));

        // Maximum number of permutations tested for one MR batch.
        //
        // For small nMR, allow up to 120 permutations.
        // For larger nMR, cap aggressively.
        //
        // Higher -> more complete local search, much more expensive.
        // Lower -> faster but may miss good reorderings.
        LSP.mrBatchMaxPerms = (nMR <= 8)
            ? 120
            : std::min(24, std::max(2, 4 * nMR));

        // Maximum number of MR tasks considered in MR-move neighborhoods.
        // Higher -> more MR tasks explored each pass.
        // Lower -> faster, more selective MR search.
        LSP.mrMoveMaxTasks = std::max(10, nMR);

        // Maximum insertion-position trials when relocating/reinserting MR tasks.
        // Higher -> better position search.
        // Lower -> faster but more myopic.
        LSP.mrMoveMaxPosTrials = std::max(4, static_cast<int>(std::lround(static_cast<double>(instance.m) / 2.0)));

        // Quantile cutoff for SR swap filtering.
        // Higher -> stricter filter, fewer swaps survive.
        // Lower -> more SR swap candidates explored.
        LSP.srSwapQuantileCutoff = std::min(0.35, 0.3 + 0.05 * nSR);

        // Gap-window look-back length.
        // Higher -> larger reorder window behind a chosen anchor gap.
        // Lower -> more local search only.
        LSP.gapWindowBack = std::min(5, std::max(2, static_cast<int>(std::lround(static_cast<double>(instance.n) / (2.0 * instance.m)))));

        // Gap-window look-forward length.
        // Higher -> wider forward reorder context.
        // Lower -> more local/cheaper.
        LSP.gapWindowForward = std::min(5, std::max(1, static_cast<int>(std::lround(static_cast<double>(instance.n) / (3.0 * instance.m)))));

        // Maximum number of candidate gaps inspected by gap-window.
        // Higher -> broader gap exploration.
        // Lower -> faster but less likely to catch useful windows.
        LSP.gapWindowMaxGaps = std::max(10, instance.m + 2);

        // Maximum MR tasks considered per outer LS pass.
        // Higher -> broader MR reallocation exploration.
        // Lower -> faster and more selective.
        LSP.maxMrTasksPerOuter = std::max(20, nMR * 2);

        // Maximum MR reallocation candidates checked per MR task.
        // Higher -> more thorough MR reassignment.
        // Lower -> faster.
        LSP.maxMrCandidatesPerTask = std::max(10, instance.m * 2);

        // Maximum tasks per robot considered by some LS loops/structures.
        // Higher -> more permissive for dense schedules.
        // Lower -> tighter control, potentially unsafe if too small.
        LSP.maxTasksPerRobot = std::max(50, instance.n);

        // Number of RR (ruin-recreate) trials.
        // Higher -> more diversification.
        // Lower -> faster, less diversification.
        LSP.rrNumTrials = std::max(8, static_cast<int>(std::lround(1.5 * instance.n)));

        // Maximum MR tasks modified in one RR move.
        // Higher -> larger neighborhoods, more expensive.
        // Lower -> smaller neighborhoods, faster.
        LSP.rrMaxMrTasks = std::min(4, nMR);

        // Maximum MR swap options generated per MR task in RR.
        // Higher -> more swap variety.
        // Lower -> faster generation.
        LSP.rrMaxMrSwapsPerTask = std::min(4, instance.m);

        // Maximum number of MR combo patterns tested in RR.
        // Higher -> more structural search.
        // Lower -> less combinatorial cost.
        LSP.rrMaxMrCombos = 20;

        // Maximum number of SR/virtual tasks stripped in RR.
        // Higher -> more destructive ruin phase, potentially better diversification.
        // Lower -> gentler RR, cheaper.
        LSP.rrMaxSrStrip = std::min(10, nSR + nVirt);

        // Maximum permutations considered in RR ordering subroutines.
        // Higher -> more complete reorder exploration.
        // Lower -> faster.
        LSP.rrMaxPerms = 24;

        // Maximum candidate SR insertion positions in RR reconstruction.
        // Higher -> stronger reconstruction.
        // Lower -> cheaper.
        LSP.rrMaxSrPos = std::min(6, instance.n);

        // Number of repair iterations used in relocation-style evaluations.
        // Higher -> more accurate repaired candidate evaluation.
        // Lower -> faster but rougher estimates.
        LSP.nRepairReloc = 110;

        // Backtracking depth in RR dive search.
        // Higher -> more combinational search.
        // Lower -> greedier and faster.
        LSP.rrDiveBacktrack = std::min(3, nSR);

        // Number of random RR samples.
        // Higher -> more stochastic exploration.
        // Lower -> faster.
        LSP.rrNumRandomSamples = std::min(
            200,
            std::max(50, static_cast<int>(std::lround(std::pow(3.0, nSR) / 5.0)))
        );

        // Maximum segment length in x-opt / exchange neighborhoods.
        // Higher -> more powerful exchange moves, more expensive.
        // Lower -> simpler exchanges.
        LSP.xoptMaxSegLen = 6;

        // Maximum robot-pair count considered in x-opt.
        // Higher -> broader inter-robot exchange search.
        // Lower -> cheaper.
        LSP.xoptMaxPairs = std::min(20, instance.m * (instance.m - 1));

        // Fraction of near-critical robots/tasks emphasized in x-opt.
        // Smaller -> tighter focus near bottlenecks.
        // Larger -> broader exploration.
        LSP.xoptNearCritFrac = 0.15;

        // Number of RR2 trials.
        // Higher -> more RR2 attempts, more runtime.
        // Lower -> fewer strong perturbation attempts.
        LSP.rr2NumTrials = 3;

        // Maximum number of MR combo structures explored in RR2.
        // Higher -> stronger diversification, more expensive.
        // Lower -> cheaper but narrower RR2.
        LSP.rr2MaxMrCombos = 10;

        // Number of perturbation samples in non-exhaustive RR2.
        // Higher -> more coverage.
        // Lower -> faster
        LSP.rr2NumPerturbations = 3;

        // RR2 post-candidate order-polish iterations.
        // Higher -> better polishing after RR2.
        // Lower -> faster.
        LSP.rr2PolishNInner = std::max(3, std::min(20, static_cast<int>(std::round(10.0 * scale))));

        // Exhaustive RR2 threshold.
        // If the number of candidate assignments is <= this limit, enumerate all.
        // Higher -> more exhaustive search, can become very expensive.
        // Lower -> switches earlier to approximate/perturbative RR2.
        LSP.rr2ExhaustLimit = std::max(2000, std::min(10000,
            static_cast<int>(std::round(1500.0 * scale))));
        // Number of top screened RR2 candidates sent to deep evaluation in exhaustive mode.
        // Higher -> more deep checks, slower but possibly better.
        // Lower -> faster but may miss recoverable candidates.
        LSP.rr2ExhaustTopK = std::max(5, std::min(50, static_cast<int>(std::round(15.0 * scale))));

        // Number of repair iterations for the initial LS state.
        // Higher -> better initial feasibility polishing.
        // Lower -> faster startup.
        LSP.nRepairInit = AP.nRepair;

        // Number of repair iterations for frozen-order evaluations.
        // Higher -> more accurate frozen-order evaluation.
        // Lower -> faster but rougher.
        LSP.nRepairFrozen = AP.nRepair;

        // Maximum allowed degradation over current incumbent before RR2 deep/thorough polish is skipped.
        // Higher -> more candidates receive expensive polishing.
        // Lower -> cheaper RR2, but more aggressive pruning.
        LSP.rr2PolishMaxDegradation = 150.0;

        // Maximum allowed degradation for a cheap RR2 candidate before it is skipped
        // before deep/thorough evaluation.
        //
        // Higher -> more candidates survive to deepEval/thoroughEval.
        // Lower -> stronger pruning and faster runtime.
        LSP.rr2EvalMaxDegradation = 300.0;

        // If the exhaustive RR2 neighborhood is already mostly cached and only a very small
        // number of new candidates remain, mark the combo as exhausted.
        //
        // Higher -> you tolerate a few more unseen assignments before calling it exhausted.
        // Lower -> more aggressive exhausted-combo marking.
        LSP.rr2CacheDominatedMaxNew = 5;

        // If cached assignments are at least this close to the full assignment set,
        // treat the RR2 combo as cache-dominated and mark it exhausted.
        //
        // Example:
        // - nAssign = 27
        // - threshold = 2
        // - if cached >= 25 and new <= rr2CacheDominatedMaxNew, mark exhausted
        //
        // Higher -> more aggressive exhausted-combo marking.
        // Lower -> more conservative.
        //LSP.rr2CacheDominatedSlack = 2;

        // ==========================
        // Lightweight polish settings
        // ==========================

        // MR batch trials during post-assignment polish.
        // Higher -> stronger polish after assignment changes.
        // Lower -> faster polish.
        LSP.polishMrBatchTrials = std::min(8, LSP.mrBatchNTrials);

        // Restrict polish MR batch search to critical robots only.
        // true  -> cheaper, focused polish
        // false -> broader but more expensive polish
        LSP.polishMrBatchCriticalOnly = true;

        // MR move batch trials during lightweight polish.
        // Higher -> stronger polish.
        // Lower -> faster.
        LSP.polishMrMoveBatchTrials = std::min(3, LSP.mrMoveNumBatchTrials);

        // Maximum insertion positions tried during lightweight MR move polish.
        // Higher -> more complete.
        // Lower -> cheaper.
        LSP.polishMrMoveMaxPos = std::min(2, LSP.mrMoveMaxPosTrials);

        // ============================
        // Relocation-specific LS tuning
        // ============================

        // MR batch trials when evaluating MR relocation neighborhoods.
        // Higher -> better relocation exploration.
        // Lower -> faster.
        LSP.relocateMrBatchTrials = 5;

        // MR move batch trials during relocation exploration.
        // Higher -> stronger relocation polish.
        // Lower -> faster.
        LSP.relocateMrMoveBatchTrials = 3;

        // Maximum insertion positions tried during relocation-based MR adjustments.
        // Higher -> more accurate relocation evaluation.
        // Lower -> cheaper.
        LSP.relocateMrMoveMaxPos = 2;

        return out;
    }

} // namespace mrta