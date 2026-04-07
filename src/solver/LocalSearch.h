#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/InstanceData.h"
#include "OperatorStats.h"
#include "ScheduleRepairer.h"
#include "ScheduleScorer.h"

#include <limits>
#include <random>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace mrta {

    // =============================================================================
    //  SR-REINSERTION HEURISTIC
    //  Controls how stripped SR tasks are reinserted after an MR permutation.
    //  Swap `type` to change strategy without touching any loop logic.
    // =============================================================================
    struct SrReinsertionHeuristic {
        enum class Type {
            Greedy,  ///< Always pick the cheapest (robot, pos) pair.
            Grasp,   ///< Sample uniformly from a Restricted Candidate List.
            // Future: RegretK, RandomOrder, …
        };

        Type   type = Type::Grasp;
        double alpha = 0.30; ///< GRASP: RCL threshold = bestCost * (1 + alpha).
        int    trials = 5;    ///< GRASP: number of independent construction trials.
    };


    // =============================================================================
    //  JOINT MR+SR REINSERTION OPERATOR PARAMETERS
    //  Passed to improveJointMrSrReinsertion().
    // =============================================================================
    struct JointMrSrReinsertionParams {
        int  nGroupsToPerturb = 1;
        int  maxPermsPerGroup = 0;
        bool tryAllGroupCombos = true;

        // NEW: also try swapping one robot between the two selected groups.
        // Only meaningful when nGroupsToPerturb >= 2.
        // For each pair of groups, tries replacing one robot from group A
        // with one from group B (if capability allows) before permuting.
        bool tryRobotSwaps = true;
        int  maxRobotSwaps = 4;  // cap on swap candidates per group pair

        SrReinsertionHeuristic srHeuristic;
    };


    // =============================================================================
    //  LOCAL SEARCH TUNING OPTIONS
    //  All algorithm parameters in one place.  Passed by const-ref into LocalSearch.
    // =============================================================================
    struct LocalSearchOptions {

        // ── Outer-loop control ────────────────────────────────────────────────────
        int    nOuter{ 10 };
        int    nInnerOrder{ 10 };
        double minRelImprovementPct{ 0.1 }; ///< Stop if improvement < this %.
        double timeLimitSeconds{ std::numeric_limits<double>::infinity() };

        // ── Tabu / hashing ───────────────────────────────────────────────────────
        bool useTabuHashing{ false }; ///< Globally enable/disable tabu.

        // ── Repair iteration budgets ──────────────────────────────────────────────
        int nRepairInit{ 3 }; ///< Used when building the initial state.
        int nRepairFrozen{ 3 }; ///< Used with frozen orders (intra-order moves).
        int nRepairReloc{ 10 }; ///< Used when the assignment (z) changes.
        int nRepairMR{ 3 }; ///< Used inside improveMrReallocation.

        // ── Polish (called after any assignment change) ───────────────────────────
        int POLISH_N_INNER{ 15 };

        // ── Gap-window operator ───────────────────────────────────────────────────
        double GAP_BEFORE_MIN_LEN{ 5.0 };
        int    GAP_WINDOW_BACK{ 3 };
        int    GAP_WINDOW_FORWARD{ 2 };
        int    GAP_WINDOW_MAX_GAPS{ 5 };
        int    GAP_MOVE_MAX_TASKS{ 5 };

        // ── Coupled MR move operator ──────────────────────────────────────────────
        int  MR_MOVE_MAX_POS_TRIALS{ 3 };
        int  MR_BATCH_N_TRIALS{ 8 };
        bool MR_BATCH_CRITICAL_ONLY{ true };
        int  MR_MOVE_NUM_BATCH_TRIALS{ 5 };
        int  MR_MOVE_BATCH_SIZE{ 3 };
        int  MR_MOVE_MAX_TASKS{ 10 };
        int  MR_BATCH_K{ 3 };
        int  MR_BATCH_MAX_PERMS{ 12 };
        bool MR_BATCH_ENABLE{ true };

        // ── SR-assignment operator ────────────────────────────────────────────────
        double SR_SWAP_QUANTILE_CUTOFF{ 0.5 };

        // ── MR reallocation operator ──────────────────────────────────────────────
        int MAX_MR_TASKS_PER_OUTER{ 20 };
        int MAX_MR_CANDIDATES_PER_J{ 20 };

        // ── Virtual gap insertion ─────────────────────────────────────────────────
        int VGI_MAX_GAPS_PER_TASK{ 1000000 };

        // ── Ruin & Recreate ───────────────────────────────────────────────────────
        bool   RR2_ENABLE{ true };
        int    RR2_NUM_TRIALS{ 3 };
        int    RR2_MAX_MR_TASKS{ 10 };
        int    RR2_MAX_MR_SWAPS_PER_J{ 10 };
        int    RR2_MAX_MR_COMBOS{ 15 };
        int    RR2_MAX_SR_STRIP{ 15 };
        int    RR2_NUM_PERTURBATIONS{ 10 };
        int    RR2_EXHAUST_LIMIT{ 1000 };
        int    RR2_EXHAUST_TOP_K{ 10 };
        bool   RR2_ALSO_PURE_SR{ true };
        bool   RR2_ALSO_VIRT{ true };
        int    RR2_POLISH_N_INNER{ 15 };
        double RR2_CACHE_DOMINATED_MAX_NE{ 2.0 };
        double RR2_EVAL_MAX_DEGRADATION{ 100.0 };
        double RR2_POLISH_MAX_DEGRADATION{ 50.0 }; ///< = RR2_EVAL_MAX_DEGRADATION * 0.5
        int    RR2_CACHE_DOMINATED_MAX_NEW{ 50 };

        // ── Polish parameters (light MR pass after assignment moves) ──────────────
        int  polishMrBatchTrials{ 8 };
        bool polishMrBatchCriticalOnly{ true };
        int  polishMrMoveBatchTrials{ 3 };
        int  polishMrMoveMaxPos{ 2 };

        // ── Relocation parameters (unused currently, reserved) ───────────────────
        int relocateMrBatchTrials{ 5 };
        int relocateMrMoveBatchTrials{ 3 };
        int relocateMrMoveMaxPos{ 2 };
    };


    // =============================================================================
    //  LOCAL SEARCH STATE
    //  Carries everything needed to evaluate and advance from a solution point.
    // =============================================================================
    struct LocalSearchState {
        MatrixDouble       z;            ///< Assignment matrix  [robot][task] ∈ {0,1}
        MatrixDouble       tau;          ///< Start times        [robot][task]
        VecDouble          theta;        ///< Sync start times   [task] (MR tasks only)
        VecDouble          t;            ///< Generic time seed  [task]
        std::vector<VecInt> ordP;        ///< Physical task order per robot
        std::vector<VecInt> ordV;        ///< Virtual task order per robot
        VecInt             lastPhysNode; ///< Last physical node visited per robot
        double             mksp{ std::numeric_limits<double>::infinity() };
        VecInt             endDepot;     ///< End-depot indices per robot
    };


    // =============================================================================
    //  LOCAL SEARCH OUTPUT
    //  Returned by run().  Contains the best solution found.
    // =============================================================================
    struct LocalSearchOutput {
        MatrixDouble        z_best;
        MatrixDouble        tau_best;
        VecDouble           theta_best;
        VecDouble           t_best;
        std::vector<VecInt> orders_phys_best;
        std::vector<VecInt> orders_virt_best;
        double              mksp_best{ std::numeric_limits<double>::infinity() };
        VecInt              endDepot_best;
        VecInt              lastPhysNode_best;
    };


    // =============================================================================
    //  LOCAL SEARCH
    //
    //  Implements a multi-pass local search with the following operators:
    //    • improveIntraOrder          — relocate / swap / 2-opt of SR tasks per robot
    //    • improveMrOrder             — reorder MR tasks via batch / single moves
    //    • improveGapWindow           — permute tasks inside idle-time windows
    //    • improveGapFill             — move tasks into large gaps
    //    • improveSrAssignment        — reassign SR tasks across robots
    //    • improveMrReallocation      — swap MR task participants
    //    • improveRuinRecreate        — ruin a subset, rebuild by enumeration/GRASP
    //    • improveJointMrOrderSr      — try MR reorders + simultaneous SR relocation
    //    • improveJointMrSrReinsertion — [NEW] strip SR tasks, try all MR perms,
    //                                    reinsert SR via GRASP on any capable robot
    // =============================================================================
    class LocalSearch {
    public:

        // ── Construction ──────────────────────────────────────────────────────────

        LocalSearch(
            const InstanceData& inst,
            const LocalSearchOptions& options,
            const ScheduleRepairer& repairer,
            const ScheduleScorer& scorer,
            OperatorStats* stats = nullptr); ///< nullptr → stats disabled

        // ── Main entry point ──────────────────────────────────────────────────────

        LocalSearchOutput run(
            const MatrixDouble& z0,
            const VecDouble& theta0,
            const VecDouble& t0);


    private:

        // =========================================================================
        //  DEPENDENCIES
        // =========================================================================

        const InstanceData& inst_;
        const LocalSearchOptions& opt_;
        const ScheduleRepairer& repairer_;
        const ScheduleScorer& scorer_;
        OperatorStats* stats_;    ///< May be nullptr — always guard before use.

        /// RNG used by GRASP-style operators (shuffling SR task insertion order).
        /// Seeded once from std::random_device at construction.
        mutable std::mt19937 rng_;


        // =========================================================================
        //  INTERNAL DATA STRUCTURES
        // =========================================================================

        /// Represents a group of MR tasks that must be executed together.
        /// Identified by their current participant set (robots assigned to all tasks).
        struct MrGroupInfo_ {
            std::vector<int> taskIds; ///< Physical MR task ids in this group (unordered).
            std::vector<int> robots;  ///< Robots currently assigned to every task in the group.
            int              k;       ///< Group size = participants.size().
        };


        // =========================================================================
        //  STATE MANAGEMENT HELPERS
        // =========================================================================

        LocalSearchState buildInitialState(
            const MatrixDouble& z0,
            const VecDouble& theta0,
            const VecDouble& t0);

        /// Repair + score a new assignment matrix.  Returns false if infeasible.
        bool evaluateAssignment(
            const MatrixDouble& zCand,
            LocalSearchState& outState,
            const VecDouble* thetaWarm,
            const VecDouble* tWarm,
            int                 nRepairIters);

        /// Repair + score with optional frozen or unfrozen orders.
        bool evaluateState(
            const LocalSearchState& seed,
            const MatrixDouble& zCand,
            const std::vector<VecInt>* ordPCand,
            const std::vector<VecInt>* ordVCand,
            bool                      freezeOrders,
            int                       nRepairIters,
            LocalSearchState& outState);

        /// Repair + score with a pinned virtual task start time.
        bool evaluateStateVirtualPinned(
            const LocalSearchState& seed,
            const MatrixDouble& zCand,
            const std::vector<VecInt>& ordP,
            const std::vector<VecInt>& ordV,
            int                       nRepairIters,
            const VecDouble& virtPinnedStart,
            const VecDouble& tWarm,
            LocalSearchState& outState);

        /// Light intra-order + MR-batch polish run after any assignment change.
        LocalSearchState polishAfterAssignment(LocalSearchState S, int nInnerPolish);


        // =========================================================================
        //  OPERATOR STATS HELPER
        //  Called at the END of each operator, after the best delta is known.
        // =========================================================================

        void recordOp(const std::string& name,
            double mkspBefore,
            double mkspAfter) const
        {
            if (!stats_) return;
            const bool   hit = mkspAfter < mkspBefore - 1e-9;
            const double delta = hit ? (mkspBefore - mkspAfter) : 0.0;
            stats_->record(name, hit, delta);
        }


        // =========================================================================
        //  PRIMITIVE OPERATORS
        //  Each records its own stats via recordOp() at the end.
        // =========================================================================

        /// Relocate, adjacent-swap, and 2-opt of SR tasks within each robot's sequence.
        bool improveIntraOrder(LocalSearchState& S, int nInnerOrder);

        /// Reorder MR tasks — delegates to batch-order and single-move sub-operators.
        bool improveMrOrder(LocalSearchState& S);

        /// Permute tasks inside the window around large idle-time gaps.
        bool improveGapWindow(LocalSearchState& S);

        /// Move tasks from other robots into the cheapest idle-time gap.
        bool improveGapFill(LocalSearchState& S);

        /// Relocate or swap SR tasks across robots (focuses on critical-path robots).
        bool improveSrAssignment(LocalSearchState& S);

        /// Swap a robot in/out of an MR task's participant set.
        bool improveMrReallocation(LocalSearchState& S);

        /// Pin virtual task start times to exploit idle gaps.
        bool improveVirtualGapInsertion(LocalSearchState& S);

        /// Ruin a subset of SR (and optionally MR) tasks, recreate by enumeration
        /// or GRASP perturbation.
        bool improveRuinRecreate(LocalSearchState& S);

        /// Try all permutations of each critical robot's MR sequence; for each perm
        /// also try relocating each SR task to another robot (warm-start hint repair).
        bool improveJointMrOrderSr(LocalSearchState& S);

        /// [NEW] For each combination of nGroupsToPerturb MR groups:
        ///   1. Strip all SR tasks from the group's robots.
        ///   2. Try every permutation of the MR task ordering.
        ///   3. Reinsert freed SR tasks on ANY capable robot via GRASP (or Greedy).
        ///   4. Accept if makespan improves.
        /// This is the only operator that can simultaneously change the MR ordering
        /// AND scatter SR tasks to different robots without needing strongPerturb.
        bool improveJointMrSrReinsertion(
            LocalSearchState& S,
            const JointMrSrReinsertionParams& P = {});

        // ── Sub-operators called by improveMrOrder ────────────────────────────────

        /// Try all length-kBatch windows of MR tasks and their permutations.
        bool improveCoupledMrBatchOrder(LocalSearchState& S, const LocalSearchOptions& P);

        /// Regret-based sequential reinsertion of individual MR tasks.
        bool improveCoupledMrOrder(LocalSearchState& S, const LocalSearchOptions& P);


        // =========================================================================
        //  improveJointMrSrReinsertion — PRIVATE HELPERS
        //
        //  These three functions are the core pipeline of the new operator:
        //    collectMrGroups_  →  buildMrSkeleton_  →  reinsertSr_
        // =========================================================================

        /// Scan S.z to find every distinct MR group and its current participants.
        /// Groups are identified by their participant set (sorted robot indices),
        /// so tasks sharing the same set are treated as one batch.
        std::vector<MrGroupInfo_> collectMrGroups_(const LocalSearchState& S) const;

        /// Build a skeleton state for the given groups under one specific permutation:
        ///   • Strips all SR tasks from participating robots → returned as freeSr.
        ///   • Encodes the requested MR ordering in ordP as a warm-start hint.
        ///   • Clears ordV for affected robots (repair will rebuild it).
        ///   • Leaves unaffected robots completely untouched.
        std::pair<LocalSearchState, std::vector<int>>
            buildMrSkeleton_(
                const LocalSearchState& S,
                const std::vector<MrGroupInfo_>& groups,
                const std::vector<std::vector<int>>& mrPerms) const;

        /// Fast rank-only cost estimate for placing task t on robot r at position pos.
        /// Used to build the GRASP Restricted Candidate List — no repair is called here.
        /// Replace this body to experiment with different cost models without touching
        /// the GRASP loop above.
        double estimateSrInsertionCost_(
            const LocalSearchState& S, int t, int r, int pos) const;

        /// GRASP (or Greedy) SR reinsertion on top of a skeleton state.
        /// Per GRASP trial:
        ///   1. Shuffle freeSr processing order.
        ///   2. For each task: score all (robot, pos) pairs; build RCL; sample.
        ///   3. Commit; advance the running state.
        ///   4. Full callRepairHinted + score.
        /// Returns the best repaired LocalSearchState across all trials.
        LocalSearchState reinsertSr_(
            LocalSearchState              skeleton,
            const std::vector<int>& freeSr,
            const SrReinsertionHeuristic& h);


        // =========================================================================
        //  MR TASK REINSERTION HELPERS
        //  Used by improveMrOrder, improveGapWindow, improveGapFill.
        // =========================================================================

        /// Reinsert jMR on all robots in its participant set except sAnchor,
        /// using regret-based position selection with follow-anchor semantics.
        bool reinsertMrTaskFollowAnchor(
            int                    jMR,
            int                    sAnchor,
            std::vector<VecInt>& ordP_try,
            const std::vector<VecInt>& ordV_ref,
            const LocalSearchState& Sseed,
            int nRepairFrozen,
            int maxPosTrials);

        /// Reinsert jMR on all robots in Pset; biases sGap toward idxGap position.
        bool reinsertMrTaskRegretAnchor(
            int                    jMR,
            const VecInt& Pset,
            int                    sGap,
            int                    idxGap1Based,
            std::vector<VecInt>& ordP_try,
            const std::vector<VecInt>& ordV_ref,
            const LocalSearchState& Sseed,
            int nRepairFrozen);

        /// Reinsert jMR on all robots in Pset using pure regret-based selection.
        bool reinsertMrTaskRegret(
            int                    jMR,
            const VecInt& Pset,
            const VecInt& origPosByRobot1Based,
            std::vector<VecInt>& ordP_out,
            const std::vector<VecInt>& ordV_ref,
            const LocalSearchState& Sseed,
            int nRepairFrozen,
            int maxPosTrials);


        // =========================================================================
        //  REPAIR WRAPPERS
        //  Thin wrappers around repairer_.repairPushforward() with different options.
        // =========================================================================

        /// Unfrozen repair — repairer sorts orders from scratch.
        RepairResult callRepairUnfrozen(
            const MatrixDouble& zCand,
            const VecDouble& thetaWarm,
            const VecDouble& tWarm,
            int                 nRepairIters) const;

        /// Frozen repair — orders are fixed exactly as given in ordP / ordV.
        RepairResult callRepairFrozen(
            const MatrixDouble& zCand,
            const VecDouble& thetaWarm,
            const VecDouble& tWarm,
            const std::vector<VecInt>& ordP,
            const std::vector<VecInt>& ordV,
            int                        nRepairIters) const;

        /// Frozen repair with one virtual task's start time pinned.
        RepairResult callRepairFrozenVirtualPinned(
            const MatrixDouble& zCand,
            const VecDouble& thetaWarm,
            const VecDouble& tWarm,
            const std::vector<VecInt>& ordP,
            const std::vector<VecInt>& ordV,
            int                        nRepairIters,
            const VecDouble& virtPinnedStart) const;

        /// [NEW — used by reinsertSr_] Unfrozen repair with ordP / ordV as a
        /// warm-start ordering hint (freezeOrders = false, but hint is respected).
        RepairResult callRepairHinted(
            const MatrixDouble& zCand,
            const VecDouble& thetaWarm,
            const VecDouble& tWarm,
            const std::vector<VecInt>& ordPHint,
            const std::vector<VecInt>& ordVHint,
            int                        nRepairIters) const;


        // =========================================================================
        //  DEEP / THOROUGH EVALUATION
        //  Used by improveRuinRecreate to polish promising candidates.
        // =========================================================================

        /// Quick polish: intraOrder + light MR-batch/order passes.
        std::pair<LocalSearchState, double> deepEval(
            const MatrixDouble& zTry,
            const LocalSearchState& SBase,
            const std::vector<VecInt>* ordPHint = nullptr,
            const std::vector<VecInt>* ordVHint = nullptr) const;

        /// Thorough polish: intraOrder + full MR-batch/order + gap passes.
        std::pair<LocalSearchState, double> thoroughEval(
            const MatrixDouble& zTry,
            const LocalSearchState& SBase,
            const std::vector<VecInt>* ordPHint = nullptr,
            const std::vector<VecInt>* ordVHint = nullptr) const;


        // =========================================================================
        //  ASSIGNMENT CONSTRUCTION HELPERS
        //  Used by improveRuinRecreate.
        // =========================================================================

        /// Enumerate every combination of (robot → task) given per-task capability lists.
        std::vector<std::vector<int>> enumerateAllAssignments(
            const std::vector<VecInt>& capLists) const;

        /// Assign SR physical tasks greedily (min-workload robot) then virtuals.
        MatrixDouble buildGreedyAssignment(
            const MatrixDouble& zStripped,
            const VecInt& srPhys,
            const VecInt& srVirt,
            const std::vector<VecInt>& capLists,
            const std::vector<VecInt>& virtCapLists) const;

        /// Assign virtual tasks to the least-loaded capable robot.
        MatrixDouble assignVirtuals(
            const MatrixDouble& zIn,
            const VecInt& srVirt,
            const std::vector<VecInt>& virtCapLists) const;

        /// Swap pairs of SR tasks to produce a perturbed assignment from greedy.
        MatrixDouble perturbAssignment(
            const MatrixDouble& zGreedy,
            const VecInt& srPhys,
            const std::vector<VecInt>& capLists) const;


        // =========================================================================
        //  ORDER / SEQUENCE HELPERS
        // =========================================================================

        /// Build all permutations of batch (excluding batch itself), capped at maxPerms.
        std::vector<VecInt> buildBatchPermutations(
            const VecInt& batch,
            int           maxPerms) const;

        /// Apply a coupled reordering of `batch → batchPerm` on all robots that
        /// carry any task in batch, keeping sAnchor as the canonical ordering reference.
        std::vector<VecInt> applyCoupledBatchOrder(
            const std::vector<VecInt>& ordP_in,
            const MatrixDouble& z,
            const VecInt& batch,
            const VecInt& batchPerm,
            int                        sAnchor) const;

        static void removeTaskFromSequence(VecInt& seq, int task);
        static bool containsTask(const VecInt& seq, int task);

        VecInt insertVirtualByPinTime(
            const VecInt& ordVold,
            const MatrixDouble& tauCurrent,
            int                  s,
            int                  jv,
            double               tPin) const;


        // =========================================================================
        //  SCHEDULE QUERY HELPERS
        // =========================================================================

        int    m() const; ///< Number of robots.
        int    n() const; ///< Number of tasks.

        bool   taskIsMR(int j)      const;
        bool   taskIsVirtual(int j) const;

        /// Returns the effective service duration of task j on robot s.
        double currentTaskDuration(int s, int j) const;

        /// Returns theta[j] (sync start time) for MR task j; 0 if absent.
        double thetaTaskStart(const LocalSearchState& S, int j) const;

        /// Score the current state and return per-robot arrival-to-depot times.
        VecDouble robotArrivalToDepot(const LocalSearchState& S) const;

        /// Earliest start times for all tasks given current completion times.
        VecDouble computeCurrentEST(const LocalSearchState& S) const;

        /// Build a t-seed vector from current tau / theta values.
        VecDouble buildTaskSeedFromState(const LocalSearchState& S) const;

        /// Return (gapStart, gapEnd) intervals in robot s's physical schedule.
        std::vector<std::pair<double, double>> buildPhysicalGaps(
            const LocalSearchState& S,
            int                     s,
            const VecInt& ordP) const;

        static double quantile(std::vector<double> v, double q);


        // =========================================================================
        //  HASHING HELPERS
        //  Used for tabu / neighbourhood deduplication.
        // =========================================================================

        std::string hashStateOrders(
            const MatrixDouble& z,
            const std::vector<VecInt>& ordP,
            const std::vector<VecInt>& ordV) const;

        std::string hashStateOrders(const LocalSearchState& S) const;

        std::string hashAssignment(const MatrixDouble& z) const;

        std::string makeBatchMoveKey(
            int           sAnchor,
            const VecInt& fromBatch,
            const VecInt& toBatch) const;

        std::string makeBatchNeighborhoodKey(
            const LocalSearchState& S,
            int                     sAnchor,
            const VecInt& batch,
            const VecInt& batchPerm) const;

        bool isRecentInverseBatchMove(
            int           sAnchor,
            const VecInt& batch,
            const VecInt& batchPerm) const;

        void rememberAcceptedBatchMove(
            int           sAnchor,
            const VecInt& batch,
            const VecInt& batchPerm);

        std::string makeRr2ComboKey(
            const MatrixDouble& zMr,
            const VecInt& disruptedRobots,
            const VecInt& srToStrip,
            const std::vector<VecInt>& combo) const;


        // =========================================================================
        //  ITERATION STATE  (reset each outer iteration via beginOuterIteration())
        // =========================================================================

        void beginOuterIteration();

        std::unordered_set<std::string> seenAcceptedStateHashes_;
        std::unordered_set<std::string> seenEvaluatedStateHashes_;
        std::unordered_set<std::string> seenBatchNeighborhoods_;
        std::unordered_set<std::string> exhaustedRr2Combos_;
        std::deque<std::string>         recentAcceptedBatchMoves_;

        /// Set true by improveCoupledMrBatchOrder when it commits a move, so that
        /// improveCoupledMrOrder (which would re-examine the same neighbourhood)
        /// can be skipped in the same outer iteration.
        bool skipCheckpointedOrderAfterBatch_{ false };

        static constexpr std::size_t kRecentBatchTabuMax = 64;

        /// Unused dummy stats sink — kept so stats_ is never a dangling nullptr
        /// when the caller passes nullptr.  All recordOp calls guard on stats_ first.
        mutable OperatorStats dummy_;
    };

} // namespace mrta