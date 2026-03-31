#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/InstanceData.h"
#include "ScheduleRepairer.h"
#include "ScheduleScorer.h"

#include <limits>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace mrta {

	struct LocalSearchOptions {
		int nOuter{ 10 };
		int nInnerOrder{ 10 };
		double minRelImprovementPct{ 0.1 };

		int nRepairInit{ 3 };
		int nRepairFrozen{ 3 };
		int nRepairReloc{ 10 };
		int nRepairMR{ 3 };

		int POLISH_N_INNER{ 15 };

		double GAP_BEFORE_MIN_LEN{ 5.0 };
		int GAP_WINDOW_BACK{ 3 };
		int GAP_WINDOW_FORWARD{ 2 };
		int GAP_WINDOW_MAX_GAPS{ 5 };
		int GAP_MOVE_MAX_TASKS{ 5 };

		int MR_MOVE_MAX_POS_TRIALS{ 3 };
		int MR_BATCH_N_TRIALS{ 8 };
		bool MR_BATCH_CRITICAL_ONLY{ true };
		int MR_MOVE_NUM_BATCH_TRIALS{ 5 };
		int MR_MOVE_BATCH_SIZE{ 3 };
		int MR_MOVE_MAX_TASKS{ 10 };
		int MR_BATCH_K{ 3 };
		int MR_BATCH_MAX_PERMS{ 12 };
		bool MR_BATCH_ENABLE{ true };

		double SR_SWAP_QUANTILE_CUTOFF{ 0.5 };

		int MAX_MR_TASKS_PER_OUTER{ 20 };
		int MAX_MR_CANDIDATES_PER_J{ 20 };

		int VGI_MAX_GAPS_PER_TASK{ 1000000 };

		bool RR2_ENABLE{ true };
		int RR2_NUM_TRIALS{ 3 };
		int RR2_MAX_MR_TASKS{ 4 };
		int RR2_MAX_MR_SWAPS_PER_J{ 4 };
		int RR2_MAX_MR_COMBOS{ 10 };
		int RR2_MAX_SR_STRIP{ 10 };
		int RR2_NUM_PERTURBATIONS{ 3 };
		int RR2_EXHAUST_LIMIT{ 1000 };
		int RR2_EXHAUST_TOP_K{ 10 };
		bool RR2_ALSO_PURE_SR{ true };
		bool RR2_ALSO_VIRT{ true };
		int RR2_POLISH_N_INNER{ 15 };
		double RR2_CACHE_DOMINATED_MAX_NEW = 2;
		double RR2_EVAL_MAX_DEGRADATION = 100.0;
		double RR2_POLISH_MAX_DEGRADATION = RR2_EVAL_MAX_DEGRADATION * 0.5;

		double timeLimitSeconds{ std::numeric_limits<double>::infinity() };
		// Preset overrides for inline option structs
		int  polishMrBatchTrials{ 8 };
		bool polishMrBatchCriticalOnly{ true };
		int  polishMrMoveBatchTrials{ 3 };
		int  polishMrMoveMaxPos{ 2 };

		int  relocateMrBatchTrials{ 5 };
		int  relocateMrMoveBatchTrials{ 3 };
		int  relocateMrMoveMaxPos{ 2 };
	};

	struct LocalSearchState {
		MatrixDouble z;
		MatrixDouble tau;
		VecDouble theta;
		VecDouble t;
		std::vector<VecInt> ordP;
		std::vector<VecInt> ordV;
		VecInt lastPhysNode;
		double mksp{ std::numeric_limits<double>::infinity() };
		VecInt endDepot;
	};

	struct LocalSearchOutput {
		MatrixDouble z_best;
		MatrixDouble tau_best;
		VecDouble theta_best;
		VecDouble t_best;
		std::vector<VecInt> orders_phys_best;
		std::vector<VecInt> orders_virt_best;
		double mksp_best{ std::numeric_limits<double>::infinity() };
		VecInt endDepot_best;
		VecInt lastPhysNode_best;
	};

	class LocalSearch {
	public:
		LocalSearch(
			const InstanceData& inst,
			const LocalSearchOptions& options,
			const ScheduleRepairer& repairer,
			const ScheduleScorer& scorer);

		LocalSearchOutput run(
			const MatrixDouble& z0,
			const VecDouble& theta0,
			const VecDouble& t0);

	private:
		const InstanceData& inst_;
		const LocalSearchOptions& opt_;
		const ScheduleRepairer& repairer_;
		const ScheduleScorer& scorer_;

	private:
		LocalSearchState buildInitialState(
			const MatrixDouble& z0,
			const VecDouble& theta0,
			const VecDouble& t0);

		bool evaluateAssignment(
			const MatrixDouble& zCand,
			LocalSearchState& outState,
			const VecDouble* thetaWarm,
			const VecDouble* tWarm,
			int nRepairIters);

		bool evaluateState(
			const LocalSearchState& seed,
			const MatrixDouble& zCand,
			const std::vector<VecInt>* ordPCand,
			const std::vector<VecInt>* ordVCand,
			bool freezeOrders,
			int nRepairIters,
			LocalSearchState& outState);

		bool evaluateStateVirtualPinned(
			const LocalSearchState& seed,
			const MatrixDouble& zCand,
			const std::vector<VecInt>& ordP,
			const std::vector<VecInt>& ordV,
			int nRepairIters,
			const VecDouble& virtPinnedStart,
			const VecDouble& tWarm,
			LocalSearchState& outState);

		LocalSearchState polishAfterAssignment(LocalSearchState S, int nInnerPolish);

	private:
		bool improveIntraOrder(LocalSearchState& S, int nInnerOrder);
		bool improveMrOrder(LocalSearchState& S);
		bool improveGapWindow(LocalSearchState& S);
		bool improveGapFill(LocalSearchState& S);
		bool improveSrAssignment(LocalSearchState& S);
		bool improveMrReallocation(LocalSearchState& S);
		bool improveVirtualGapInsertion(LocalSearchState& S);
		bool improveRuinRecreate(LocalSearchState& S);
		bool improveCoupledMrBatchOrder(LocalSearchState& S, const LocalSearchOptions& P);
		bool improveCoupledMrOrder(LocalSearchState& S, const LocalSearchOptions& P);

		bool reinsertMrTaskFollowAnchor(
			int jMR,
			int sAnchor,
			std::vector<VecInt>& ordP_try,
			const std::vector<VecInt>& ordV_ref,
			const LocalSearchState& Sseed,
			int nRepairFrozen,
			int maxPosTrials);

		bool reinsertMrTaskRegretAnchor(
			int jMR,
			const VecInt& Pset,
			int sGap,
			int idxGap1Based,
			std::vector<VecInt>& ordP_try,
			const std::vector<VecInt>& ordV_ref,
			const LocalSearchState& Sseed,
			int nRepairFrozen);

		bool reinsertMrTaskRegret(
			int jMR,
			const VecInt& Pset,
			const VecInt& origPosByRobot1Based,
			std::vector<VecInt>& ordP_out,
			const std::vector<VecInt>& ordV_ref,
			const LocalSearchState& Sseed,
			int nRepairFrozen,
			int maxPosTrials);

		RepairResult callRepairUnfrozen(
			const MatrixDouble& zCand,
			const VecDouble& thetaWarm,
			const VecDouble& tWarm,
			int nRepairIters) const;

		RepairResult callRepairFrozen(
			const MatrixDouble& zCand,
			const VecDouble& thetaWarm,
			const VecDouble& tWarm,
			const std::vector<VecInt>& ordP,
			const std::vector<VecInt>& ordV,
			int nRepairIters) const;

		RepairResult callRepairFrozenVirtualPinned(
			const MatrixDouble& zCand,
			const VecDouble& thetaWarm,
			const VecDouble& tWarm,
			const std::vector<VecInt>& ordP,
			const std::vector<VecInt>& ordV,
			int nRepairIters,
			const VecDouble& virtPinnedStart) const;

		int m() const;
		int n() const;

		bool taskIsMR(int j) const;
		bool taskIsVirtual(int j) const;

		double currentTaskDuration(int s, int j) const;
		double thetaTaskStart(const LocalSearchState& S, int j) const;
		VecDouble robotArrivalToDepot(const LocalSearchState& S) const;
		VecDouble computeCurrentEST(const LocalSearchState& S) const;
		VecDouble buildTaskSeedFromState(const LocalSearchState& S) const;

		std::vector<std::pair<double, double>> buildPhysicalGaps(
			const LocalSearchState& S,
			int s,
			const VecInt& ordP) const;

		VecInt insertVirtualByPinTime(
			const VecInt& ordVold,
			const MatrixDouble& tauCurrent,
			int s,
			int jv,
			double tPin) const;

		MatrixDouble buildGreedyAssignment(
			const MatrixDouble& zStripped,
			const VecInt& srPhys,
			const VecInt& srVirt,
			const std::vector<VecInt>& capLists,
			const std::vector<VecInt>& virtCapLists) const;

		MatrixDouble assignVirtuals(
			const MatrixDouble& zIn,
			const VecInt& srVirt,
			const std::vector<VecInt>& virtCapLists) const;

		MatrixDouble perturbAssignment(
			const MatrixDouble& zGreedy,
			const VecInt& srPhys,
			const std::vector<VecInt>& capLists) const;

		std::pair<LocalSearchState, double> deepEval(
			const MatrixDouble& zTry,
			const LocalSearchState& SBase) const;

		std::pair<LocalSearchState, double> thoroughEval(
			const MatrixDouble& zTry,
			const LocalSearchState& SBase) const;

		std::vector<std::vector<int>> enumerateAllAssignments(
			const std::vector<VecInt>& capLists) const;

		std::vector<VecInt> buildBatchPermutations(
			const VecInt& batch,
			int maxPerms) const;

		std::vector<VecInt> applyCoupledBatchOrder(
			const std::vector<VecInt>& ordP_in,
			const MatrixDouble& z,
			const VecInt& batch,
			const VecInt& batchPerm,
			int sAnchor) const;

		std::string hashAssignment(const MatrixDouble& z) const;

		static void removeTaskFromSequence(VecInt& seq, int task);
		static bool containsTask(const VecInt& seq, int task);
		static double quantile(std::vector<double> v, double q);

	private:
		void beginOuterIteration();

		std::string hashStateOrders(
			const MatrixDouble& z,
			const std::vector<VecInt>& ordP,
			const std::vector<VecInt>& ordV) const;

		std::string hashStateOrders(const LocalSearchState& S) const;

		std::string makeBatchMoveKey(
			int sAnchor,
			const VecInt& fromBatch,
			const VecInt& toBatch) const;

		std::string makeBatchNeighborhoodKey(
			const LocalSearchState& S,
			int sAnchor,
			const VecInt& batch,
			const VecInt& batchPerm) const;

		bool isRecentInverseBatchMove(
			int sAnchor,
			const VecInt& batch,
			const VecInt& batchPerm) const;

		void rememberAcceptedBatchMove(
			int sAnchor,
			const VecInt& batch,
			const VecInt& batchPerm);

		std::string makeRr2ComboKey(
			const MatrixDouble& zMr,
			const VecInt& disruptedRobots,
			const VecInt& srToStrip,
			const std::vector<VecInt>& combo) const;


		std::unordered_set<std::string> seenAcceptedStateHashes_;
		std::unordered_set<std::string> seenEvaluatedStateHashes_;
		std::unordered_set<std::string> seenBatchNeighborhoods_;
		std::unordered_set<std::string> exhaustedRr2Combos_;
		std::deque<std::string> recentAcceptedBatchMoves_;
		bool skipCheckpointedOrderAfterBatch_{ false };
		static constexpr std::size_t kRecentBatchTabuMax = 64;
	};

} // namespace mrta