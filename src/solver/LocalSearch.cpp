#include "LocalSearch.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <unordered_set>

namespace mrta {

	LocalSearch::LocalSearch(
		const InstanceData& inst,
		const LocalSearchOptions& options,
		const ScheduleRepairer& repairer,
		const ScheduleScorer& scorer)
		: inst_(inst), opt_(options), repairer_(repairer), scorer_(scorer) {
	}

	void LocalSearch::beginOuterIteration()
	{
		seenAcceptedStateHashes_.clear();
		seenEvaluatedStateHashes_.clear();
		seenBatchNeighborhoods_.clear();
		exhaustedRr2Combos_.clear();
		skipCheckpointedOrderAfterBatch_ = false;
	}

	std::string LocalSearch::hashStateOrders(
		const MatrixDouble& z,
		const std::vector<VecInt>& ordP,
		const std::vector<VecInt>& ordV) const
	{
		std::ostringstream oss;
		oss.precision(17);

		oss << "Z|";
		for (int r = 0; r < static_cast<int>(z.size()); ++r) {
			oss << r << ":";
			for (int j = 0; j < static_cast<int>(z[r].size()); ++j) {
				if (z[r][j] > 0.5) {
					oss << j << ",";
				}
			}
			oss << "|";
		}

		oss << "P|";
		for (int r = 0; r < static_cast<int>(ordP.size()); ++r) {
			oss << r << ":";
			for (int j : ordP[r]) {
				oss << j << ",";
			}
			oss << "|";
		}

		oss << "V|";
		for (int r = 0; r < static_cast<int>(ordV.size()); ++r) {
			oss << r << ":";
			for (int j : ordV[r]) {
				oss << j << ",";
			}
			oss << "|";
		}

		return oss.str();
	}

	std::string LocalSearch::hashStateOrders(const LocalSearchState& S) const
	{
		return hashStateOrders(S.z, S.ordP, S.ordV);
	}

	std::string LocalSearch::makeBatchMoveKey(
		int sAnchor,
		const VecInt& fromBatch,
		const VecInt& toBatch) const
	{
		std::ostringstream oss;
		oss << "A" << sAnchor << "|F:";
		for (int x : fromBatch) oss << x << ",";
		oss << "|T:";
		for (int x : toBatch) oss << x << ",";
		return oss.str();
	}

	std::string LocalSearch::makeBatchNeighborhoodKey(
		const LocalSearchState& S,
		int sAnchor,
		const VecInt& batch,
		const VecInt& batchPerm) const
	{
		std::ostringstream oss;

		// Anchor-local neighborhood fingerprint, not full global state.
		oss << "ANCHOR=" << sAnchor << "|SEQ:";
		for (int x : S.ordP[sAnchor]) {
			oss << x << ",";
		}

		oss << "|B:";
		for (int x : batch) {
			oss << x << ",";
		}

		oss << "|P:";
		for (int x : batchPerm) {
			oss << x << ",";
		}

		return oss.str();
	}

	bool LocalSearch::isRecentInverseBatchMove(
		int sAnchor,
		const VecInt& batch,
		const VecInt& batchPerm) const
	{
		const std::string inverseKey = makeBatchMoveKey(sAnchor, batchPerm, batch);
		return std::find(
			recentAcceptedBatchMoves_.begin(),
			recentAcceptedBatchMoves_.end(),
			inverseKey) != recentAcceptedBatchMoves_.end();
	}

	void LocalSearch::rememberAcceptedBatchMove(
		int sAnchor,
		const VecInt& batch,
		const VecInt& batchPerm)
	{
		recentAcceptedBatchMoves_.push_back(
			makeBatchMoveKey(sAnchor, batch, batchPerm));

		while (recentAcceptedBatchMoves_.size() > kRecentBatchTabuMax) {
			recentAcceptedBatchMoves_.pop_front();
		}
	}

	std::string LocalSearch::makeRr2ComboKey(
		const MatrixDouble&,
		const VecInt& disruptedRobots,
		const VecInt& srToStrip,
		const std::vector<VecInt>& combo) const
	{
		std::ostringstream oss;

		VecInt dr = disruptedRobots;
		VecInt sr = srToStrip;
		std::sort(dr.begin(), dr.end());
		std::sort(sr.begin(), sr.end());

		oss << "DR:";
		for (int r : dr) {
			oss << r << ",";
		}

		oss << "|SR:";
		for (int j : sr) {
			oss << j << ",";
		}

		// Canonicalize combo entries as strings, then sort them.
		std::vector<std::string> comboParts;
		comboParts.reserve(combo.size());

		for (const auto& sw : combo) {
			std::ostringstream part;
			for (int x : sw) {
				part << x << ",";
			}
			comboParts.push_back(part.str());
		}

		std::sort(comboParts.begin(), comboParts.end());

		oss << "|MR:";
		for (const auto& part : comboParts) {
			oss << "[" << part << "]";
		}

		return oss.str();
	}

	LocalSearchOutput LocalSearch::run(
		const MatrixDouble& z0,
		const VecDouble& theta0,
		const VecDouble& t0)
	{
		const auto startTime = std::chrono::steady_clock::now();
		bool did = false;
		auto elapsedSeconds = [&]() -> double {
			return std::chrono::duration<double>(
				std::chrono::steady_clock::now() - startTime).count();
			};

		auto timeExpired = [&]() -> bool {
			return elapsedSeconds() >= opt_.timeLimitSeconds;
			};

		LocalSearchState S = buildInitialState(z0, theta0, t0);

		std::cout << "LS init: mksp = " << S.mksp << "\n";

		double mkspPrev = S.mksp;

		for (int outer = 0; outer < opt_.nOuter; ++outer) {

			beginOuterIteration();
			const std::string startKey = hashStateOrders(S);
			seenAcceptedStateHashes_.insert(startKey);
			seenEvaluatedStateHashes_.insert(startKey);

			if (timeExpired()) {
				std::cout << "LS time limit reached at outer " << outer
					<< " (" << elapsedSeconds() << "s)\n";
				break;
			}

			bool improved = false;

			improved = improveIntraOrder(S, opt_.nInnerOrder) || improved;
			improved = improveMrOrder(S) || improved;
			improved = improveGapWindow(S) || improved;
			improved = improveGapFill(S) || improved;

			did = improveSrAssignment(S);
			if (did) {
				improved = true;
				S = polishAfterAssignment(S, opt_.POLISH_N_INNER);
			}

			did = improveMrReallocation(S);
			if (did) {
				improved = true;
				S = polishAfterAssignment(S, opt_.POLISH_N_INNER);
			}

			did = improveRuinRecreate(S);
			if (did) {
				improved = true;
				S = polishAfterAssignment(S, opt_.POLISH_N_INNER);
			}


			improveVirtualGapInsertion(S);

			std::cout << "LS outer " << (outer + 1)
				<< ": mksp_best = " << S.mksp
				<< (improved ? " (improved)" : "") << "\n";

			if (!improved) {
				break;
			}

			if (outer >= 1 && mkspPrev > 0.0) {
				const double deltaPct = 100.0 * (mkspPrev - S.mksp) / mkspPrev;
				if (deltaPct >= 0.0 && deltaPct < opt_.minRelImprovementPct) {
					std::cout << "LS stall: " << deltaPct
						<< "% improvement — stopping early.\n";
					break;
				}
			}

			mkspPrev = S.mksp;
		}

		if (!timeExpired()) {
			improveVirtualGapInsertion(S);
		}
		std::cout << "LS finished: real mksp = " << S.mksp
			<< " elapsed = " << elapsedSeconds() << "s\n";

		LocalSearchOutput out;
		out.z_best = S.z;
		out.tau_best = S.tau;
		out.theta_best = S.theta;
		out.t_best = S.t;
		out.orders_phys_best = S.ordP;
		out.orders_virt_best = S.ordV;
		out.mksp_best = S.mksp;
		out.endDepot_best = S.endDepot;
		out.lastPhysNode_best = S.lastPhysNode;
		return out;
	}



	LocalSearchState LocalSearch::buildInitialState(
		const MatrixDouble& z0,
		const VecDouble& theta0,
		const VecDouble& t0)
	{
		LocalSearchState S;
		const bool ok = evaluateAssignment(z0, S, &theta0, &t0, opt_.nRepairInit);
		if (!ok) {
			throw std::runtime_error("LocalSearch: failed to build initial repaired state.");
		}
		return S;
	}

	bool LocalSearch::evaluateAssignment(
		const MatrixDouble& zCand,
		LocalSearchState& outState,
		const VecDouble* thetaWarm,
		const VecDouble* tWarm,
		int nRepairIters)
	{
		const RepairResult rep = callRepairUnfrozen(
			zCand,
			thetaWarm ? *thetaWarm : VecDouble(n(), 0.0),
			tWarm ? *tWarm : VecDouble(n(), 0.0),
			nRepairIters);

		const ScheduleScore scr = scorer_.scoreScheduleExact(
			inst_,
			zCand,
			rep.tauFeas,
			rep.ordersPhys,
			rep.ordersVirt,
			rep.lastPhysNode);

		if (!std::isfinite(scr.mksp)) {
			return false;
		}

		outState.z = zCand;
		outState.tau = rep.tauFeas;
		outState.theta = rep.thetaFeas;
		outState.t = rep.tFeas;
		outState.ordP = rep.ordersPhys;
		outState.ordV = rep.ordersVirt;
		outState.lastPhysNode = rep.lastPhysNode;
		outState.mksp = scr.mksp;
		outState.endDepot = scr.endDepot;
		return true;
	}

	bool LocalSearch::evaluateState(
		const LocalSearchState& seed,
		const MatrixDouble& zCand,
		const std::vector<VecInt>* ordPCand,
		const std::vector<VecInt>* ordVCand,
		bool freezeOrders,
		int nRepairIters,
		LocalSearchState& outState)
	{
		RepairResult rep;

		if (freezeOrders && ordPCand && ordVCand) {
			rep = callRepairFrozen(
				zCand,
				seed.theta,
				seed.t,
				*ordPCand,
				*ordVCand,
				nRepairIters);
		}
		else {
			rep = callRepairUnfrozen(
				zCand,
				seed.theta,
				seed.t,
				nRepairIters);
		}

		const ScheduleScore scr = scorer_.scoreScheduleExact(
			inst_,
			zCand,
			rep.tauFeas,
			rep.ordersPhys,
			rep.ordersVirt,
			rep.lastPhysNode);

		if (!std::isfinite(scr.mksp)) {
			return false;
		}

		outState.z = zCand;
		outState.tau = rep.tauFeas;
		outState.theta = rep.thetaFeas;
		outState.t = rep.tFeas;
		outState.ordP = rep.ordersPhys;
		outState.ordV = rep.ordersVirt;
		outState.lastPhysNode = rep.lastPhysNode;
		outState.mksp = scr.mksp;
		outState.endDepot = scr.endDepot;
		return true;
	}

	bool LocalSearch::evaluateStateVirtualPinned(
		const LocalSearchState& seed,
		const MatrixDouble& zCand,
		const std::vector<VecInt>& ordP,
		const std::vector<VecInt>& ordV,
		int nRepairIters,
		const VecDouble& virtPinnedStart,
		const VecDouble& tWarm,
		LocalSearchState& outState)
	{
		const RepairResult rep = callRepairFrozenVirtualPinned(
			zCand,
			seed.theta,
			tWarm,
			ordP,
			ordV,
			nRepairIters,
			virtPinnedStart);

		const ScheduleScore scr = scorer_.scoreScheduleExact(
			inst_,
			zCand,
			rep.tauFeas,
			rep.ordersPhys,
			rep.ordersVirt,
			rep.lastPhysNode);

		if (!std::isfinite(scr.mksp)) {
			return false;
		}

		outState.z = zCand;
		outState.tau = rep.tauFeas;
		outState.theta = rep.thetaFeas;
		outState.t = rep.tFeas;
		outState.ordP = rep.ordersPhys;
		outState.ordV = rep.ordersVirt;
		outState.lastPhysNode = rep.lastPhysNode;
		outState.mksp = scr.mksp;
		outState.endDepot = scr.endDepot;
		return true;
	}

	LocalSearchState LocalSearch::polishAfterAssignment(LocalSearchState S, int nInnerPolish)
	{
		const double mkspBefore = S.mksp;

		improveIntraOrder(S, nInnerPolish);

		LocalSearchOptions P_light = opt_;
		P_light.MR_BATCH_N_TRIALS = std::min(8, opt_.MR_BATCH_N_TRIALS);
		P_light.MR_BATCH_CRITICAL_ONLY = true;
		P_light.MR_MOVE_NUM_BATCH_TRIALS = std::min(3, opt_.MR_MOVE_NUM_BATCH_TRIALS);
		P_light.MR_MOVE_MAX_POS_TRIALS = std::min(2, opt_.MR_MOVE_MAX_POS_TRIALS);

		improveCoupledMrBatchOrder(S, P_light);
		improveCoupledMrOrder(S, P_light);
		improveGapWindow(S);

		const double delta = mkspBefore - S.mksp;
		if (delta > 1e-9) {
			std::cout << "    Polish: " << delta
				<< " improvement (" << mkspBefore
				<< " -> " << S.mksp << ")\n";
		}

		return S;
	}

	bool LocalSearch::improveIntraOrder(LocalSearchState& S, int nInnerOrder)
	{
		bool improved = false;

		for (int s = 0; s < m(); ++s) {
			VecInt seqBase = S.ordP[s];
			if (static_cast<int>(seqBase.size()) <= 2) {
				continue;
			}

			VecInt movableIdx;
			for (int p = 0; p < static_cast<int>(seqBase.size()); ++p) {
				const int j = seqBase[p];
				if (!taskIsMR(j) && !taskIsVirtual(j)) {
					movableIdx.push_back(p);
				}
			}

			if (static_cast<int>(movableIdx.size()) <= 1) {
				continue;
			}

			for (int iter = 0; iter < nInnerOrder; ++iter) {
				bool localImproved = false;

				{
					LocalSearchState bestLocal = S;
					VecInt bestSeq = seqBase;

					for (int from = 0; from < static_cast<int>(seqBase.size()); ++from) {
						const int x = seqBase[from];
						if (taskIsMR(x) || taskIsVirtual(x)) {
							continue;
						}

						VecInt seqTry = seqBase;
						seqTry.erase(seqTry.begin() + from);

						for (int b = 0; b <= static_cast<int>(seqTry.size()); ++b) {
							VecInt seqIns = seqTry;
							seqIns.insert(seqIns.begin() + b, x);

							if (seqIns == seqBase) {
								continue;
							}

							auto ordP_try = S.ordP;
							ordP_try[s] = seqIns;

							LocalSearchState St;
							if (!evaluateState(S, S.z, &ordP_try, &S.ordV, true, opt_.nRepairFrozen, St)) {
								continue;
							}

							if (St.mksp + 1e-9 < bestLocal.mksp) {
								bestLocal = std::move(St);
								bestSeq = seqIns;
							}
						}
					}

					if (bestLocal.mksp + 1e-9 < S.mksp) {
						S = std::move(bestLocal);
						seqBase = std::move(bestSeq);
						improved = true;
						localImproved = true;
					}
				}

				{
					LocalSearchState bestLocal = S;
					VecInt bestSeq = seqBase;

					for (int q = 0; q + 1 < static_cast<int>(seqBase.size()); ++q) {
						const int j1 = seqBase[q];
						const int j2 = seqBase[q + 1];

						if (taskIsMR(j1) || taskIsMR(j2) || taskIsVirtual(j1) || taskIsVirtual(j2)) {
							continue;
						}

						VecInt seqTry = seqBase;
						std::swap(seqTry[q], seqTry[q + 1]);

						auto ordP_try = S.ordP;
						ordP_try[s] = seqTry;

						LocalSearchState St;
						if (!evaluateState(S, S.z, &ordP_try, &S.ordV, true, opt_.nRepairFrozen, St)) {
							continue;
						}

						if (St.mksp + 1e-9 < bestLocal.mksp) {
							bestLocal = std::move(St);
							bestSeq = seqTry;
						}
					}

					if (bestLocal.mksp + 1e-9 < S.mksp) {
						S = std::move(bestLocal);
						seqBase = std::move(bestSeq);
						improved = true;
						localImproved = true;
					}
				}

				{
					LocalSearchState bestLocal = S;
					VecInt bestSeq = seqBase;

					if (static_cast<int>(seqBase.size()) >= 4) {
						for (int i = 0; i + 1 < static_cast<int>(seqBase.size()) - 1; ++i) {
							for (int k = i + 1; k < static_cast<int>(seqBase.size()) - 1; ++k) {
								bool ok = true;
								for (int t = i; t <= k; ++t) {
									const int j = seqBase[t];
									if (taskIsMR(j) || taskIsVirtual(j)) {
										ok = false;
										break;
									}
								}
								if (!ok) {
									continue;
								}

								VecInt seqTry = seqBase;
								std::reverse(seqTry.begin() + i, seqTry.begin() + k + 1);

								auto ordP_try = S.ordP;
								ordP_try[s] = seqTry;

								LocalSearchState St;
								if (!evaluateState(S, S.z, &ordP_try, &S.ordV, true, opt_.nRepairFrozen, St)) {
									continue;
								}

								if (St.mksp + 1e-9 < bestLocal.mksp) {
									bestLocal = std::move(St);
									bestSeq = seqTry;
								}
							}
						}
					}

					if (bestLocal.mksp + 1e-9 < S.mksp) {
						S = std::move(bestLocal);
						seqBase = std::move(bestSeq);
						improved = true;
						localImproved = true;
					}
				}

				if (!localImproved) {
					break;
				}
			}
		}

		return improved;
	}

	bool LocalSearch::improveMrOrder(LocalSearchState& S)
	{
		bool improved = false;

		skipCheckpointedOrderAfterBatch_ = false;
		const bool didBatch = improveCoupledMrBatchOrder(S, opt_);
		improved = didBatch || improved;

		if (!skipCheckpointedOrderAfterBatch_) {
			improved = improveCoupledMrOrder(S, opt_) || improved;
		}

		return improved;
	}

	bool LocalSearch::improveGapWindow(LocalSearchState& S)
	{
		bool improved = false;
		double bestDelta = 0.0;
		LocalSearchState bestS = S;

		struct GapCand {
			double score;
			int sGap;
			int idx;
			int jAfter;
		};

		const VecDouble arrivalTmp = robotArrivalToDepot(S);

		std::vector<GapCand> gapList;
		for (int s = 0; s < m(); ++s) {
			const VecInt& seq = S.ordP[s];
			if (seq.empty()) {
				continue;
			}

			double prevEnd = 0.0;
			for (int idx = 0; idx < static_cast<int>(seq.size()); ++idx) {
				const int jAfter = seq[idx];
				const double stAfter = taskIsMR(jAfter)
					? thetaTaskStart(S, jAfter)
					: S.tau[s][jAfter];

				const double durAfter = currentTaskDuration(s, jAfter);
				const double gapLen = stAfter - prevEnd;

				if (gapLen >= opt_.GAP_BEFORE_MIN_LEN) {
					gapList.push_back({ gapLen, s, idx, jAfter });
				}

				prevEnd = stAfter + durAfter;
			}
		}

		if (gapList.empty()) {
			return false;
		}

		std::sort(gapList.begin(), gapList.end(),
			[](const GapCand& a, const GapCand& b) { return a.score > b.score; });

		if (static_cast<int>(gapList.size()) > opt_.GAP_WINDOW_MAX_GAPS) {
			gapList.resize(opt_.GAP_WINDOW_MAX_GAPS);
		}

		for (const auto& g : gapList) {
			const int sGap = g.sGap;
			const int jAfter = g.jAfter;

			int sAnchor = sGap;
			if (taskIsMR(jAfter)) {
				VecInt Pset;
				for (int r = 0; r < m(); ++r) {
					if (S.z[r][jAfter] > 0.5) {
						Pset.push_back(r);
					}
				}
				if (Pset.empty()) {
					continue;
				}

				int bestR = Pset.front();
				double bestArrival = arrivalTmp[bestR];
				for (int r : Pset) {
					if (arrivalTmp[r] > bestArrival) {
						bestArrival = arrivalTmp[r];
						bestR = r;
					}
				}
				sAnchor = bestR;
			}

			const VecInt& seqA = S.ordP[sAnchor];
			auto it = std::find(seqA.begin(), seqA.end(), jAfter);
			if (it == seqA.end()) {
				continue;
			}

			const int idxA = static_cast<int>(std::distance(seqA.begin(), it));
			const int a = std::max(0, idxA - opt_.GAP_WINDOW_BACK);
			const int b = std::min(static_cast<int>(seqA.size()) - 1, idxA + opt_.GAP_WINDOW_FORWARD);

			VecInt win(seqA.begin() + a, seqA.begin() + b + 1);
			if (static_cast<int>(win.size()) <= 1 || static_cast<int>(win.size()) > 5) {
				continue;
			}

			VecInt mrWin;
			for (int j : win) {
				if (taskIsMR(j) && !taskIsVirtual(j)) {
					mrWin.push_back(j);
				}
			}

			auto ordP_base = S.ordP;
			VecInt seqAnchorBase = seqA;
			seqAnchorBase.erase(seqAnchorBase.begin() + a, seqAnchorBase.begin() + b + 1);
			ordP_base[sAnchor] = seqAnchorBase;

			bool feasibleBase = true;
			for (int jMR : mrWin) {
				for (int s = 0; s < m(); ++s) {
					if (S.z[s][jMR] > 0.5 && s != sAnchor) {
						auto& seq = ordP_base[s];
						auto itj = std::find(seq.begin(), seq.end(), jMR);
						if (itj == seq.end()) {
							feasibleBase = false;
							break;
						}
						seq.erase(itj);
					}
				}
				if (!feasibleBase) {
					break;
				}
			}
			if (!feasibleBase) {
				continue;
			}

			VecInt permWin = win;
			std::sort(permWin.begin(), permWin.end());

			do {
				if (permWin == win) {
					continue;
				}

				auto ordP_try = ordP_base;
				VecInt rebuilt;
				rebuilt.insert(rebuilt.end(), seqAnchorBase.begin(), seqAnchorBase.begin() + a);
				rebuilt.insert(rebuilt.end(), permWin.begin(), permWin.end());
				rebuilt.insert(rebuilt.end(), seqAnchorBase.begin() + a, seqAnchorBase.end());
				ordP_try[sAnchor] = rebuilt;

				bool feasible = true;
				VecInt mrPerm;
				for (int j : permWin) {
					if (taskIsMR(j) && !taskIsVirtual(j)) {
						mrPerm.push_back(j);
					}
				}

				for (int jMR : mrPerm) {
					if (!reinsertMrTaskFollowAnchor(
						jMR, sAnchor, ordP_try, S.ordV, S,
						opt_.nRepairFrozen, opt_.MR_MOVE_MAX_POS_TRIALS)) {
						feasible = false;
						break;
					}
				}

				if (!feasible) {
					continue;
				}

				LocalSearchState St;
				if (!evaluateState(S, S.z, &ordP_try, &S.ordV, true, opt_.nRepairFrozen, St)) {
					continue;
				}

				const double delta = S.mksp - St.mksp;
				if (delta > bestDelta + 1e-9) {
					bestDelta = delta;
					bestS = std::move(St);
				}

			} while (std::next_permutation(permWin.begin(), permWin.end()));
		}

		if (bestDelta > 1e-9) {
			S = std::move(bestS);
			improved = true;
			std::cout << "  Gap-window improved: mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveGapFill(LocalSearchState& S)
	{
		bool improved = false;
		double bestDelta = 0.0;
		LocalSearchState bestS = S;

		struct GapCand {
			int sGap;
			int idxGap;
			double score;
		};

		std::vector<GapCand> gapList;

		for (int sGap = 0; sGap < m(); ++sGap) {
			const VecInt& seq = S.ordP[sGap];
			if (seq.empty()) {
				continue;
			}

			double prevEnd = 0.0;
			for (int idxGap = 0; idxGap < static_cast<int>(seq.size()); ++idxGap) {
				const int jAfter = seq[idxGap];
				const double stAfter = taskIsMR(jAfter)
					? thetaTaskStart(S, jAfter)
					: S.tau[sGap][jAfter];

				const double gapLen = stAfter - prevEnd;

				if (gapLen >= opt_.GAP_BEFORE_MIN_LEN) {
					gapList.push_back({ sGap, idxGap, gapLen });
				}

				prevEnd = stAfter + currentTaskDuration(sGap, jAfter);
			}
		}

		if (gapList.empty()) {
			return false;
		}

		std::sort(gapList.begin(), gapList.end(),
			[](const GapCand& a, const GapCand& b) { return a.score > b.score; });

		if (static_cast<int>(gapList.size()) > opt_.GAP_MOVE_MAX_TASKS) {
			gapList.resize(opt_.GAP_MOVE_MAX_TASKS);
		}

		for (const auto& g : gapList) {
			const int sGap = g.sGap;
			const int idxGap = g.idxGap;

			for (int j = 0; j < n(); ++j) {
				if (taskIsMR(j) || taskIsVirtual(j)) {
					continue;
				}

				int src = -1;
				for (int r = 0; r < m(); ++r) {
					if (S.z[r][j] > 0.5) {
						src = r;
						break;
					}
				}

				if (src < 0 || inst_.cap[sGap][j] < 0.5) {
					continue;
				}

				MatrixDouble z_try = S.z;
				auto ordP_try = S.ordP;
				auto ordV_try = S.ordV;

				z_try[src][j] = 0.0;
				z_try[sGap][j] = 1.0;

				removeTaskFromSequence(ordP_try[src], j);
				removeTaskFromSequence(ordV_try[src], j);
				removeTaskFromSequence(ordP_try[sGap], j);
				removeTaskFromSequence(ordV_try[sGap], j);

				VecInt seqBase = ordP_try[sGap];
				const int idxUse = std::min(idxGap, static_cast<int>(seqBase.size()));
				seqBase.insert(seqBase.begin() + idxUse, j);
				ordP_try[sGap] = seqBase;

				LocalSearchState St;
				if (!evaluateState(S, z_try, &ordP_try, &ordV_try, true, opt_.nRepairFrozen, St)) {
					continue;
				}

				const double delta = S.mksp - St.mksp;
				if (delta > bestDelta + 1e-9) {
					bestDelta = delta;
					bestS = std::move(St);
				}
			}

			for (int j = 0; j < n(); ++j) {
				if (!taskIsMR(j) || taskIsVirtual(j)) {
					continue;
				}

				VecInt Pset;
				for (int r = 0; r < m(); ++r) {
					if (S.z[r][j] > 0.5) {
						Pset.push_back(r);
					}
				}

				if (Pset.empty()) {
					continue;
				}

				if (std::find(Pset.begin(), Pset.end(), sGap) == Pset.end()) {
					continue;
				}

				auto ordP_try = S.ordP;
				bool feasible = true;

				for (int r : Pset) {
					if (!containsTask(ordP_try[r], j)) {
						feasible = false;
						break;
					}
					removeTaskFromSequence(ordP_try[r], j);
				}

				if (!feasible) {
					continue;
				}

				auto ordP_work = ordP_try;
				if (!reinsertMrTaskRegretAnchor(
					j, Pset, sGap, idxGap + 1, ordP_work, S.ordV, S, opt_.nRepairFrozen)) {
					continue;
				}

				LocalSearchState St;
				if (!evaluateState(S, S.z, &ordP_work, &S.ordV, true, opt_.nRepairFrozen, St)) {
					continue;
				}

				const double delta = S.mksp - St.mksp;
				if (delta > bestDelta + 1e-9) {
					bestDelta = delta;
					bestS = std::move(St);
				}
			}
		}

		if (bestDelta > 1e-9) {
			S = std::move(bestS);
			improved = true;
			std::cout << "  Gap-fill move improved: mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveSrAssignment(LocalSearchState& S)
	{
		bool improved = false;
		double bestDelta = 0.0;
		LocalSearchState bestS = S;

		LocalSearchOptions P_light = opt_;
		P_light.MR_BATCH_N_TRIALS = 5;
		P_light.MR_BATCH_CRITICAL_ONLY = false;
		P_light.MR_MOVE_NUM_BATCH_TRIALS = 3;
		P_light.MR_MOVE_MAX_POS_TRIALS = 2;

		const VecDouble arrivalTmp = robotArrivalToDepot(S);
		const double mkspCurr = *std::max_element(arrivalTmp.begin(), arrivalTmp.end());

		VecInt critRobots;
		for (int r = 0; r < m(); ++r) {
			if (std::abs(arrivalTmp[r] - mkspCurr) <= 1e-9) {
				critRobots.push_back(r);
			}
		}

		for (int cr : critRobots) {
			for (int j = 0; j < n(); ++j) {
				if (taskIsMR(j)) {
					continue;
				}
				if (S.z[cr][j] <= 0.5) {
					continue;
				}

				for (int dst = 0; dst < m(); ++dst) {
					if (dst == cr) {
						continue;
					}
					if (inst_.cap[dst][j] < 0.5) {
						continue;
					}

					MatrixDouble z_try = S.z;
					z_try[cr][j] = 0.0;
					z_try[dst][j] = 1.0;

					LocalSearchState St;
					if (!evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairReloc, St)) {
						continue;
					}

					improveCoupledMrBatchOrder(St, P_light);
					improveCoupledMrOrder(St, P_light);

					const double delta = S.mksp - St.mksp;
					if (delta > bestDelta + 1e-9) {
						bestDelta = delta;
						bestS = std::move(St);
					}
				}
			}
		}

		VecInt allSRTasks;
		VecInt physSRTasks;

		for (int j = 0; j < n(); ++j) {
			if (!taskIsMR(j)) {
				allSRTasks.push_back(j);
			}
			if (!taskIsMR(j) && !taskIsVirtual(j)) {
				physSRTasks.push_back(j);
			}
		}

		VecInt srcTasks;
		for (int j : physSRTasks) {
			bool onCritical = false;
			for (int cr : critRobots) {
				if (S.z[cr][j] > 0.5) {
					onCritical = true;
					break;
				}
			}
			if (onCritical) {
				srcTasks.push_back(j);
			}
		}

		std::vector<double> quickDeltas;
		std::vector<LocalSearchState> quickStates;
		struct SwapPair { int j, a, k, b; };
		std::vector<SwapPair> pairs;

		for (int j : srcTasks) {
			int a = -1;
			for (int r = 0; r < m(); ++r) {
				if (S.z[r][j] > 0.5) {
					a = r;
					break;
				}
			}
			if (a < 0) {
				continue;
			}

			for (int k : allSRTasks) {
				if (k == j) {
					continue;
				}

				int b = -1;
				for (int r = 0; r < m(); ++r) {
					if (S.z[r][k] > 0.5) {
						b = r;
						break;
					}
				}

				if (b < 0 || b == a) {
					continue;
				}
				if (inst_.cap[a][k] < 0.5 || inst_.cap[b][j] < 0.5) {
					continue;
				}

				MatrixDouble z_try = S.z;
				z_try[a][j] = 0.0; z_try[b][j] = 1.0;
				z_try[b][k] = 0.0; z_try[a][k] = 1.0;

				LocalSearchState St_quick;
				if (!evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairReloc, St_quick)) {
					continue;
				}

				quickDeltas.push_back(S.mksp - St_quick.mksp);
				quickStates.push_back(std::move(St_quick));
				pairs.push_back({ j, a, k, b });
			}
		}

		if (!quickDeltas.empty()) {
			const double thresh =
				(static_cast<int>(quickDeltas.size()) <= 6)
				? -std::numeric_limits<double>::infinity()
				: quantile(quickDeltas, 1.0 - opt_.SR_SWAP_QUANTILE_CUTOFF);

			double bestCheapDelta = -std::numeric_limits<double>::infinity();
			int bestCheapIdx = -1;

			for (int i = 0; i < static_cast<int>(quickDeltas.size()); ++i) {
				if (quickDeltas[i] < thresh) {
					continue;
				}

				LocalSearchState St = quickStates[i];
				improveCoupledMrBatchOrder(St, P_light);
				improveCoupledMrOrder(St, P_light);

				const double delta = S.mksp - St.mksp;

				std::cout << "    SR-swap candidate: task" << pairs[i].j
					<< "(R" << pairs[i].a << ")<->task" << pairs[i].k
					<< "(R" << pairs[i].b << ") | delta=" << delta
					<< " | St.mksp=" << St.mksp << "\n";

				if (delta > bestDelta + 1e-9) {
					bestDelta = delta;
					bestS = St;
				}

				if (delta > bestCheapDelta) {
					bestCheapDelta = delta;
					bestCheapIdx = i;
				}
			}

			if (bestCheapIdx >= 0 && bestCheapDelta > -0.1 * S.mksp) {
				const auto p = pairs[bestCheapIdx];
				MatrixDouble z_try = S.z;
				z_try[p.a][p.j] = 0.0; z_try[p.b][p.j] = 1.0;
				z_try[p.b][p.k] = 0.0; z_try[p.a][p.k] = 1.0;

				LocalSearchOptions P_deep = opt_;
				P_deep.MR_BATCH_N_TRIALS = opt_.MR_BATCH_N_TRIALS * 4;
				P_deep.MR_BATCH_CRITICAL_ONLY = false;
				P_deep.MR_MOVE_NUM_BATCH_TRIALS = opt_.MR_MOVE_NUM_BATCH_TRIALS * 4;
				P_deep.MR_MOVE_MAX_POS_TRIALS = opt_.MR_MOVE_MAX_POS_TRIALS;

				LocalSearchState St;
				if (evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairReloc, St)) {
					improveCoupledMrBatchOrder(St, P_deep);
					improveCoupledMrOrder(St, P_deep);

					const double delta = S.mksp - St.mksp;

					std::cout << "    SR-swap DEEP: task" << p.j
						<< "(R" << p.a << ")<->task" << p.k
						<< "(R" << p.b << ") | delta=" << delta
						<< " | St.mksp=" << St.mksp << "\n";

					if (delta > bestDelta + 1e-9) {
						bestDelta = delta;
						bestS = St;
					}
				}
			}
		}

		if (bestDelta > 1e-9) {
			S = std::move(bestS);
			improved = true;
			std::cout << "  SR-assignment improved: mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveMrReallocation(LocalSearchState& S)
	{
		bool improved = false;

		VecInt mrTasks;
		for (int j = 0; j < n(); ++j) {
			if (taskIsMR(j)) {
				mrTasks.push_back(j);
			}
		}

		if (static_cast<int>(mrTasks.size()) > opt_.MAX_MR_TASKS_PER_OUTER) {
			mrTasks.resize(opt_.MAX_MR_TASKS_PER_OUTER);
		}

		double bestDelta = 0.0;
		LocalSearchState bestS = S;
		VecInt bestMove(3, -1);

		for (int j : mrTasks) {
			VecInt Pold;
			VecInt Q;

			for (int r = 0; r < m(); ++r) {
				if (S.z[r][j] > 0.5) {
					Pold.push_back(r);
				}
				else if (inst_.cap[r][j] > 0.5) {
					Q.push_back(r);
				}
			}

			if (Pold.empty() || Q.empty()) {
				continue;
			}

			int candCount = 0;
			for (int a : Pold) {
				for (int b : Q) {
					if (candCount >= opt_.MAX_MR_CANDIDATES_PER_J) {
						break;
					}
					if (a == b) {
						continue;
					}

					MatrixDouble z_try = S.z;
					z_try[a][j] = 0.0;
					z_try[b][j] = 1.0;

					LocalSearchState St;
					if (!evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairMR, St)) {
						++candCount;
						continue;
					}

					const double delta = S.mksp - St.mksp;
					if (delta > bestDelta + 1e-9) {
						bestDelta = delta;
						bestS = std::move(St);
						bestMove = { j, a, b };
					}

					++candCount;
				}
				if (candCount >= opt_.MAX_MR_CANDIDATES_PER_J) {
					break;
				}
			}
		}

		if (bestDelta > 1e-9) {
			S = std::move(bestS);
			improved = true;
			std::cout << "  Best-MR reassign: task " << bestMove[0]
				<< " swap out R" << bestMove[1]
				<< " in R" << bestMove[2]
				<< " | mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveVirtualGapInsertion(LocalSearchState& S)
	{
		bool improved = false;
		//double bestMksp = S.mksp;
		LocalSearchState bestS = S;

		double bestMksp = std::numeric_limits<double>::max();
		const VecDouble ESTcur = computeCurrentEST(S);
		const VecDouble tSeedBase = buildTaskSeedFromState(S);

		for (int s = 0; s < m(); ++s) {
			const VecInt ordV = S.ordV[s];
			const VecInt ordP = S.ordP[s];

			if (ordV.empty()) {
				continue;
			}

			auto gaps = buildPhysicalGaps(S, s, ordP);
			if (static_cast<int>(gaps.size()) > opt_.VGI_MAX_GAPS_PER_TASK) {
				gaps.resize(opt_.VGI_MAX_GAPS_PER_TASK);
			}

			for (int jv : ordV) {
				if (!taskIsVirtual(jv)) {
					continue;
				}

				std::vector<double> candStarts;
				candStarts.push_back(ESTcur[jv]);

				for (const auto& g : gaps) {
					candStarts.push_back(std::max(ESTcur[jv], g.first));
				}

				std::sort(candStarts.begin(), candStarts.end());
				candStarts.erase(std::unique(candStarts.begin(), candStarts.end()), candStarts.end());

				for (double tPin : candStarts) {
					auto ordP_try = S.ordP;
					auto ordV_try = S.ordV;
					ordV_try[s] = insertVirtualByPinTime(ordV_try[s], S.tau, s, jv, tPin);

					VecDouble virtPinnedStart(n(), std::numeric_limits<double>::quiet_NaN());
					virtPinnedStart[jv] = tPin;

					VecDouble tSeed = tSeedBase;
					tSeed[jv] = tPin;

					LocalSearchState St;
					if (!evaluateStateVirtualPinned(
						S, S.z, ordP_try, ordV_try,
						opt_.nRepairFrozen, virtPinnedStart, tSeed, St)) {
						continue;
					}

					if (St.mksp + 1e-9 < bestMksp) {
						bestMksp = St.mksp;
						bestS = std::move(St);
					}
				}
			}
		}

		if (bestMksp + 1e-9 < S.mksp) {
			S = std::move(bestS);
			improved = true;
			std::cout << "  Virtual-gap insertion improved: mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveCoupledMrBatchOrder(LocalSearchState& S, const LocalSearchOptions& P)
	{
		bool improved = false;

		if (!P.MR_BATCH_ENABLE) {
			return false;
		}

		const VecDouble arrivalTmp = robotArrivalToDepot(S);
		const double mkspCurr = *std::max_element(arrivalTmp.begin(), arrivalTmp.end());

		VecInt robotPool;
		if (P.MR_BATCH_CRITICAL_ONLY) {
			for (int r = 0; r < m(); ++r) {
				if (std::abs(arrivalTmp[r] - mkspCurr) <= 1e-9) {
					robotPool.push_back(r);
				}
			}
			if (robotPool.empty()) {
				for (int r = 0; r < m(); ++r) {
					robotPool.push_back(r);
				}
			}
		}
		else {
			for (int r = 0; r < m(); ++r) {
				robotPool.push_back(r);
			}
		}

		double bestDelta = 0.0;
		LocalSearchState bestS = S;
		int skipSeenNeighborhood = 0;
		int skipInverseMove = 0;
		int skipSeenCandidateState = 0;
		int skipSeenEvaluatedState = 0;
		int bestAnchor = -1;
		VecInt bestBatch;
		VecInt bestPerm;

		for (int trial = 0; trial < P.MR_BATCH_N_TRIALS; ++trial) {
			if (robotPool.empty()) {
				break;
			}

			const int sAnchor = robotPool[trial % static_cast<int>(robotPool.size())];
			const VecInt& seqAnchor = S.ordP[sAnchor];
			if (seqAnchor.empty()) {
				continue;
			}

			VecInt mrSeqAnchor;
			for (int j : seqAnchor) {
				if (taskIsMR(j) && !taskIsVirtual(j)) {
					mrSeqAnchor.push_back(j);
				}
			}

			if (static_cast<int>(mrSeqAnchor.size()) < 2) {
				continue;
			}

			const int kBatch = std::min(P.MR_BATCH_K, static_cast<int>(mrSeqAnchor.size()));
			if (kBatch < 2) {
				continue;
			}

			const int maxFirst = static_cast<int>(mrSeqAnchor.size()) - kBatch;
			for (int firstPos = 0; firstPos <= maxFirst; ++firstPos) {
				VecInt batch(mrSeqAnchor.begin() + firstPos, mrSeqAnchor.begin() + firstPos + kBatch);
				const auto permList = buildBatchPermutations(batch, P.MR_BATCH_MAX_PERMS);

				for (const auto& batchPerm : permList) {
					if (batchPerm == batch) {
						continue;
					}

					const std::string neighKey =
						makeBatchNeighborhoodKey(S, sAnchor, batch, batchPerm);
					if (!seenBatchNeighborhoods_.insert(neighKey).second) {
						++skipSeenNeighborhood;
						continue;
					}

					if (isRecentInverseBatchMove(sAnchor, batch, batchPerm)) {
						++skipInverseMove;
						continue;
					}

					auto ordP_try =
						applyCoupledBatchOrder(S.ordP, S.z, batch, batchPerm, sAnchor);

					if (ordP_try[sAnchor] == S.ordP[sAnchor]) {
						continue;
					}

					const std::string candStateKey =
						hashStateOrders(S.z, ordP_try, S.ordV);
					if (seenEvaluatedStateHashes_.find(candStateKey) != seenEvaluatedStateHashes_.end()) {
						++skipSeenCandidateState;
						continue;
					}
					seenEvaluatedStateHashes_.insert(candStateKey);

					LocalSearchState St;
					if (!evaluateState(S, S.z, &ordP_try, &S.ordV, true, P.nRepairFrozen, St)) {
						continue;
					}

					const std::string evalStateKey = hashStateOrders(St);
					if (seenEvaluatedStateHashes_.find(evalStateKey) != seenEvaluatedStateHashes_.end()) {
						++skipSeenEvaluatedState;
						continue;
					}
					seenEvaluatedStateHashes_.insert(evalStateKey);

					const double delta = S.mksp - St.mksp;
					if (delta > bestDelta + 1e-9) {
						bestDelta = delta;
						bestS = std::move(St);
						bestAnchor = sAnchor;
						bestBatch = batch;
						bestPerm = batchPerm;
					}
				}
			}
		}

		if (bestDelta > 1e-3) {
			const std::string acceptedKey = hashStateOrders(bestS);
			seenAcceptedStateHashes_.insert(acceptedKey);
			seenEvaluatedStateHashes_.insert(acceptedKey);

			S = std::move(bestS);
			improved = true;
			skipCheckpointedOrderAfterBatch_ = true;
			rememberAcceptedBatchMove(bestAnchor, bestBatch, bestPerm);

			std::cout << "  Coupled-MR-batch improved: anchor R" << bestAnchor << " | batch ["; for (size_t i = 0; i < bestBatch.size(); ++i) {
				std::cout << bestBatch[i] << (i + 1 < bestBatch.size() ? " " : "");
			}
			std::cout << "] -> [";
			for (size_t i = 0; i < bestPerm.size(); ++i) {
				std::cout << bestPerm[i] << (i + 1 < bestPerm.size() ? " " : "");
			}
			std::cout << "] | mksp=" << S.mksp << "\n";
		}


		if (skipSeenNeighborhood > 0 || skipInverseMove > 0 ||
			skipSeenCandidateState > 0 || skipSeenEvaluatedState > 0) {
			std::cout << "    MR-batch skips: neigh=" << skipSeenNeighborhood
				<< " inverse=" << skipInverseMove
				<< " candState=" << skipSeenCandidateState
				<< " evalState=" << skipSeenEvaluatedState << "\n";
		}

		return improved;
	}

	bool LocalSearch::improveCoupledMrOrder(LocalSearchState& S, const LocalSearchOptions& P)
	{
		bool improved = false;

		double bestDelta = 0.0;
		LocalSearchState bestS = S;
		VecInt bestBatchTasks;

		VecInt mrCandAll;
		for (int j = 0; j < n(); ++j) {
			if (taskIsMR(j) && !taskIsVirtual(j)) {
				mrCandAll.push_back(j);
			}
		}

		if (mrCandAll.empty()) {
			return false;
		}

		if (static_cast<int>(mrCandAll.size()) > P.MR_MOVE_MAX_TASKS) {
			mrCandAll.resize(P.MR_MOVE_MAX_TASKS);
		}

		for (int bt = 0; bt < P.MR_MOVE_NUM_BATCH_TRIALS; ++bt) {
			LocalSearchState S_work = S;
			LocalSearchState S_best_in_batch = S;
			double bestMkspInBatch = S.mksp;
			VecInt bestTasksInBatch;
			VecInt movedTasks;
			int nApplied = 0;

			VecInt mrCand = mrCandAll;

			for (int j : mrCand) {
				if (nApplied >= P.MR_MOVE_BATCH_SIZE) {
					break;
				}
				if (std::find(movedTasks.begin(), movedTasks.end(), j) != movedTasks.end()) {
					continue;
				}

				VecInt Pset;
				for (int r = 0; r < m(); ++r) {
					if (S_work.z[r][j] > 0.5) {
						Pset.push_back(r);
					}
				}
				if (Pset.empty()) {
					continue;
				}

				auto ordP_base = S_work.ordP;
				bool feasible = true;
				VecInt origPosByRobot(m(), 0);

				for (int s : Pset) {
					auto& seq = ordP_base[s];
					auto it = std::find(seq.begin(), seq.end(), j);
					if (it == seq.end()) {
						feasible = false;
						break;
					}
					origPosByRobot[s] = static_cast<int>(std::distance(seq.begin(), it)) + 1;
					seq.erase(it);
				}

				if (!feasible) {
					continue;
				}

				auto ordP_try = ordP_base;
				if (!reinsertMrTaskRegret(
					j, Pset, origPosByRobot, ordP_try,
					S_work.ordV, S_work, P.nRepairFrozen, P.MR_MOVE_MAX_POS_TRIALS)) {
					continue;
				}

				const std::string candStateKey =
					hashStateOrders(S_work.z, ordP_try, S_work.ordV);
				if (seenEvaluatedStateHashes_.find(candStateKey) != seenEvaluatedStateHashes_.end()) {
					continue;
				}
				seenEvaluatedStateHashes_.insert(candStateKey);

				LocalSearchState S_try;
				if (!evaluateState(S_work, S_work.z, &ordP_try, &S_work.ordV, true, P.nRepairFrozen, S_try)) {
					continue;
				}

				const std::string evalStateKey = hashStateOrders(S_try);
				if (seenEvaluatedStateHashes_.find(evalStateKey) != seenEvaluatedStateHashes_.end()) {
					continue;
				}
				seenEvaluatedStateHashes_.insert(evalStateKey);

				S_work = S_try;
				movedTasks.push_back(j);
				++nApplied;

				if (S_try.mksp < bestMkspInBatch - 1e-9) {
					bestMkspInBatch = S_try.mksp;
					S_best_in_batch = S_try;
					bestTasksInBatch = movedTasks;
				}
			}

			if (nApplied == 0) {
				continue;
			}

			const double deltaBatch = S.mksp - S_best_in_batch.mksp;
			if (deltaBatch > bestDelta + 1e-9) {
				bestDelta = deltaBatch;
				bestS = S_best_in_batch;
				bestBatchTasks = bestTasksInBatch;
			}
		}

		if (bestDelta > 1e-9) {
			S = bestS;
			improved = true;

			std::cout << "  Coupled-MR-order improved: "
				<< bestBatchTasks.size()
				<< " checkpointed batched MR moves | tasks=";
			for (int j : bestBatchTasks) {
				std::cout << j << " ";
			}
			std::cout << "| mksp=" << S.mksp << "\n";
		}

		return improved;
	}

	bool LocalSearch::reinsertMrTaskFollowAnchor(
		int jMR,
		int sAnchor,
		std::vector<VecInt>& ordP_try,
		const std::vector<VecInt>& ordV_ref,
		const LocalSearchState& Sseed,
		int nRepairFrozen,
		int maxPosTrials)
	{
		VecInt Pset;
		for (int r = 0; r < m(); ++r) {
			if (Sseed.z[r][jMR] > 0.5) {
				Pset.push_back(r);
			}
		}

		VecInt remaining;
		for (int r : Pset) {
			if (r != sAnchor) {
				remaining.push_back(r);
			}
		}

		while (!remaining.empty()) {
			int bestRobotIdx = -1;
			VecInt bestRobotSeq;
			double bestRobotRegret = -std::numeric_limits<double>::infinity();

			for (int rr = 0; rr < static_cast<int>(remaining.size()); ++rr) {
				const int s = remaining[rr];
				const VecInt& seqBase = ordP_try[s];
				const int L = static_cast<int>(seqBase.size());

				std::set<int> posSet = { 1, (L + 2) / 2, L + 1 };
				for (int p = 1; p <= L + 1 && static_cast<int>(posSet.size()) < 3 + maxPosTrials; ++p) {
					posSet.insert(p);
				}
				std::vector<int> candPos(posSet.begin(), posSet.end());

				std::vector<double> scoreList(candPos.size(), std::numeric_limits<double>::infinity());
				std::vector<VecInt> seqList(candPos.size());

				for (int kk = 0; kk < static_cast<int>(candPos.size()); ++kk) {
					const int p1 = candPos[kk];
					const int p = std::clamp(p1 - 1, 0, L);

					VecInt seqTry = seqBase;
					seqTry.insert(seqTry.begin() + p, jMR);

					auto ordP_eval = ordP_try;
					ordP_eval[s] = seqTry;

					LocalSearchState St;
					if (!evaluateState(Sseed, Sseed.z, &ordP_eval, &ordV_ref, true, nRepairFrozen, St)) {
						continue;
					}

					scoreList[kk] = St.mksp;
					seqList[kk] = seqTry;
				}

				std::vector<int> idx(scoreList.size());
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
					return scoreList[a] < scoreList[b];
					});

				if (idx.empty() || !std::isfinite(scoreList[idx[0]])) {
					continue;
				}

				const double best1 = scoreList[idx[0]];
				const double best2 = (idx.size() >= 2 && std::isfinite(scoreList[idx[1]])) ? scoreList[idx[1]] : best1;
				const double regret = best2 - best1;

				if (regret > bestRobotRegret + 1e-9) {
					bestRobotRegret = regret;
					bestRobotIdx = rr;
					bestRobotSeq = seqList[idx[0]];
				}
			}

			if (bestRobotIdx < 0) {
				return false;
			}

			const int sCommit = remaining[bestRobotIdx];
			ordP_try[sCommit] = bestRobotSeq;
			remaining.erase(remaining.begin() + bestRobotIdx);
		}

		return true;
	}

	bool LocalSearch::reinsertMrTaskRegretAnchor(
		int jMR,
		const VecInt& Pset,
		int sGap,
		int idxGap1Based,
		std::vector<VecInt>& ordP_try,
		const std::vector<VecInt>& ordV_ref,
		const LocalSearchState& Sseed,
		int nRepairFrozen)
	{
		VecInt remaining = Pset;

		while (!remaining.empty()) {
			int bestRobotIdx = -1;
			VecInt bestRobotSeq;
			double bestRobotRegret = -std::numeric_limits<double>::infinity();

			for (int rr = 0; rr < static_cast<int>(remaining.size()); ++rr) {
				const int s = remaining[rr];
				const VecInt& seqBase = ordP_try[s];
				const int L = static_cast<int>(seqBase.size());

				std::vector<int> candPos;
				if (s == sGap) {
					candPos.push_back(std::min(idxGap1Based, L + 1));
				}
				else {
					candPos = { 1, (L + 2) / 2, L + 1 };
					std::sort(candPos.begin(), candPos.end());
					candPos.erase(std::unique(candPos.begin(), candPos.end()), candPos.end());
				}

				std::vector<double> scoreList(candPos.size(), std::numeric_limits<double>::infinity());
				std::vector<VecInt> seqList(candPos.size());

				for (int kk = 0; kk < static_cast<int>(candPos.size()); ++kk) {
					const int p1 = candPos[kk];
					const int p = std::clamp(p1 - 1, 0, L);

					VecInt seqTry = seqBase;
					seqTry.insert(seqTry.begin() + p, jMR);

					auto ordP_eval = ordP_try;
					ordP_eval[s] = seqTry;

					LocalSearchState St;
					if (!evaluateState(Sseed, Sseed.z, &ordP_eval, &ordV_ref, true, nRepairFrozen, St)) {
						continue;
					}

					scoreList[kk] = St.mksp;
					seqList[kk] = seqTry;
				}

				std::vector<int> idx(scoreList.size());
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
					return scoreList[a] < scoreList[b];
					});

				if (idx.empty() || !std::isfinite(scoreList[idx[0]])) {
					continue;
				}

				const double best1 = scoreList[idx[0]];
				const double best2 = (idx.size() >= 2 && std::isfinite(scoreList[idx[1]])) ? scoreList[idx[1]] : best1;
				const double regret = best2 - best1;

				if (regret > bestRobotRegret + 1e-9) {
					bestRobotRegret = regret;
					bestRobotIdx = rr;
					bestRobotSeq = seqList[idx[0]];
				}
			}

			if (bestRobotIdx < 0) {
				return false;
			}

			const int sCommit = remaining[bestRobotIdx];
			ordP_try[sCommit] = bestRobotSeq;
			remaining.erase(remaining.begin() + bestRobotIdx);
		}

		return true;
	}

	bool LocalSearch::reinsertMrTaskRegret(
		int jMR,
		const VecInt& Pset,
		const VecInt& origPosByRobot1Based,
		std::vector<VecInt>& ordP_out,
		const std::vector<VecInt>& ordV_ref,
		const LocalSearchState& Sseed,
		int nRepairFrozen,
		int maxPosTrials)
	{
		VecInt remaining = Pset;

		while (!remaining.empty()) {
			int bestRobotIdx = -1;
			VecInt bestRobotSeq;
			double bestRobotRegret = -std::numeric_limits<double>::infinity();

			for (int rr = 0; rr < static_cast<int>(remaining.size()); ++rr) {
				const int s = remaining[rr];
				const VecInt& seqBase = ordP_out[s];
				const int L = static_cast<int>(seqBase.size());
				const int origPos = origPosByRobot1Based[s];

				std::vector<int> candPos = { 1, (L + 2) / 2, L + 1 };
				std::sort(candPos.begin(), candPos.end());
				candPos.erase(std::unique(candPos.begin(), candPos.end()), candPos.end());

				candPos.erase(std::remove(candPos.begin(), candPos.end(), origPos), candPos.end());

				std::vector<int> allPos;
				for (int p = 1; p <= L + 1; ++p) {
					if (p != origPos) {
						allPos.push_back(p);
					}
				}

				std::vector<int> extra;
				for (int p : allPos) {
					if (std::find(candPos.begin(), candPos.end(), p) == candPos.end()) {
						extra.push_back(p);
					}
				}

				for (int k = 0; k < static_cast<int>(extra.size()) && k < maxPosTrials; ++k) {
					candPos.push_back(extra[k]);
				}

				std::sort(candPos.begin(), candPos.end());
				candPos.erase(std::unique(candPos.begin(), candPos.end()), candPos.end());

				if (candPos.empty()) {
					continue;
				}

				std::vector<double> scoreList(candPos.size(), std::numeric_limits<double>::infinity());
				std::vector<VecInt> seqList(candPos.size());

				for (int kk = 0; kk < static_cast<int>(candPos.size()); ++kk) {
					const int p1 = candPos[kk];
					const int p = std::clamp(p1 - 1, 0, L);

					VecInt seqTry = seqBase;
					seqTry.insert(seqTry.begin() + p, jMR);

					auto ordP_eval = ordP_out;
					ordP_eval[s] = seqTry;

					LocalSearchState St;
					if (!evaluateState(Sseed, Sseed.z, &ordP_eval, &ordV_ref, true, nRepairFrozen, St)) {
						continue;
					}

					scoreList[kk] = St.mksp;
					seqList[kk] = seqTry;
				}

				std::vector<int> idx(scoreList.size());
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
					return scoreList[a] < scoreList[b];
					});

				if (idx.empty() || !std::isfinite(scoreList[idx[0]])) {
					continue;
				}

				const double best1 = scoreList[idx[0]];
				const double best2 = (idx.size() >= 2 && std::isfinite(scoreList[idx[1]])) ? scoreList[idx[1]] : best1;
				const double regret = best2 - best1;

				if (regret > bestRobotRegret + 1e-9) {
					bestRobotRegret = regret;
					bestRobotIdx = rr;
					bestRobotSeq = seqList[idx[0]];
				}
			}

			if (bestRobotIdx < 0) {
				return false;
			}

			const int sCommit = remaining[bestRobotIdx];
			ordP_out[sCommit] = bestRobotSeq;
			remaining.erase(remaining.begin() + bestRobotIdx);
		}

		return true;
	}

	bool LocalSearch::improveRuinRecreate(LocalSearchState& S)
	{
		bool improved = false;
		if (!opt_.RR2_ENABLE) {
			return false;
		}

		const VecDouble arrivalTmp = robotArrivalToDepot(S);
		const double mkspCurr = *std::max_element(arrivalTmp.begin(), arrivalTmp.end());

		VecInt critRobots;
		for (int r = 0; r < m(); ++r) {
			if (std::abs(arrivalTmp[r] - mkspCurr) <= 1e-9) {
				critRobots.push_back(r);
			}
		}

		std::cout << "  RR2: critical robots = [";
		for (size_t i = 0; i < critRobots.size(); ++i) {
			std::cout << critRobots[i] << (i + 1 < critRobots.size() ? " " : "");
		}
		std::cout << "] | mksp = " << mkspCurr << "\n";

		double bestDelta = 0.0;
		LocalSearchState bestS = S;
		std::vector<VecInt> bestMrSwaps;
		VecInt bestSrTasks;
		int totalDeep = 0;
		int totalSkip = 0;
		int totalComboExhausted = 0;

		std::unordered_map<std::string, double> evalCache;

		for (int trial = 0; trial < opt_.RR2_NUM_TRIALS; ++trial) {
			VecInt mrCandA;
			for (int cr : critRobots) {
				for (int j = 0; j < n(); ++j) {
					if (S.z[cr][j] > 0.5 && taskIsMR(j) && !taskIsVirtual(j)) {
						if (std::find(mrCandA.begin(), mrCandA.end(), j) == mrCandA.end()) {
							mrCandA.push_back(j);
						}
					}
				}
			}

			VecInt mrCandB;
			for (int cr : critRobots) {
				for (int j = 0; j < n(); ++j) {
					if (inst_.cap[cr][j] > 0.5 && taskIsMR(j) && !taskIsVirtual(j) && S.z[cr][j] < 0.5) {
						if (std::find(mrCandB.begin(), mrCandB.end(), j) == mrCandB.end()) {
							mrCandB.push_back(j);
						}
					}
				}
			}

			VecInt mrCand = mrCandA;
			for (int j : mrCandB) {
				if (std::find(mrCand.begin(), mrCand.end(), j) == mrCand.end()) {
					mrCand.push_back(j);
				}
			}
			if (mrCand.empty()) {
				for (int j = 0; j < n(); ++j) {
					if (taskIsMR(j) && !taskIsVirtual(j)) {
						mrCand.push_back(j);
					}
				}
			}

			if (static_cast<int>(mrCand.size()) > opt_.RR2_MAX_MR_TASKS) {
				mrCand.resize(opt_.RR2_MAX_MR_TASKS);
			}

			std::vector<std::vector<VecInt>> mrSwapSets(mrCand.size());
			for (int ii = 0; ii < static_cast<int>(mrCand.size()); ++ii) {
				const int j = mrCand[ii];
				VecInt Pold, capable, nonPart;

				for (int r = 0; r < m(); ++r) {
					if (S.z[r][j] > 0.5) {
						Pold.push_back(r);
					}
					if (inst_.cap[r][j] > 0.5) {
						capable.push_back(r);
					}
				}

				for (int r : capable) {
					if (std::find(Pold.begin(), Pold.end(), r) == Pold.end()) {
						nonPart.push_back(r);
					}
				}

				for (int a : Pold) {
					for (int b : nonPart) {
						mrSwapSets[ii].push_back({ j, a, b });
						if (static_cast<int>(mrSwapSets[ii].size()) >= opt_.RR2_MAX_MR_SWAPS_PER_J) {
							break;
						}
					}
					if (static_cast<int>(mrSwapSets[ii].size()) >= opt_.RR2_MAX_MR_SWAPS_PER_J) {
						break;
					}
				}
			}

			std::vector<std::vector<VecInt>> mrCombos;
			for (int ii = 0; ii < static_cast<int>(mrCand.size()); ++ii) {
				for (const auto& sw : mrSwapSets[ii]) {
					mrCombos.push_back({ sw });
				}
			}

			for (int ii = 0; ii < static_cast<int>(mrCand.size()); ++ii) {
				for (int jj = ii + 1; jj < static_cast<int>(mrCand.size()); ++jj) {
					for (const auto& s1 : mrSwapSets[ii]) {
						for (const auto& s2 : mrSwapSets[jj]) {
							mrCombos.push_back({ s1, s2 });
						}
					}
				}
			}

			if (opt_.RR2_ALSO_PURE_SR) {
				mrCombos.push_back({});
			}

			if (static_cast<int>(mrCombos.size()) > opt_.RR2_MAX_MR_COMBOS) {
				mrCombos.resize(opt_.RR2_MAX_MR_COMBOS);
				if (opt_.RR2_ALSO_PURE_SR) {
					bool hasPure = false;
					for (const auto& c : mrCombos) {
						if (c.empty()) {
							hasPure = true;
							break;
						}
					}
					if (!hasPure) {
						mrCombos.back().clear();
					}
				}
			}

			for (int mc = 0; mc < static_cast<int>(mrCombos.size()); ++mc) {
				const auto& combo = mrCombos[mc];

				MatrixDouble z_mr = S.z;
				VecInt disruptedRobots = critRobots;

				for (const auto& sw : combo) {
					const int j_mr = sw[0];
					const int oldR = sw[1];
					const int newR = sw[2];
					z_mr[oldR][j_mr] = 0.0;
					z_mr[newR][j_mr] = 1.0;

					if (std::find(disruptedRobots.begin(), disruptedRobots.end(), oldR) == disruptedRobots.end()) {
						disruptedRobots.push_back(oldR);
					}
					if (std::find(disruptedRobots.begin(), disruptedRobots.end(), newR) == disruptedRobots.end()) {
						disruptedRobots.push_back(newR);
					}
				}

				std::sort(disruptedRobots.begin(), disruptedRobots.end());
				disruptedRobots.erase(std::unique(disruptedRobots.begin(), disruptedRobots.end()), disruptedRobots.end());

				VecInt srToStrip;
				for (int dr : disruptedRobots) {
					for (int j = 0; j < n(); ++j) {
						if (z_mr[dr][j] <= 0.5) continue;
						if (taskIsMR(j)) continue;
						if (taskIsVirtual(j) && !opt_.RR2_ALSO_VIRT) continue;
						if (std::find(srToStrip.begin(), srToStrip.end(), j) == srToStrip.end()) {
							srToStrip.push_back(j);
						}
					}
				}

				if (static_cast<int>(srToStrip.size()) > opt_.RR2_MAX_SR_STRIP) {
					srToStrip.resize(opt_.RR2_MAX_SR_STRIP);
				}

				if (srToStrip.empty() && combo.empty()) {
					continue;
				}

				const std::string comboKey =
					makeRr2ComboKey(z_mr, disruptedRobots, srToStrip, combo);

				if (exhaustedRr2Combos_.find(comboKey) != exhaustedRr2Combos_.end()) {
					++totalSkip;
					++totalComboExhausted;
					std::cout << "      RR2 combo skipped: exhausted\n";
					continue;
				}

				if (trial == 0) {
					std::cout << "    RR2 combo " << mc
						<< ": MR swaps=" << combo.size()
						<< ", disrupted=[";
					for (size_t i = 0; i < disruptedRobots.size(); ++i) {
						std::cout << disruptedRobots[i] << (i + 1 < disruptedRobots.size() ? " " : "");
					}
					std::cout << "], stripped=[";
					for (size_t i = 0; i < srToStrip.size(); ++i) {
						std::cout << srToStrip[i] << (i + 1 < srToStrip.size() ? " " : "");
					}
					std::cout << "]\n";
				}

				MatrixDouble z_stripped = z_mr;
				auto ordP_stripped = S.ordP;
				auto ordV_stripped = S.ordV;

				for (int j : srToStrip) {
					for (int s = 0; s < m(); ++s) {
						z_stripped[s][j] = 0.0;
					}
					for (int s = 0; s < m(); ++s) {
						removeTaskFromSequence(ordP_stripped[s], j);
						removeTaskFromSequence(ordV_stripped[s], j);
					}
				}

				for (const auto& sw : combo) {
					const int j_mr = sw[0];
					const int oldR = sw[1];
					const int newR = sw[2];
					removeTaskFromSequence(ordP_stripped[oldR], j_mr);
					if (!containsTask(ordP_stripped[newR], j_mr)) {
						ordP_stripped[newR].push_back(j_mr);
					}
				}

				VecInt srPhysStripped;
				VecInt srVirtStripped;
				for (int j : srToStrip) {
					if (taskIsVirtual(j)) srVirtStripped.push_back(j);
					else srPhysStripped.push_back(j);
				}

				std::vector<VecInt> capLists(srPhysStripped.size());
				for (int ii = 0; ii < static_cast<int>(srPhysStripped.size()); ++ii) {
					const int j = srPhysStripped[ii];
					for (int r = 0; r < m(); ++r) {
						if (inst_.cap[r][j] > 0.5) {
							capLists[ii].push_back(r);
						}
					}
				}

				std::vector<VecInt> virtCapLists(srVirtStripped.size());
				for (int ii = 0; ii < static_cast<int>(srVirtStripped.size()); ++ii) {
					const int j = srVirtStripped[ii];
					for (int r = 0; r < m(); ++r) {
						if (inst_.cap[r][j] > 0.5) {
							virtCapLists[ii].push_back(r);
						}
					}
				}

				long long totalFeasible = 1;
				for (const auto& caps : capLists) {
					totalFeasible *= static_cast<long long>(caps.size());
					if (totalFeasible > opt_.RR2_EXHAUST_LIMIT) {
						break;
					}
				}

				if (totalFeasible <= opt_.RR2_EXHAUST_LIMIT) {
					const auto assignments = enumerateAllAssignments(capLists);
					const int nAssign = static_cast<int>(assignments.size());

					std::vector<double> cheapMksp(nAssign, std::numeric_limits<double>::infinity());
					std::vector<std::string> zHashes(nAssign);
					int nScreened = 0;
					int nCheapHit = 0;

					for (int aa = 0; aa < nAssign; ++aa) {
						MatrixDouble z_try = z_stripped;
						for (int ii = 0; ii < static_cast<int>(srPhysStripped.size()); ++ii) {
							z_try[assignments[aa][ii]][srPhysStripped[ii]] = 1.0;
						}
						z_try = assignVirtuals(z_try, srVirtStripped, virtCapLists);

						const std::string zHash = hashAssignment(z_try);
						zHashes[aa] = zHash;

						const auto it = evalCache.find(zHash);
						if (it != evalCache.end()) {
							cheapMksp[aa] = it->second;
							++nCheapHit;
						}
						else {
							LocalSearchState St;
							if (evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairReloc, St)) {
								cheapMksp[aa] = St.mksp;
							}
							++nScreened;
						}
					}

					std::cout << "      Exhaustive: " << nAssign
						<< " assignments for " << srPhysStripped.size()
						<< " tasks (new=" << nScreened
						<< ", cached=" << nCheapHit << ")\n";

					// If this neighborhood is mostly cached already, do not spend more effort here.
					if (nScreened <= 2 && nCheapHit >= nAssign - 2) {
						exhaustedRr2Combos_.insert(comboKey);
						std::cout << "      RR2 combo marked exhausted (cache-dominated)\n";
						continue;
					}


					if (nScreened == 0) {
						exhaustedRr2Combos_.insert(comboKey);
						std::cout << "      RR2 combo marked exhausted\n";
						continue;
					}

					std::vector<int> sortIdx(nAssign);
					std::iota(sortIdx.begin(), sortIdx.end(), 0);
					std::sort(sortIdx.begin(), sortIdx.end(), [&](int a, int b) {
						return cheapMksp[a] < cheapMksp[b];
						});

					int nDeepDone = 0;
					for (int kk = 0; kk < nAssign; ++kk) {
						if (nDeepDone >= opt_.RR2_EXHAUST_TOP_K) {
							break;
						}

						const int aa = sortIdx[kk];
						const std::string& zHash = zHashes[aa];

						if (evalCache.find(zHash) != evalCache.end()) {
							++totalSkip;
							continue;
						}

						MatrixDouble z_try = z_stripped;
						for (int ii = 0; ii < static_cast<int>(srPhysStripped.size()); ++ii) {
							z_try[assignments[aa][ii]][srPhysStripped[ii]] = 1.0;
						}
						z_try = assignVirtuals(z_try, srVirtStripped, virtCapLists);

						LocalSearchState Squick;
						if (!evaluateState(S, z_try, nullptr, nullptr, false, opt_.nRepairReloc, Squick)) {
							continue;
						}

						// Skip obviously bad candidates before deep/thorough evaluation.
						if (Squick.mksp > S.mksp + opt_.RR2_EVAL_MAX_DEGRADATION) {
							++totalSkip;
							evalCache[zHash] = Squick.mksp;  // optional: cache cheap result
							continue;
						}

						++nDeepDone;
						const auto evalRes = (nDeepDone == 1) ? thoroughEval(z_try, S) : deepEval(z_try, S);

						++totalDeep;
						evalCache[zHash] = evalRes.second;

						const double delta = S.mksp - evalRes.second;
						if (delta > bestDelta + 1e-9) {
							bestDelta = delta;
							bestS = evalRes.first;
							bestSrTasks = srToStrip;
							bestMrSwaps = combo;

							std::cout << "  RR2: no improvement found ("
								<< opt_.RR2_NUM_TRIALS << " trials, "
								<< totalDeep << " deep evals, "
								<< totalSkip << " skipped, "
								<< totalComboExhausted << " combo-exhausted)\n";
						}
					}
				}
				else {
					bool anyNewEval = false;

					MatrixDouble z_greedy = buildGreedyAssignment(
						z_stripped, srPhysStripped, srVirtStripped, capLists, virtCapLists);

					std::string zHash = hashAssignment(z_greedy);
					if (evalCache.find(zHash) == evalCache.end()) {
						anyNewEval = true;
						LocalSearchState Squick;
						if (!evaluateState(S, z_greedy, nullptr, nullptr, false, opt_.nRepairReloc, Squick)) {
							++totalSkip;
						}
						else if (Squick.mksp > S.mksp + opt_.RR2_EVAL_MAX_DEGRADATION) {
							++totalSkip;
							evalCache[zHash] = Squick.mksp;
						}
						else {
							const auto evalRes = deepEval(z_greedy, S);
							++totalDeep;
							evalCache[zHash] = evalRes.second;

							const double delta = S.mksp - evalRes.second;
							if (delta > bestDelta + 1e-9) {
								bestDelta = delta;
								bestS = evalRes.first;
								bestSrTasks = srToStrip;
								bestMrSwaps = combo;

								std::cout << "    RR2 greedy: delta=" << delta
									<< " mksp=" << evalRes.second
									<< " | MR swaps=" << combo.size() << "\n";
							}
						}

					}
					else {
						++totalSkip;
					}

					for (int pp = 0; pp < opt_.RR2_NUM_PERTURBATIONS; ++pp) {
						MatrixDouble z_pert = perturbAssignment(z_greedy, srPhysStripped, capLists);
						z_pert = assignVirtuals(z_pert, srVirtStripped, virtCapLists);

						zHash = hashAssignment(z_pert);
						if (evalCache.find(zHash) != evalCache.end()) {
							++totalSkip;
							continue;
						}

						anyNewEval = true;
						LocalSearchState Squick;
						if (!evaluateState(S, z_pert, nullptr, nullptr, false, opt_.nRepairReloc, Squick)) {
							++totalSkip;
							continue;
						}

						if (Squick.mksp > S.mksp + opt_.RR2_EVAL_MAX_DEGRADATION) {
							++totalSkip;
							evalCache[zHash] = Squick.mksp;
							continue;
						}

						const auto evalRes = deepEval(z_pert, S);
						++totalDeep;
						evalCache[zHash] = evalRes.second;

						const double delta = S.mksp - evalRes.second;
						if (delta > bestDelta + 1e-9) {
							bestDelta = delta;
							bestS = evalRes.first;
							bestSrTasks = srToStrip;
							bestMrSwaps = combo;

							std::cout << "    RR2 perturb[" << pp + 1
								<< "]: delta=" << delta
								<< " mksp=" << evalRes.second << "\n";
						}
					}

					if (!anyNewEval) {
						exhaustedRr2Combos_.insert(comboKey);
					}
				}
			}
		}

		if (bestDelta > 1e-9) {
			S = bestS;
			improved = true;

			std::cout << "  RR2 improved: ";
			if (bestMrSwaps.empty()) {
				std::cout << "pure-SR";
			}
			else {
				for (size_t kk = 0; kk < bestMrSwaps.size(); ++kk) {
					const auto& sw = bestMrSwaps[kk];
					std::cout << "MR" << sw[0] << ":R" << sw[1] << "->R" << sw[2];
					if (kk + 1 < bestMrSwaps.size()) {
						std::cout << ", ";
					}
				}
			}

			std::cout << " | SR=[";
			for (size_t i = 0; i < bestSrTasks.size(); ++i) {
				std::cout << bestSrTasks[i] << (i + 1 < bestSrTasks.size() ? " " : "");
			}
			std::cout << "] | delta=" << bestDelta
				<< " | mksp=" << S.mksp << "\n";
		}
		else {
			std::cout << "  RR2: no improvement found ("
				<< opt_.RR2_NUM_TRIALS << " trials, "
				<< totalDeep << " deep evals, "
				<< totalSkip << " skipped)\n";
		}

		return improved;
	}

	RepairResult LocalSearch::callRepairUnfrozen(
		const MatrixDouble& zCand,
		const VecDouble& thetaWarm,
		const VecDouble& tWarm,
		int nRepairIters) const
	{
		RepairOptions opt;
		opt.nIters = std::max(1, nRepairIters);
		opt.freezeOrders = false;
		return repairer_.repairPushforward(zCand, inst_, thetaWarm, tWarm, opt);
	}

	RepairResult LocalSearch::callRepairFrozen(
		const MatrixDouble& zCand,
		const VecDouble& thetaWarm,
		const VecDouble& tWarm,
		const std::vector<VecInt>& ordP,
		const std::vector<VecInt>& ordV,
		int nRepairIters) const
	{
		RepairOptions opt;
		opt.nIters = std::max(1, nRepairIters);
		opt.freezeOrders = true;
		opt.ordersPhys0 = ordP;
		opt.ordersVirt0 = ordV;
		return repairer_.repairPushforward(zCand, inst_, thetaWarm, tWarm, opt);
	}

	RepairResult LocalSearch::callRepairFrozenVirtualPinned(
		const MatrixDouble& zCand,
		const VecDouble& thetaWarm,
		const VecDouble& tWarm,
		const std::vector<VecInt>& ordP,
		const std::vector<VecInt>& ordV,
		int nRepairIters,
		const VecDouble& virtPinnedStart) const
	{
		RepairOptions opt;
		opt.nIters = std::max(1, nRepairIters);
		opt.freezeOrders = true;
		opt.ordersPhys0 = ordP;
		opt.ordersVirt0 = ordV;
		opt.virtPinnedStart = virtPinnedStart;
		return repairer_.repairPushforward(zCand, inst_, thetaWarm, tWarm, opt);
	}

	int LocalSearch::m() const { return inst_.m; }
	int LocalSearch::n() const { return inst_.n; }

	bool LocalSearch::taskIsMR(int j) const { return inst_.isMR[j]; }
	bool LocalSearch::taskIsVirtual(int j) const { return inst_.isVirtual[j]; }

	double LocalSearch::currentTaskDuration(int s, int j) const
	{
		if (taskIsMR(j)) {
			double best = 0.0;
			bool found = false;
			for (int r = 0; r < m(); ++r) {
				const double v = inst_.svcPhysSJ[r][j];
				if (std::isfinite(v) && v >= 0.0) {
					best = found ? std::max(best, v) : v;
					found = true;
				}
			}
			return found ? best : 0.0;
		}

		const double raw = taskIsVirtual(j) ? inst_.svcVirtSJ[s][j] : inst_.svcPhysSJ[s][j];
		if (!std::isfinite(raw) || raw < 0.0) {
			return 0.0;
		}
		return raw;
	}

	double LocalSearch::thetaTaskStart(const LocalSearchState& S, int j) const
	{
		if (j < 0 || j >= static_cast<int>(S.theta.size())) {
			return 0.0;
		}
		return std::isfinite(S.theta[j]) ? S.theta[j] : 0.0;
	}

	VecDouble LocalSearch::robotArrivalToDepot(const LocalSearchState& S) const
	{
		const ScheduleScore score = scorer_.scoreScheduleExact(
			inst_, S.z, S.tau, S.ordP, S.ordV, S.lastPhysNode);
		return score.arrival;
	}

	VecDouble LocalSearch::computeCurrentEST(const LocalSearchState& S) const
	{
		VecDouble C(n(), 0.0);

		for (int j = 0; j < n(); ++j) {
			VecInt Pset;
			for (int r = 0; r < m(); ++r) {
				if (S.z[r][j] > 0.5) {
					Pset.push_back(r);
				}
			}
			if (Pset.empty()) {
				continue;
			}

			if (taskIsMR(j)) {
				const double tj = thetaTaskStart(S, j);
				C[j] = tj + currentTaskDuration(Pset.front(), j);
			}
			else {
				const int s = Pset.front();
				const double tj = std::isfinite(S.tau[s][j]) ? S.tau[s][j] : 0.0;
				C[j] = tj + currentTaskDuration(s, j);
			}
		}

		VecDouble EST(n(), 0.0);
		for (const auto& e : inst_.predPairs) {
			EST[e.to] = std::max(EST[e.to], C[e.from]);
		}

		return EST;
	}

	VecDouble LocalSearch::buildTaskSeedFromState(const LocalSearchState& S) const
	{
		VecDouble tSeed(n(), 0.0);
		for (int j = 0; j < n(); ++j) {
			if (taskIsMR(j)) {
				tSeed[j] = thetaTaskStart(S, j);
			}
			else {
				for (int r = 0; r < m(); ++r) {
					if (S.z[r][j] > 0.5 && std::isfinite(S.tau[r][j])) {
						tSeed[j] = S.tau[r][j];
						break;
					}
				}
			}
		}
		return tSeed;
	}

	std::vector<std::pair<double, double>> LocalSearch::buildPhysicalGaps(
		const LocalSearchState& S,
		int s,
		const VecInt& ordP) const
	{
		std::vector<std::pair<double, double>> gaps;

		double rel = 0.0;
		if (!inst_.Tstart.empty()) {
			double best = std::numeric_limits<double>::infinity();
			for (int j = 0; j < n(); ++j) {
				const double v = inst_.Tstart[s][j];
				if (std::isfinite(v)) {
					best = std::min(best, v);
				}
			}
			rel = std::isfinite(best) ? best : 0.0;
		}

		if (ordP.empty()) {
			gaps.push_back({ rel, std::numeric_limits<double>::infinity() });
			return gaps;
		}

		struct Interval { double st; double en; };
		std::vector<Interval> physIntervals;
		physIntervals.reserve(ordP.size());

		for (int j : ordP) {
			const double st = taskIsMR(j) ? thetaTaskStart(S, j) : S.tau[s][j];
			const double d = currentTaskDuration(s, j);
			physIntervals.push_back({ st, st + d });
		}

		gaps.push_back({ rel, physIntervals.front().st });
		for (int k = 0; k + 1 < static_cast<int>(physIntervals.size()); ++k) {
			gaps.push_back({ physIntervals[k].en, physIntervals[k + 1].st });
		}
		gaps.push_back({ physIntervals.back().en, std::numeric_limits<double>::infinity() });

		return gaps;
	}

	VecInt LocalSearch::insertVirtualByPinTime(
		const VecInt& ordVold,
		const MatrixDouble& tauCurrent,
		int s,
		int jv,
		double tPin) const
	{
		VecInt ordVnew = ordVold;
		removeTaskFromSequence(ordVnew, jv);

		if (ordVnew.empty()) {
			ordVnew.push_back(jv);
			return ordVnew;
		}

		int pos = -1;
		for (int i = 0; i < static_cast<int>(ordVnew.size()); ++i) {
			const int j = ordVnew[i];
			double st = std::numeric_limits<double>::infinity();
			if (std::isfinite(tauCurrent[s][j])) {
				st = tauCurrent[s][j];
			}
			if (st >= tPin) {
				pos = i;
				break;
			}
		}

		if (pos < 0) {
			ordVnew.push_back(jv);
		}
		else {
			ordVnew.insert(ordVnew.begin() + pos, jv);
		}
		return ordVnew;
	}

	MatrixDouble LocalSearch::assignVirtuals(
		const MatrixDouble& zIn,
		const VecInt& srVirt,
		const std::vector<VecInt>& virtCapLists) const
	{
		MatrixDouble zOut = zIn;
		VecDouble virtLoad(m(), 0.0);

		for (int r = 0; r < m(); ++r) {
			for (int j = 0; j < n(); ++j) {
				virtLoad[r] += zOut[r][j];
			}
		}

		for (int ii = 0; ii < static_cast<int>(srVirt.size()); ++ii) {
			const int j = srVirt[ii];
			const VecInt& caps = virtCapLists[ii];
			if (caps.empty()) {
				continue;
			}

			int bestR = caps.front();
			double bestLoad = virtLoad[bestR];
			for (int r : caps) {
				if (virtLoad[r] < bestLoad) {
					bestLoad = virtLoad[r];
					bestR = r;
				}
			}

			zOut[bestR][j] = 1.0;
			virtLoad[bestR] += 1.0;
		}

		return zOut;
	}

	MatrixDouble LocalSearch::buildGreedyAssignment(
		const MatrixDouble& zStripped,
		const VecInt& srPhys,
		const VecInt& srVirt,
		const std::vector<VecInt>& capLists,
		const std::vector<VecInt>& virtCapLists) const
	{
		MatrixDouble zOut = zStripped;

		VecDouble workload(m(), 0.0);
		for (int s = 0; s < m(); ++s) {
			for (int j = 0; j < n(); ++j) {
				if (zOut[s][j] > 0.5) {
					workload[s] += currentTaskDuration(s, j);
				}
			}
		}

		for (int ii = 0; ii < static_cast<int>(srPhys.size()); ++ii) {
			const int j = srPhys[ii];
			const VecInt& caps = capLists[ii];
			if (caps.empty()) {
				continue;
			}

			int bestR = caps.front();
			double bestW = workload[bestR];
			for (int r : caps) {
				if (workload[r] < bestW) {
					bestW = workload[r];
					bestR = r;
				}
			}

			zOut[bestR][j] = 1.0;
			workload[bestR] += currentTaskDuration(bestR, j);
		}

		zOut = assignVirtuals(zOut, srVirt, virtCapLists);
		return zOut;
	}

	MatrixDouble LocalSearch::perturbAssignment(
		const MatrixDouble& zGreedy,
		const VecInt& srPhys,
		const std::vector<VecInt>& capLists) const
	{
		MatrixDouble zPert = zGreedy;
		const int nPhys = static_cast<int>(srPhys.size());
		if (nPhys < 2) {
			return zPert;
		}

		const int nSwaps = std::min(2, nPhys / 2);
		int swapCount = 0;

		for (int ii = 0; ii + 1 < nPhys && swapCount < nSwaps; ii += 2) {
			const int j1 = srPhys[ii];
			const int j2 = srPhys[ii + 1];

			int r1 = -1, r2 = -1;
			for (int r = 0; r < m(); ++r) {
				if (zPert[r][j1] > 0.5) r1 = r;
				if (zPert[r][j2] > 0.5) r2 = r;
			}

			if (r1 < 0 || r2 < 0 || r1 == r2) {
				continue;
			}

			if (inst_.cap[r2][j1] < 0.5 || inst_.cap[r1][j2] < 0.5) {
				continue;
			}

			zPert[r1][j1] = 0.0; zPert[r2][j1] = 1.0;
			zPert[r2][j2] = 0.0; zPert[r1][j2] = 1.0;
			++swapCount;
		}

		if (swapCount == 0 && nPhys >= 1) {
			const int j = srPhys.front();
			const VecInt& caps = capLists.front();

			if (caps.size() >= 2) {
				int oldR = -1;
				for (int r = 0; r < m(); ++r) {
					if (zPert[r][j] > 0.5) {
						oldR = r;
						break;
					}
				}

				for (int newR : caps) {
					if (newR != oldR) {
						zPert[oldR][j] = 0.0;
						zPert[newR][j] = 1.0;
						break;
					}
				}
			}
		}

		return zPert;
	}

	std::pair<LocalSearchState, double> LocalSearch::deepEval(
		const MatrixDouble& zTry,
		const LocalSearchState& SBase) const
	{
		LocalSearchState Sout;
		LocalSearch& self = const_cast<LocalSearch&>(*this);

		if (!self.evaluateState(SBase, zTry, nullptr, nullptr, false, opt_.nRepairReloc, Sout)) {
			return { SBase, std::numeric_limits<double>::infinity() };
		}

		// Snapshot outer-search caches so RR2 local polishing does not pollute them.
		auto savedSeenAcceptedStateHashes = self.seenAcceptedStateHashes_;
		auto savedSeenEvaluatedStateHashes = self.seenEvaluatedStateHashes_;
		auto savedSeenBatchNeighborhoods = self.seenBatchNeighborhoods_;
		auto savedExhaustedRr2Combos = self.exhaustedRr2Combos_;
		auto savedRecentAcceptedBatchMoves = self.recentAcceptedBatchMoves_;
		const bool savedSkipCheckpointedOrderAfterBatch = self.skipCheckpointedOrderAfterBatch_;

		self.improveIntraOrder(Sout, opt_.RR2_POLISH_N_INNER);

		LocalSearchOptions Pmr = opt_;
		Pmr.MR_BATCH_N_TRIALS = 3;
		Pmr.MR_BATCH_CRITICAL_ONLY = true;
		Pmr.MR_BATCH_K = 2;
		Pmr.MR_MOVE_NUM_BATCH_TRIALS = 1;
		Pmr.MR_MOVE_MAX_POS_TRIALS = 1;
		Pmr.MR_MOVE_BATCH_SIZE = 1;

		// Only do MR polishing if the candidate is not already much worse than the base.
		if (Sout.mksp <= SBase.mksp + opt_.RR2_POLISH_MAX_DEGRADATION) {
			self.improveCoupledMrBatchOrder(Sout, Pmr);
			self.improveCoupledMrOrder(Sout, Pmr);
		}
		else {
			std::cout << "      deepEval: skipped MR polish (mksp="
				<< Sout.mksp << ", base=" << SBase.mksp
				<< ", max degradation=" << opt_.RR2_POLISH_MAX_DEGRADATION << ")\n";
		}
		// Restore outer-search caches.
		self.seenAcceptedStateHashes_ = std::move(savedSeenAcceptedStateHashes);
		self.seenEvaluatedStateHashes_ = std::move(savedSeenEvaluatedStateHashes);
		self.seenBatchNeighborhoods_ = std::move(savedSeenBatchNeighborhoods);
		self.exhaustedRr2Combos_ = std::move(savedExhaustedRr2Combos);
		self.recentAcceptedBatchMoves_ = std::move(savedRecentAcceptedBatchMoves);
		self.skipCheckpointedOrderAfterBatch_ = savedSkipCheckpointedOrderAfterBatch;

		return { Sout, Sout.mksp };
	}

	std::pair<LocalSearchState, double> LocalSearch::thoroughEval(
		const MatrixDouble& zTry,
		const LocalSearchState& SBase) const
	{
		LocalSearchState Sout;
		LocalSearch& self = const_cast<LocalSearch&>(*this);

		if (!self.evaluateState(SBase, zTry, nullptr, nullptr, false, opt_.nRepairReloc, Sout)) {
			return { SBase, std::numeric_limits<double>::infinity() };
		}

		// Snapshot outer-search caches so RR2 local polishing does not pollute them.
		auto savedSeenAcceptedStateHashes = self.seenAcceptedStateHashes_;
		auto savedSeenEvaluatedStateHashes = self.seenEvaluatedStateHashes_;
		auto savedSeenBatchNeighborhoods = self.seenBatchNeighborhoods_;
		auto savedExhaustedRr2Combos = self.exhaustedRr2Combos_;
		auto savedRecentAcceptedBatchMoves = self.recentAcceptedBatchMoves_;
		const bool savedSkipCheckpointedOrderAfterBatch = self.skipCheckpointedOrderAfterBatch_;

		self.improveIntraOrder(Sout, 25);

		LocalSearchOptions Pmr = opt_;
		Pmr.MR_BATCH_N_TRIALS = std::min(6, opt_.MR_BATCH_N_TRIALS);
		Pmr.MR_BATCH_CRITICAL_ONLY = false;
		Pmr.MR_MOVE_NUM_BATCH_TRIALS = std::min(2, opt_.MR_MOVE_NUM_BATCH_TRIALS);
		Pmr.MR_MOVE_MAX_POS_TRIALS = std::min(2, opt_.MR_MOVE_MAX_POS_TRIALS);

		if (Sout.mksp <= SBase.mksp + opt_.RR2_POLISH_MAX_DEGRADATION) {
			self.improveCoupledMrBatchOrder(Sout, Pmr);
			self.improveCoupledMrOrder(Sout, Pmr);
			self.improveGapWindow(Sout);
			self.improveGapFill(Sout);
			self.improveIntraOrder(Sout, 15);
		}

		// Restore outer-search caches.
		self.seenAcceptedStateHashes_ = std::move(savedSeenAcceptedStateHashes);
		self.seenEvaluatedStateHashes_ = std::move(savedSeenEvaluatedStateHashes);
		self.seenBatchNeighborhoods_ = std::move(savedSeenBatchNeighborhoods);
		self.exhaustedRr2Combos_ = std::move(savedExhaustedRr2Combos);
		self.recentAcceptedBatchMoves_ = std::move(savedRecentAcceptedBatchMoves);
		self.skipCheckpointedOrderAfterBatch_ = savedSkipCheckpointedOrderAfterBatch;

		return { Sout, Sout.mksp };
	}

	std::vector<std::vector<int>> LocalSearch::enumerateAllAssignments(
		const std::vector<VecInt>& capLists) const
	{
		const int K = static_cast<int>(capLists.size());
		if (K == 0) {
			return { {} };
		}

		std::vector<std::vector<int>> assignments(1, std::vector<int>());

		for (int i = 0; i < K; ++i) {
			std::vector<std::vector<int>> next;
			for (const auto& prefix : assignments) {
				for (int r : capLists[i]) {
					auto row = prefix;
					row.push_back(r);
					next.push_back(std::move(row));
				}
			}
			assignments = std::move(next);
		}

		return assignments;
	}

	std::vector<VecInt> LocalSearch::buildBatchPermutations(
		const VecInt& batch,
		int maxPerms) const
	{
		std::vector<VecInt> out;
		if (batch.size() < 2) {
			return out;
		}

		VecInt perm = batch;
		std::sort(perm.begin(), perm.end());

		do {
			if (perm != batch) {
				out.push_back(perm);
			}
		} while (std::next_permutation(perm.begin(), perm.end()));

		if (static_cast<int>(out.size()) > maxPerms) {
			out.resize(maxPerms);
		}
		return out;
	}

	std::vector<VecInt> LocalSearch::applyCoupledBatchOrder(
		const std::vector<VecInt>& ordP_in,
		const MatrixDouble&,
		const VecInt& batch,
		const VecInt& batchPerm,
		int sAnchor) const
	{
		auto ordP_out = ordP_in;

		for (int r = 0; r < static_cast<int>(ordP_out.size()); ++r) {
			VecInt seq = ordP_out[r];
			if (seq.empty()) {
				continue;
			}

			VecInt presentTasks;
			for (int j : batch) {
				if (containsTask(seq, j)) {
					presentTasks.push_back(j);
				}
			}
			if (presentTasks.empty()) {
				continue;
			}

			std::vector<bool> mask(seq.size(), false);
			int countMask = 0;
			for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
				if (std::find(batch.begin(), batch.end(), seq[i]) != batch.end()) {
					mask[i] = true;
					++countMask;
				}
			}

			VecInt repl;
			for (int j : batchPerm) {
				if (std::find(presentTasks.begin(), presentTasks.end(), j) != presentTasks.end()) {
					repl.push_back(j);
				}
			}

			if (static_cast<int>(repl.size()) != countMask) {
				throw std::runtime_error("Coupled MR batch reorder mismatch.");
			}

			int ptr = 0;
			for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
				if (mask[i]) {
					seq[i] = repl[ptr++];
				}
			}

			ordP_out[r] = seq;
		}

		VecInt seqA = ordP_out[sAnchor];
		std::vector<int> idxA;
		for (int i = 0; i < static_cast<int>(seqA.size()); ++i) {
			if (std::find(batch.begin(), batch.end(), seqA[i]) != batch.end()) {
				idxA.push_back(i);
			}
		}

		if (static_cast<int>(idxA.size()) != static_cast<int>(batch.size())) {
			throw std::runtime_error("Anchor robot lost batch tasks unexpectedly.");
		}

		for (int k = 0; k < static_cast<int>(idxA.size()); ++k) {
			seqA[idxA[k]] = batchPerm[k];
		}
		ordP_out[sAnchor] = seqA;

		return ordP_out;
	}

	std::string LocalSearch::hashAssignment(const MatrixDouble& z) const
	{
		std::ostringstream oss;
		const int mm = static_cast<int>(z.size());
		const int nn = mm == 0 ? 0 : static_cast<int>(z[0].size());

		for (int j = 0; j < nn; ++j) {
			unsigned long long val = 0ULL;
			for (int s = 0; s < mm; ++s) {
				if (z[s][j] > 0.5) {
					val |= (1ULL << s);
				}
			}
			oss << val << '|';
		}
		return oss.str();
	}

	void LocalSearch::removeTaskFromSequence(VecInt& seq, int task)
	{
		seq.erase(std::remove(seq.begin(), seq.end(), task), seq.end());
	}

	bool LocalSearch::containsTask(const VecInt& seq, int task)
	{
		return std::find(seq.begin(), seq.end(), task) != seq.end();
	}

	double LocalSearch::quantile(std::vector<double> v, double q)
	{
		if (v.empty()) return 0.0;
		q = std::clamp(q, 0.0, 1.0);
		std::sort(v.begin(), v.end());
		const double pos = q * static_cast<double>(v.size() - 1);
		const int lo = static_cast<int>(std::floor(pos));
		const int hi = static_cast<int>(std::ceil(pos));
		if (lo == hi) return v[lo];
		const double w = pos - static_cast<double>(lo);
		return (1.0 - w) * v[lo] + w * v[hi];
	}


} // namespace mrta