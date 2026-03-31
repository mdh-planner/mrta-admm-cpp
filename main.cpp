#include "src/domain/UserConfig.h"
#include "src/domain/SolverParameters.h"
#include "src/io/InstanceLoader.h"
#include "src/solver/DerivedDataBuilder.h"
#include "src/solver/ParameterScaler.h"
#include "src/solver/AdmmSolver.h"
#include "src/solver/AssignmentRounder.h"
#include "src/solver/ScheduleRepairer.h"
#include "src/solver/ScheduleScorer.h"
#include "src/solver/LocalSearch.h"

#include <exception>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>


// To test ADMM contribution: set to 1 for random, 0 for real ADMM
#define USE_RANDOM_ADMM 0
#define VERBOSE 0

void printVector(const std::string& label, const mrta::VecDouble& v);
void printVectorInt1Based(const std::string& label, const mrta::VecInt& v);
void printMatrix(const std::string& label, const mrta::MatrixDouble& M);
void printOrders1Based(const std::string& label, const std::vector<mrta::VecInt>& orders);
void printBinaryAssignmentMatrix1Based(const std::string& label, const mrta::MatrixDouble& Z);

void exportScheduleToMatlabScript(
	const std::string& fileName,
	int instId,
	const mrta::MatrixDouble& z_hard,
	const mrta::MatrixDouble& tau,
	const mrta::BoolVec& isVirtual,
	const mrta::BoolVec& isMR,
	const mrta::VecDouble& theta,
	const mrta::VecDouble& durTask,
	const mrta::MatrixDouble& svcPhysSJ,
	const mrta::MatrixDouble& svcVirtSJ,
	const std::vector<mrta::VecInt>& orders_phys,
	const std::vector<mrta::VecInt>& orders_virt,
	const mrta::MatrixDouble& W,
	const mrta::VecInt& endDepot,
	const mrta::VecDouble& Tstart
);

std::string matlabBoolRowVector(const mrta::BoolVec& v);
std::string matlabRowVector(const mrta::VecDouble& v);
std::string matlabRowVectorInt1Based(const mrta::VecInt& v);
std::string matlabMatrix(const mrta::MatrixDouble& M);
std::string matlabCellOrders1Based(const std::vector<mrta::VecInt>& orders);
static void printScheduleLikeCp(
	const mrta::InstanceData& inst,
	const mrta::LocalSearchOutput& out,
	double solveSeconds);

int main() {


	static std::ofstream logFile("run.log");

	// Save original buffer
	static std::streambuf* coutBuf = std::cout.rdbuf();

	// Create tee buffer
	class TeeBuf : public std::streambuf {
	public:
		TeeBuf(std::streambuf* sb1, std::streambuf* sb2) : sb1(sb1), sb2(sb2) {}

	protected:
		virtual int overflow(int c) override {
			if (c == EOF) return !EOF;
			if (sb1->sputc(c) == EOF || sb2->sputc(c) == EOF)
				return EOF;
			return c;
		}

		virtual int sync() override {
			return sb1->pubsync() == 0 && sb2->pubsync() == 0 ? 0 : -1;
		}

	private:
		std::streambuf* sb1;
		std::streambuf* sb2;
	};

	static TeeBuf teeBuf(coutBuf, logFile.rdbuf());
	std::cout.rdbuf(&teeBuf);

	// Optional: also capture std::cerr
	std::cerr.rdbuf(&teeBuf);

	try {
		// Testing Instances
		// Optimal = [1-14,16,30
		// Sub-Optimal = [15,17-29
		mrta::UserConfig config;
		config.instId = 17;
		config.cpEnabled = false;
		config.useFixedEndDepot = false;

		mrta::InstanceLoader loader;
		mrta::ParameterScaler parameterScaler;
		mrta::DerivedDataBuilder derivedBuilder;
		mrta::AdmmSolver admmSolver;
		mrta::ScheduleRepairer repairer;
		mrta::ScheduleScorer scorer;
		mrta::RepairedAssignmentMakespanEvaluator eval(repairer, scorer);
		mrta::AssignmentRounder rounder(eval);

		const mrta::InstanceData instance = loader.load(config);
		const mrta::SolverParameters params = parameterScaler.build(instance);
		const mrta::DerivedData derived = derivedBuilder.build(instance);

		std::cout << "\n=== INSTANCE LOADED ===\n";
		std::cout << "m = " << instance.m << ", n = " << instance.n << '\n';
		std::cout << "depots = " << instance.depotNodes.size() << '\n';

		const auto t_start = std::chrono::steady_clock::now();


#if USE_RANDOM_ADMM
		mrta::AdmmResult admm;
		admm.converged = true;
		admm.theta = mrta::VecDouble(instance.n, 0.0);
		admm.z = mrta::MatrixDouble(instance.m, mrta::VecDouble(instance.n, 0.0));
		admm.histCard = {};
		admm.histSync = {};
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(0.0, 1.0 / instance.m);
		for (int s = 0; s < instance.m; ++s)
			for (int j = 0; j < instance.n; ++j)
				admm.z[s][j] = dist(rng);
#else
		const mrta::AdmmResult admm = admmSolver.run(instance, derived, params);
#endif
		std::cout << "\n=== ADMM ===\n";
		std::cout << "converged = " << (admm.converged ? "true" : "false") << '\n';
		std::cout << "histCard size = " << admm.histCard.size() << '\n';
		std::cout << "histSync size = " << admm.histSync.size() << '\n';
		printVector("ADMM theta", admm.theta);

		const mrta::RoundingResult rounded =
			rounder.roundAndClean(admm.z, instance, derived, admm.theta, derived.t0);

		std::cout << "\n=== ROUNDING ===\n";
		std::cout << "zHard rows = " << rounded.zHard.size() << '\n';
		std::cout << "workH size = " << rounded.workH.size() << '\n';
		printVector("Rounded workH", rounded.workH);
		printBinaryAssignmentMatrix1Based("Rounded zHard (binary by robot x task)", rounded.zHard);

		// ------------------------------------------------------------
		// Baseline repair right after rounding
		// ------------------------------------------------------------
		mrta::RepairOptions repairOpts;
		repairOpts.nIters = params.admm.nRepair;
		repairOpts.freezeOrders = false;

		const mrta::RepairResult repaired = repairer.repairPushforward(
			rounded.zHard,
			instance,
			admm.theta,
			derived.t0,
			repairOpts
		);

		const mrta::ScheduleScore baselineScore = scorer.scoreScheduleExact(
			instance,
			rounded.zHard,
			repaired.tauFeas,
			repaired.ordersPhys,
			repaired.ordersVirt,
			repaired.lastPhysNode
		);

		std::cout << "\n=== BASELINE REPAIRED SCHEDULE ===\n";
		printVector("Baseline repaired theta", repaired.thetaFeas);
		printVectorInt1Based("Baseline last physical nodes (1-based node ids)", repaired.lastPhysNode);
		printOrders1Based("Baseline physical task order per robot", repaired.ordersPhys);
		printOrders1Based("Baseline virtual task order per robot", repaired.ordersVirt);

		if (config.instId == 1) {
			printMatrix("Baseline tauFeas", repaired.tauFeas);
		}

		std::cout << "\n=== BASELINE SCORE ===\n";
		std::cout << "mksp = " << baselineScore.mksp << '\n';

		std::cout << "arrival to depots:\n";
		for (std::size_t s = 0; s < baselineScore.arrival.size(); ++s) {
			std::cout << baselineScore.arrival[s]
				<< (s + 1 < baselineScore.arrival.size() ? ' ' : '\n');
		}

		std::cout << "chosen end depots:\n";
		for (std::size_t s = 0; s < baselineScore.endDepot.size(); ++s) {
			std::cout << baselineScore.endDepot[s] + 1
				<< (s + 1 < baselineScore.endDepot.size() ? ' ' : '\n');
		}

		mrta::LocalSearchOptions lsOpts;

		const auto& LSP = params.localSearch;
		lsOpts.nOuter = LSP.nOuter;
		lsOpts.nInnerOrder = LSP.nInnerOrder;
		lsOpts.nRepairInit = LSP.nRepairInit;
		lsOpts.nRepairFrozen = LSP.nRepairFrozen;
		lsOpts.nRepairReloc = LSP.nRepairReloc;
		lsOpts.MR_BATCH_N_TRIALS = LSP.mrBatchNTrials;
		lsOpts.MR_BATCH_K = LSP.mrBatchK;
		lsOpts.MR_BATCH_MAX_PERMS = LSP.mrBatchMaxPerms;
		lsOpts.MR_MOVE_NUM_BATCH_TRIALS = LSP.mrMoveNumBatchTrials;
		lsOpts.MR_MOVE_MAX_TASKS = LSP.mrMoveMaxTasks;
		lsOpts.MR_MOVE_MAX_POS_TRIALS = LSP.mrMoveMaxPosTrials;
		lsOpts.SR_SWAP_QUANTILE_CUTOFF = LSP.srSwapQuantileCutoff;
		lsOpts.GAP_WINDOW_BACK = LSP.gapWindowBack;
		lsOpts.GAP_WINDOW_FORWARD = LSP.gapWindowForward;
		lsOpts.GAP_WINDOW_MAX_GAPS = LSP.gapWindowMaxGaps;
		lsOpts.MAX_MR_TASKS_PER_OUTER = LSP.maxMrTasksPerOuter;
		lsOpts.MAX_MR_CANDIDATES_PER_J = LSP.maxMrCandidatesPerTask;
		lsOpts.RR2_NUM_TRIALS = LSP.rr2NumTrials;
		lsOpts.RR2_MAX_MR_COMBOS = LSP.rr2MaxMrCombos;
		lsOpts.RR2_NUM_PERTURBATIONS = LSP.rr2NumPerturbations;
		lsOpts.RR2_POLISH_N_INNER = LSP.rr2PolishNInner;
		lsOpts.RR2_EXHAUST_LIMIT = LSP.rr2ExhaustLimit;
		lsOpts.RR2_EXHAUST_TOP_K = LSP.rr2ExhaustTopK;
		lsOpts.POLISH_N_INNER = LSP.rr2PolishNInner;
		lsOpts.timeLimitSeconds = 200.0;  // or whatever limit you want
		lsOpts.RR2_EVAL_MAX_DEGRADATION = LSP.rr2EvalMaxDegradation;
		lsOpts.RR2_POLISH_MAX_DEGRADATION = LSP.rr2PolishMaxDegradation;
		lsOpts.RR2_CACHE_DOMINATED_MAX_NEW = LSP.rr2CacheDominatedMaxNew;

		mrta::LocalSearch localSearch(instance, lsOpts, repairer, scorer);

		const mrta::LocalSearchOutput lsOut =
			localSearch.run(
				rounded.zHard,
				admm.theta,
				derived.t0
			);

		const mrta::ScheduleScore finalScore =
			scorer.scoreScheduleExact(
				instance,
				lsOut.z_best,
				lsOut.tau_best,
				lsOut.orders_phys_best,
				lsOut.orders_virt_best,
				lsOut.lastPhysNode_best
			);

		const auto t_end = std::chrono::steady_clock::now();
		const double elapsed_sec =
			std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
		std::cout << std::endl;
		std::cout << "Elapsed time: " << elapsed_sec << " s\n";


		exportScheduleToMatlabScript(
			"cpp_ls_export_inst" + std::to_string(config.instId) + ".m",
			config.instId,
			lsOut.z_best,
			lsOut.tau_best,
			instance.isVirtual,
			instance.isMR,
			lsOut.theta_best,
			instance.durTask,
			instance.svcPhysSJ,
			instance.svcVirtSJ,
			lsOut.orders_phys_best,
			lsOut.orders_virt_best,
			instance.W,
			finalScore.endDepot,
			derived.t0
		);

		std::cout << "\n=== LOCAL SEARCH RESULT ===\n";
		printBinaryAssignmentMatrix1Based("LS z_best (binary by robot x task)", lsOut.z_best);
		printVector("LS theta_best", lsOut.theta_best);
		printVectorInt1Based("LS last physical nodes (1-based node ids)", lsOut.lastPhysNode_best);
		printOrders1Based("LS physical task order per robot", lsOut.orders_phys_best);
		printOrders1Based("LS virtual task order per robot", lsOut.orders_virt_best);

		if (config.instId == 1) {
			printMatrix("LS tau_best", lsOut.tau_best);
		}

		std::cout << "\n=== FINAL SCORE AFTER LOCAL SEARCH ===\n";
		std::cout << "mksp = " << finalScore.mksp << '\n';

		std::cout << "arrival to depots:\n";
		for (std::size_t s = 0; s < finalScore.arrival.size(); ++s) {
			std::cout << finalScore.arrival[s]
				<< (s + 1 < finalScore.arrival.size() ? ' ' : '\n');
		}

		std::cout << "chosen end depots:\n";
		for (std::size_t s = 0; s < finalScore.endDepot.size(); ++s) {
			std::cout << finalScore.endDepot[s] + 1
				<< (s + 1 < finalScore.endDepot.size() ? ' ' : '\n');
		}

		std::cout << "\n=== IMPROVEMENT SUMMARY ===\n";
		std::cout << "baseline mksp = " << baselineScore.mksp << '\n';
		std::cout << "ls mksp       = " << finalScore.mksp << '\n';
		std::cout << "delta         = " << (baselineScore.mksp - finalScore.mksp) << '\n';

		std::cout << "\nAssigned tasks per robot:\n";
		for (int s = 0; s < instance.m; ++s) {
			int cnt = 0;
			for (int j = 0; j < instance.n; ++j) {
				if (lsOut.z_best[s][j] > 0.5) {
					++cnt;
				}
			}
			std::cout << cnt << (s + 1 < instance.m ? ' ' : '\n');
		}

		printScheduleLikeCp(instance, lsOut, /*solveSeconds=*/ elapsed_sec);

		std::cout << "\nDone.\n";
		return 0;
	}
	catch (const std::exception& ex) {
		std::cerr << "Fatal error: " << ex.what() << '\n';
		return 1;
	}
}

void printVector(const std::string& label, const mrta::VecDouble& v) {
	std::cout << label << ":\n";
	for (std::size_t i = 0; i < v.size(); ++i) {
		std::cout << v[i] << (i + 1 < v.size() ? ' ' : '\n');
	}
}

void printVectorInt1Based(const std::string& label, const mrta::VecInt& v) {
	std::cout << label << ":\n";
	for (std::size_t i = 0; i < v.size(); ++i) {
		std::cout << (v[i] + 1) << (i + 1 < v.size() ? ' ' : '\n');
	}
}

void printMatrix(const std::string& label, const mrta::MatrixDouble& M) {
	std::cout << label << ":\n";
	for (const auto& row : M) {
		for (std::size_t j = 0; j < row.size(); ++j) {
			std::cout << row[j] << (j + 1 < row.size() ? ' ' : '\n');
		}
	}
}

void printOrders1Based(const std::string& label, const std::vector<mrta::VecInt>& orders) {
	std::cout << label << ":\n";
	for (std::size_t s = 0; s < orders.size(); ++s) {
		std::cout << "Robot " << (s + 1) << ": ";
		if (orders[s].empty()) {
			std::cout << "[]\n";
			continue;
		}

		std::cout << "[";
		for (std::size_t k = 0; k < orders[s].size(); ++k) {
			std::cout << (orders[s][k] + 1);
			if (k + 1 < orders[s].size()) {
				std::cout << ' ';
			}
		}
		std::cout << "]\n";
	}
}

void printBinaryAssignmentMatrix1Based(const std::string& label, const mrta::MatrixDouble& Z) {
	std::cout << label << ":\n";
	for (std::size_t s = 0; s < Z.size(); ++s) {
		std::cout << "Robot " << (s + 1) << ": ";
		for (std::size_t j = 0; j < Z[s].size(); ++j) {
			std::cout << (Z[s][j] > 0.5 ? 1 : 0) << (j + 1 < Z[s].size() ? ' ' : '\n');
		}
	}
}

std::string matlabRowVector(const mrta::VecDouble& v) {
	std::ostringstream oss;
	oss << "[";
	for (std::size_t i = 0; i < v.size(); ++i) {
		oss << std::setprecision(17) << v[i];
		if (i + 1 < v.size()) {
			oss << " ";
		}
	}
	oss << "]";
	return oss.str();
}

std::string matlabRowVectorInt1Based(const mrta::VecInt& v) {
	std::ostringstream oss;
	oss << "[";
	for (std::size_t i = 0; i < v.size(); ++i) {
		oss << (v[i] + 1);
		if (i + 1 < v.size()) {
			oss << " ";
		}
	}
	oss << "]";
	return oss.str();
}

std::string matlabMatrix(const mrta::MatrixDouble& M) {
	std::ostringstream oss;
	oss << "[";
	for (std::size_t i = 0; i < M.size(); ++i) {
		for (std::size_t j = 0; j < M[i].size(); ++j) {
			oss << std::setprecision(17) << M[i][j];
			if (j + 1 < M[i].size()) {
				oss << " ";
			}
		}
		if (i + 1 < M.size()) {
			oss << "; ";
		}
	}
	oss << "]";
	return oss.str();
}

std::string matlabCellOrders1Based(const std::vector<mrta::VecInt>& orders) {
	std::ostringstream oss;
	oss << "{";
	for (std::size_t s = 0; s < orders.size(); ++s) {
		oss << "[";
		for (std::size_t k = 0; k < orders[s].size(); ++k) {
			oss << (orders[s][k] + 1);
			if (k + 1 < orders[s].size()) {
				oss << " ";
			}
		}
		oss << "]";
		if (s + 1 < orders.size()) {
			oss << ", ";
		}
	}
	oss << "}";
	return oss.str();
}

void exportScheduleToMatlabScript(
	const std::string& fileName,
	int instId,
	const mrta::MatrixDouble& z_hard,
	const mrta::MatrixDouble& tau,
	const mrta::BoolVec& isVirtual,
	const mrta::BoolVec& isMR,
	const mrta::VecDouble& theta,
	const mrta::VecDouble& durTask,
	const mrta::MatrixDouble& svcPhysSJ,
	const mrta::MatrixDouble& svcVirtSJ,
	const std::vector<mrta::VecInt>& orders_phys,
	const std::vector<mrta::VecInt>& orders_virt,
	const mrta::MatrixDouble& W,
	const mrta::VecInt& endDepot,
	const mrta::VecDouble& Tstart
) {
	std::ofstream out(fileName);
	if (!out) {
		throw std::runtime_error("Failed to open MATLAB export file: " + fileName);
	}

	out << "% Auto-generated from C++\n";
	out << "% Instance " << instId << "\n\n";

	out << "instId = " << instId << ";\n";
	out << "z_hard = " << matlabMatrix(z_hard) << ";\n";
	out << "tau = " << matlabMatrix(tau) << ";\n";
	out << "isVirtual = " << matlabBoolRowVector(isVirtual) << ";\n";
	out << "isMR = " << matlabBoolRowVector(isMR) << ";\n";
	out << "theta = " << matlabRowVector(theta) << ";\n";
	out << "durTask = " << matlabRowVector(durTask) << ";\n";
	out << "svcPhysSJ = " << matlabMatrix(svcPhysSJ) << ";\n";
	out << "svcVirtSJ = " << matlabMatrix(svcVirtSJ) << ";\n";
	out << "orders_phys = " << matlabCellOrders1Based(orders_phys) << ";\n";
	out << "orders_virt = " << matlabCellOrders1Based(orders_virt) << ";\n";
	out << "W = " << matlabMatrix(W) << ";\n";
	out << "endDepot = " << matlabRowVectorInt1Based(endDepot) << ";\n";
	out << "Tstart = " << matlabRowVector(Tstart) << ";\n\n";

	out << "plot_gantt_mt_per_robot( ...\n";
	out << "    z_hard, tau, isVirtual, isMR, theta, durTask, ...\n";
	out << "    svcPhysSJ, svcVirtSJ, orders_phys, orders_virt, ...\n";
	out << "    W, endDepot, Tstart, instId);\n";
}

std::string matlabBoolRowVector(const mrta::BoolVec& v) {
	std::ostringstream oss;
	oss << "[";
	for (std::size_t i = 0; i < v.size(); ++i) {
		oss << (v[i] ? 1 : 0);
		if (i + 1 < v.size()) {
			oss << " ";
		}
	}
	oss << "]";
	return oss.str();
}


static void printScheduleLikeCp(
	const mrta::InstanceData& inst,
	const mrta::LocalSearchOutput& out,
	double solveSeconds)
{
	std::cout << "Makespan        : " << static_cast<int>(std::round(out.mksp_best)) << "\n";
	std::cout << "Time spent in solve: " << solveSeconds << " seconds.\n";

	struct TaskRow {
		int taskId;     // 0-based
		int robot;      // 0-based
		double start;
		double end;
		
	};

	std::vector<TaskRow> rows;


	auto currentTaskDuration = [&](int s, int j) -> double {
		if (inst.isMR[j]) {
			double best = 0.0;
			bool found = false;
			for (int r = 0; r < inst.m; ++r) {
				const double v = inst.svcPhysSJ[r][j];
				if (std::isfinite(v) && v >= 0.0) {
					best = found ? std::max(best, v) : v;
					found = true;
				}
			}
			return found ? best : 0.0;
		}

		const double raw = inst.isVirtual[j] ? inst.svcVirtSJ[s][j] : inst.svcPhysSJ[s][j];
		return (std::isfinite(raw) && raw >= 0.0) ? raw : 0.0;
		};

	auto thetaTaskStart = [&](int j) -> double {
		if (j < 0 || j >= static_cast<int>(out.theta_best.size())) {
			return 0.0;
		}
		return std::isfinite(out.theta_best[j]) ? out.theta_best[j] : 0.0;
		};

	// Physical tasks
	for (int r = 0; r < inst.m; ++r) {
		for (int j : out.orders_phys_best[r]) {
			double st = inst.isMR[j] ? thetaTaskStart(j) : out.tau_best[r][j];
			double en = st + currentTaskDuration(r, j);

		
			rows.push_back({ j, r, st, en });
		}
	}

	// Virtual tasks
	for (int r = 0; r < inst.m; ++r) {
		for (int j : out.orders_virt_best[r]) {
			double st = out.tau_best[r][j];
			double en = st + currentTaskDuration(r, j);

			
			rows.push_back({ j, r, st, en});
		}
	}

	std::sort(rows.begin(), rows.end(),
		[](const TaskRow& a, const TaskRow& b) {
			if (a.taskId != b.taskId) return a.taskId < b.taskId;   // Task 0,1,2,...
			if (a.robot != b.robot) return a.robot < b.robot;       // then by robot
			if (std::abs(a.start - b.start) > 1e-9) return a.start < b.start;
			return a.end < b.end;
		});

	for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
		const auto& row = rows[i];
		std::cout
			<< "Task " << row.taskId
			<< " on robot " << row.robot
			<< ": start time: " << static_cast<int>(std::round(row.start))
			<< " | end time: " << static_cast<int>(std::round(row.end))
			<< "\n";
	}

	std::cout << "\n";
}