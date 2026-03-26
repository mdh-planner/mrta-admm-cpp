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



    void printVector(const std::string& label, const mrta::VecDouble& v);
    void printVectorInt1Based(const std::string& label, const mrta::VecInt& v);
    void printMatrix(const std::string& label, const mrta::MatrixDouble& M);
    void printOrders1Based(const std::string& label, const std::vector<mrta::VecInt>& orders);
    void printBinaryAssignmentMatrix1Based(const std::string& label, const mrta::MatrixDouble& Z);
       

int main() {
    try {
        // Testing Instances
        // Optimal = [1-4,7-10,12
        // Sub-Optimal = [5,6,11,13,14,15
        mrta::UserConfig config;
        config.instId = 33;
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
        const mrta::AdmmResult admm = admmSolver.run(instance, derived, params);

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
