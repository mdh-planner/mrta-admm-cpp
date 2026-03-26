#include "DistributedMrtaApp.h"

#include "../io/InstanceLoader.h"
#include "../solver/ParameterScaler.h"
#include "../solver/DerivedDataBuilder.h"
#include "../solver/AdmmSolver.h"
#include "../solver/AssignmentRounder.h"
#include "../solver/ScheduleRepairer.h"
#include "../solver/LocalSearchEngine.h"
#include "../solver/ScheduleScorer.h"
#include "../reporting/SolutionReporter.h"

#include <iostream>
#include <stdexcept>

namespace mrta {

    DistributedMrtaApp::DistributedMrtaApp(
        InstanceLoader& loader,
        ParameterScaler& parameterScaler,
        DerivedDataBuilder& derivedBuilder,
        AdmmSolver& admmSolver,
        AssignmentRounder& rounder,
        ScheduleRepairer& repairer,
        LocalSearchEngine& localSearch,
        ScheduleScorer& scorer,
        SolutionReporter& reporter
    )
        : loader_(loader),
        parameterScaler_(parameterScaler),
        derivedBuilder_(derivedBuilder),
        admmSolver_(admmSolver),
        rounder_(rounder),
        repairer_(repairer),
        localSearch_(localSearch),
        scorer_(scorer),
        reporter_(reporter) {
    }

    int DistributedMrtaApp::run(const UserConfig& config) {
        InstanceData instance = loader_.load(config);
        normalizeDepotPolicy(instance);

        const SolverParameters params = parameterScaler_.build(instance);
        const DerivedData derived = derivedBuilder_.build(instance);

        AdmmResult admm = admmSolver_.solve(instance, derived, params.admm);

        if (!admm.converged) {
            std::cerr << "Warning: ADMM did not converge. Solution quality may be reduced.\n";
        }

        SolutionState rounded = rounder_.roundAndClean(instance, derived, params, admm);
        SolutionState repaired = repairer_.repairInitial(instance, derived, params, rounded);
        SolutionState improved = localSearch_.improve(instance, derived, params, repaired);
        improved = scorer_.scoreFinal(instance, improved);

        reporter_.printSummary(instance, admm, improved);
        reporter_.printFinalSchedule(instance, improved);
        reporter_.printRobotStatistics(instance, improved);
        reporter_.emitPlots(instance, admm, improved);

        if (config.cpEnabled) {
            reporter_.verifySolution(instance, improved);
        }

        return 0;
    }

    void DistributedMrtaApp::normalizeDepotPolicy(InstanceData& instance) const {
        if (instance.hasFixedDepotPolicy()) {
            if (instance.endDepotFixed.empty()) {
                throw std::runtime_error(
                    "depotPolicy is 'fixed' but no endDepotFixed was provided."
                );
            }
            if (static_cast<int>(instance.endDepotFixed.size()) != instance.m) {
                throw std::runtime_error(
                    "endDepotFixed must contain exactly one end depot per robot."
                );
            }
        }
    }

} // namespace mrta