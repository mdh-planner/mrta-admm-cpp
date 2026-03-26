#pragma once

#include "../domain/InstanceData.h"
#include "../domain/SolverParameters.h"
#include "../domain/DerivedData.h"
#include "../domain/SolutionState.h"
#include "../domain/UserConfig.h"


namespace mrta {

    class InstanceLoader;
    class ParameterScaler;
    class AdmmSolver;
    class AssignmentRounder;
    class ScheduleRepairer;
    class LocalSearchEngine;
    class ScheduleScorer;
    class SolutionReporter;
    class DerivedDataBuilder;

    class DistributedMrtaApp {
    public:
        DistributedMrtaApp(
            InstanceLoader& loader,
            ParameterScaler& parameterScaler,
            DerivedDataBuilder& derivedBuilder,
            AdmmSolver& admmSolver,
            AssignmentRounder& rounder,
            ScheduleRepairer& repairer,
            LocalSearchEngine& localSearch,
            ScheduleScorer& scorer,
            SolutionReporter& reporter
        );

        int run(const UserConfig& config);

    private:
        void normalizeDepotPolicy(InstanceData& instance) const;

        InstanceLoader& loader_;
        ParameterScaler& parameterScaler_;
        DerivedDataBuilder& derivedBuilder_;
        AdmmSolver& admmSolver_;
        AssignmentRounder& rounder_;
        ScheduleRepairer& repairer_;
        LocalSearchEngine& localSearch_;
        ScheduleScorer& scorer_;
        SolutionReporter& reporter_;
    };

} // namespace mrta#pragma once
