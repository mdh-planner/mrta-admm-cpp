#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/DerivedData.h"
#include "../domain/InstanceData.h"

namespace mrta {

    class ScheduleRepairer;
    class ScheduleScorer;

    struct RoundingResult {
        MatrixDouble zHard;
        VecDouble workH;
    };

    class IAssignmentMakespanEvaluator {
    public:
        virtual ~IAssignmentMakespanEvaluator() = default;

        virtual double evaluate(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& theta0,
            const VecDouble& t0) const = 0;
    };

    class NoOpAssignmentMakespanEvaluator final : public IAssignmentMakespanEvaluator {
    public:
        double evaluate(
            const MatrixDouble&,
            const InstanceData&,
            const VecDouble&,
            const VecDouble&) const override
        {
            return 0.0;
        }
    };

    class RepairedAssignmentMakespanEvaluator final : public IAssignmentMakespanEvaluator {
    public:
        RepairedAssignmentMakespanEvaluator(
            const ScheduleRepairer& repairer,
            const ScheduleScorer& scorer);

        double evaluate(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& theta0,
            const VecDouble& t0) const override;

    private:
        const ScheduleRepairer& repairer_;
        const ScheduleScorer& scorer_;
    };

    class AssignmentRounder {
    public:
        explicit AssignmentRounder(const IAssignmentMakespanEvaluator& evaluator);

        RoundingResult roundAndClean(
            const MatrixDouble& z,
            const InstanceData& inst,
            const DerivedData& data,
            const VecDouble& theta,
            const VecDouble& t0) const;

        MatrixDouble rebalanceSrByMakespan(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& theta,
            const VecDouble& t0,
            int nPasses,
            int maxTasksPerHeavyRobot,
            int maxDstPerTask) const;

        MatrixDouble rebalanceMrParticipantsByMakespan(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& theta,
            const VecDouble& t0,
            int nPasses,
            int maxMrTasksPerPass,
            int maxReplacementPairs) const;

        static MatrixDouble buildLocalServiceEstimate(
            const InstanceData& inst);

        const IAssignmentMakespanEvaluator& evaluator_;
    };

} // namespace mrta