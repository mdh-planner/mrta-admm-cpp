#pragma once

#include "../domain/CommonTypes.h"
#include "../domain/DerivedData.h"
#include "../domain/InstanceData.h"
#include "../domain/SolverParameters.h"

namespace mrta {

    struct AdmmResult {
        MatrixDouble z;
        VecDouble theta;
        bool converged{ false };
        VecDouble histCard;
        VecDouble histSync;
    };

    class AdmmSolver {
    public:
        AdmmResult run(
            const InstanceData& inst,
            const DerivedData& data,
            const SolverParameters& params) const;

    private:
        AdmmResult runLoop(
            const InstanceData& inst,
            const DerivedData& data,
            const SolverParameters& params,
            int maxIter,
            double rhoSync,
            double dualRelax,
            double thetaRelax,
            double tolSync) const;

        VecDouble projectBoxedSimplex(
            const VecDouble& v,
            double k,
            const VecDouble& lo,
            const VecDouble& hi) const;
    };

} // namespace mrta