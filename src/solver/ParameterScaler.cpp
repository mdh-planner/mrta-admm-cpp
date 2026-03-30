#include "ParameterScaler.h"

#include "../domain/InstanceData.h"
#include "../domain/SolverParameters.h"

#include <algorithm>
#include <cmath>

namespace mrta {

    SolverParameters ParameterScaler::build(const InstanceData& instance) const {
        SolverParameters out{};

        int nMR = 0;
        int nSR = 0;
        int nVirt = 0;


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

        const double scale = std::sqrt(static_cast<double>(instance.m * instance.n) / 48.0);

        (void)nVirt;

        const double taskPerRobot =
            static_cast<double>(instance.n) / std::max(instance.m, 1);

        auto& AP = out.admm;
        AP.rhoCard = 8.0;
        AP.rhoSync = 8.0 / std::sqrt(static_cast<double>(instance.m));
        AP.alphaY = 0.05;
        AP.alphaTau = 2.0;
        AP.shapingPeriod = 10;
        AP.lambdaLoad = 15.0 * taskPerRobot / 5.0;
        AP.lambdaMr = 6.0;
        AP.dualRelax = 0.2;
        AP.thetaRelax = 0.5;
        AP.tolCard = 1e-6;
        AP.tolSync = 1e-4;
        AP.maxIter = 200 + 50 * instance.m;
        AP.nRepair = std::max(30, 10 * instance.m);
        AP.fallbackDualRelax = 0.08;
        AP.fallbackThetaRelax = 0.25;
        AP.fallbackRhoSync = AP.rhoSync * 0.5;
        AP.fallbackTolSync = 1e-3;
        AP.fallbackMaxIter = AP.maxIter + 200;

        auto& LSP = out.localSearch;
        LSP.nOuter = 20 + 5 * instance.m;
        LSP.nInnerOrder = std::max(60, static_cast<int>(std::lround(60.0 * instance.n / 12.0)));

        LSP.mrBatchNTrials = std::max(8, 3 * nMR);
        LSP.mrMoveNumBatchTrials = std::max(5, 2 * nMR);
        
        // Fix — allow k up to nMR for small instances:
        LSP.mrBatchK = (nMR <= 10)
            ? nMR
            : std::min(4, std::max(2, static_cast<int>(std::lround(
                static_cast<double>(nMR) / 3.0))));
        // Fix:
        LSP.mrBatchMaxPerms = (nMR <= 8)
            ? 120                    // covers 5! = 120 for full sequence attempts
            : std::min(24, std::max(12, 4 * nMR)); 
        LSP.mrMoveMaxTasks = std::max(10, nMR);
        LSP.mrMoveMaxPosTrials = std::max(4, static_cast<int>(std::lround(static_cast<double>(instance.m) / 2.0)));

        LSP.srSwapQuantileCutoff = std::min(0.75, 0.3 + 0.05 * nSR);

        LSP.gapWindowBack = std::min(4, std::max(2, static_cast<int>(std::lround(static_cast<double>(instance.n) / (2.0 * instance.m)))));
        LSP.gapWindowForward = std::min(3, std::max(1, static_cast<int>(std::lround(static_cast<double>(instance.n) / (3.0 * instance.m)))));
        LSP.gapWindowMaxGaps = std::max(5, instance.m + 2);

        LSP.maxMrTasksPerOuter = std::max(20, nMR * 2);
        LSP.maxMrCandidatesPerTask = std::max(10, instance.m * 2);

        LSP.maxTasksPerRobot = std::max(30, instance.n);

        LSP.rrNumTrials = std::max(8, static_cast<int>(std::lround(1.5 * instance.n)));
        LSP.rrMaxMrTasks = std::min(4, nMR);
        LSP.rrMaxMrSwapsPerTask = std::min(4, instance.m);
        LSP.rrMaxMrCombos = 12;
        LSP.rrMaxSrStrip = std::min(10, nSR + nVirt);
        LSP.rrMaxPerms = 24;
        LSP.rrMaxSrPos = std::min(6, instance.n);
        LSP.nRepairReloc = 10;
        LSP.rrDiveBacktrack = std::min(3, nSR);
        LSP.rrNumRandomSamples = std::min(
            200,
            std::max(50, static_cast<int>(std::lround(std::pow(3.0, nSR) / 5.0)))
        );

        LSP.xoptMaxSegLen = 3;
        LSP.xoptMaxPairs = std::min(20, instance.m * (instance.m - 1));
        LSP.xoptNearCritFrac = 0.05;

        LSP.rr2NumTrials = 3;
        LSP.rr2MaxMrCombos = 10;
        LSP.rr2NumPerturbations = 3;
        //LSP.rr2PolishNInner = 10;
        //LSP.rr2ExhaustLimit = 500; //afects performance a lot
        //LSP.rr2ExhaustTopK = 15;

       

        LSP.rr2PolishNInner = std::max(3, std::min(20, static_cast<int>(std::round(10.0 * scale))));
        LSP.rr2ExhaustLimit = std::max(100, std::min(5000, static_cast<int>(std::round(500.0 * scale))));
        LSP.rr2ExhaustTopK = std::max(5, std::min(50, static_cast<int>(std::round(15.0 * scale))));

        LSP.nRepairInit = AP.nRepair;
        LSP.nRepairFrozen = AP.nRepair;

        return out;
    }

} // namespace mrta