#pragma once

namespace mrta {

    struct AdmmParams {
        double rhoCard{};
        double rhoSync{};
        double alphaY{};
        double alphaTau{};
        int shapingPeriod{};
        double lambdaLoad{};
        double lambdaMr{};
        double dualRelax{};
        double thetaRelax{};
        double tolCard{};
        double tolSync{};
        int maxIter{};
        int nRepair{};
        double fallbackDualRelax{};
        double fallbackThetaRelax{};
        double fallbackRhoSync{};
        double fallbackTolSync{};
        int fallbackMaxIter{};
    };

    struct LocalSearchParams {
        int nOuter{};
        int nInnerOrder{};

        int mrBatchNTrials{};
        int mrMoveNumBatchTrials{};
        int mrBatchK{};
        int mrBatchMaxPerms{};
        int mrMoveMaxTasks{};
        int mrMoveMaxPosTrials{};

        double srSwapQuantileCutoff{};

        int gapWindowBack{};
        int gapWindowForward{};
        int gapWindowMaxGaps{};

        int maxMrTasksPerOuter{};
        int maxMrCandidatesPerTask{};

        int maxTasksPerRobot{};

        int rrNumTrials{};
        int rrMaxMrTasks{};
        int rrMaxMrSwapsPerTask{};
        int rrMaxMrCombos{};
        int rrMaxSrStrip{};
        int rrMaxPerms{};
        int rrMaxSrPos{};
        int nRepairReloc{};
        int rrDiveBacktrack{};
        int rrNumRandomSamples{};

        int xoptMaxSegLen{};
        int xoptMaxPairs{};
        double xoptNearCritFrac{};

        int rr2NumTrials{};
        int rr2MaxMrCombos{};
        int rr2NumPerturbations{};
        int rr2PolishNInner{};
        int rr2ExhaustLimit{};
        int rr2ExhaustTopK{};

        int nRepairInit{};
        int nRepairFrozen{};
    };

    struct SolverParameters {
        AdmmParams admm;
        LocalSearchParams localSearch;
    };

} // namespace mrta