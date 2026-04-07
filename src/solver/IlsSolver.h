#pragma once

#include "../domain/InstanceData.h"
#include "LocalSearch.h"
#include "OperatorStats.h"
#include "ScheduleRepairer.h"
#include "ScheduleScorer.h"
#include "StochasticRounder.h"

#include <limits>
#include <random>
#include <vector>

namespace mrta {

    // ---------------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------------
    struct IlsOptions {
        // --- ILS outer loop ---
        int    maxIter = 100;
        int    restartAfter = 20;
        double acceptTemp = 0.0;

        // --- Perturbation ---
        int    kPerturbMin = 2;
        int    kPerturbMax = 4;

        // --- Multi-start ---
        int    nSeeds = 1;

        // --- Stats ---
        // true  = accumulate OperatorStats across all LS runs and print at end
        // false = no overhead (default for production runs)
        bool   collectStats = false;

        // --- Local search passed through per iteration ---
        LocalSearchOptions lsOpts;
    };

    // Mirrors LocalSearchOutput field names so main.cpp needs no changes.
    struct IlsOutput {
        MatrixDouble        z_best;
        MatrixDouble        tau_best;
        VecDouble           theta_best;
        VecDouble           t_best;
        std::vector<VecInt> orders_phys_best;
        std::vector<VecInt> orders_virt_best;
        VecInt              endDepot_best;
        VecInt              lastPhysNode_best;
        double              mksp_best = 0.0;
        int                 itersRun = 0;
        int                 improvements = 0;
        int                 seedUsed = -1;
    };

    // ---------------------------------------------------------------------------
    // IlsSolver
    // ---------------------------------------------------------------------------
    class IlsSolver {
    public:
        IlsSolver(const InstanceData& inst,
            const IlsOptions& opts,
            const ScheduleRepairer& repairer,
            const ScheduleScorer& scorer);

        IlsOutput run(const MatrixDouble& z0,
            const VecDouble& theta0,
            const VecDouble& t0);

        IlsOutput runMultiStart(const MatrixDouble& zFrac,
            const DerivedData& derived,
            const VecDouble& theta0,
            const VecDouble& t0,
            const StochasticRoundingOptions& srOpts);

        // Access accumulated stats after run()/runMultiStart() completes.
        // Only populated when opts_.collectStats == true.
        const OperatorStats& stats() const { return stats_; }

    private:
        MatrixDouble perturb(const MatrixDouble& z, int k) const;
        static IlsOutput lsOutputToIlsOutput(const LocalSearchOutput& lso);
        MatrixDouble strongPerturb(const MatrixDouble& z) const;
        // Creates a LocalSearch wired to stats_ (if collectStats is on)
        // or with stats=nullptr (no overhead).
        LocalSearch makeLS() const;

        const InstanceData& inst_;
        IlsOptions              opts_;
        const ScheduleRepairer& repairer_;
        const ScheduleScorer& scorer_;
        mutable std::mt19937    rng_;
        mutable OperatorStats   stats_;   // accumulates across all LS runs
    };

} // namespace mrta