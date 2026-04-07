#pragma once
#include "../domain/CommonTypes.h"
#include "../domain/InstanceData.h"
#include <optional>

namespace mrta {

    struct RepairOptions {
        int  nIters{ 10 };
        bool freezeOrders{ false };
        std::optional<std::vector<VecInt>> ordersPhys0;
        std::optional<std::vector<VecInt>> ordersVirt0;
        std::optional<VecDouble>           virtPinnedStart;
        std::optional<VecDouble>           tFeasHint;  // ← NEW
    };

    struct RepairResult {
        MatrixDouble        tauFeas;
        VecDouble           thetaFeas;
        VecDouble           tFeas;
        std::vector<VecInt> ordersPhys;
        std::vector<VecInt> ordersVirt;
        VecInt              lastPhysNode;
    };

    class ScheduleRepairer {
    public:
        // ----------------------------------------------------------------
        // v1  —  exact translation of repair_schedule_mt_pushforward.m
        //
        // - nIters is ALWAYS overridden to 10 internally (matches MATLAB
        //   line:  nIters = 10;  at the top of the function)
        // - fake_Rpar = ones  →  virtual overlap checks always continue
        //   (virtual tasks sit at EST[jv], nothing pushed)
        // - No push_past_pinned_virtuals logic
        // - No randomised tie-breaking
        // ----------------------------------------------------------------
        RepairResult repairPushforward(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& thetaInit,
            const VecDouble& tInit,
            const RepairOptions& options = {}) const;

        // ----------------------------------------------------------------
        // v2  —  exact translation of repair_schedule_mt_pushforward2.m
        //
        // - nIters is honoured (NOT overridden)
        // - fake_Rpar = Rpar  (real overlap matrix used in Step C)
        // - push_past_pinned_virtuals called in Steps A and B2
        // - pinned virtual times pre-set before Step A
        // - pinned tStart = max(EST, virtPinnedStart) in Step C
        // - RANDOMIZE_TIES = true in reorder step
        // ----------------------------------------------------------------
        RepairResult repairPushforward2(
            const MatrixDouble& zHard,
            const InstanceData& inst,
            const VecDouble& thetaInit,
            const VecDouble& tInit,
            const RepairOptions& options = {}) const;

    private:
        static std::vector<VecInt> normalizeOrders(
            const std::optional<std::vector<VecInt>>& orders, int m);

        static double safeTravel(const MatrixDouble& W, int from, int to);
        static double maxFiniteInColumn(const MatrixDouble& M, int j);

        // v2-only: push t forward past any pinned virtual on robot s
        // that would overlap physical/MR task j (Rpar[jv][j] != 1).
        // Mirrors MATLAB's push_past_pinned_virtuals() in pushforward2.
        static double pushPastPinnedVirtuals(
            int                 s,
            int                 j,
            double              t,
            const MatrixDouble& tauFeas,
            const VecInt& ordVirt,
            const VecDouble& virtPinnedStart,
            const MatrixDouble& svcDur,
            const MatrixDouble& Rpar);
    };

} // namespace mrta