#pragma once

#include "CommonTypes.h"

#include <string>
#include <vector>

namespace mrta {

    struct InstanceData {
        int m{};
        int n{};

        VecInt k;
        VecInt reqEquip;

        MatrixDouble cap;
        BoolVec isVirtual;
        BoolVec isMR;
        std::vector<PrecedenceEdge> predPairs;

        MatrixDouble Rpar;
        VecDouble durTask;

        MatrixDouble W;
        MatrixDouble Tstart;
        MatrixDouble TT;

        VecInt depotNodes;
        DepotPolicy depotPolicy{ DepotPolicy::Free };
        VecInt endDepotFixed;

        MatrixDouble svcPhysSJ;
        MatrixDouble svcVirtSJ;
        MatrixDouble cY;
        MatrixDouble mrTravel;

        int instId{};
        std::string instDir;

        [[nodiscard]] bool hasFixedDepotPolicy() const noexcept {
            return depotPolicy == DepotPolicy::Fixed;
        }
    };

} // namespace mrta