#pragma once

#include "CommonTypes.h"

namespace mrta {

    struct DerivedData {
        MatrixDouble svcEst;
        VecDouble taskFlex;
        VecDouble scarcityRobot;
        MatrixDouble mrTravel;

        VecDouble EST0;
        MatrixDouble tauBar;
        VecDouble t0;
    };

} // namespace mrta#pragma once
