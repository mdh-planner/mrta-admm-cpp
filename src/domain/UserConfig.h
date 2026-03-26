#pragma once
#pragma once

namespace mrta {

    struct UserConfig {
        int instId{ 1 };
        bool cpEnabled{ false };

        // Optional import-time defaults / global knobs.
        // These are the "user settings" from MATLAB.
        int maxIter{ 400 };
        double rhoCard{ 8.0 };
        double rhoSync{ 8.0 };
        double alphaY{ 0.05 };
        double alphaTau{ 2.0 };

        int shapingPeriod{ 10 };
        double lambdaLoad{ 20.0 };
        double lambdaMr{ 6.0 };

        double dualRelax{ 0.2 };
        double thetaRelax{ 0.5 };

        double tolCard{ 1e-6 };
        double tolSync{ 1e-4 };

        int nRepair{ 60 };

        // MATLAB used "auto" / "fixed". In C++ we normalize to enum.
        bool useFixedEndDepot{ false };
    };

} // namespace mrta