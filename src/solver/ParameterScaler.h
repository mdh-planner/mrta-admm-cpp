#pragma once

namespace mrta {

    struct InstanceData;
    struct SolverParameters;

    class ParameterScaler {
    public:
        SolverParameters build(const InstanceData& instance) const;
    };

} // namespace mrta