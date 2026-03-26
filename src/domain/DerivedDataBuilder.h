#pragma once
#include "../domain/InstanceData.h"
//#include "../domain/DerivedData.h"
#include "../MTMRTA_admm_solver/src/domain/DerivedData.h"
namespace mrta {

    class DerivedDataBuilder {
    public:
        [[nodiscard]] DerivedData build(const InstanceData& instance) const;

    private:
        [[nodiscard]] MatrixDouble buildTauBar(
            const InstanceData& instance,
            const VecDouble& est0
        ) const;

        [[nodiscard]] VecDouble buildT0(
            const InstanceData& instance,
            const VecDouble& est0
        ) const;
    };

} // namespace mrta#pragma once
