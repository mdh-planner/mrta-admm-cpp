#pragma once

#include "../domain/InstanceData.h"
#include "../domain/UserConfig.h"

#include <string>

namespace mrta {

    class InstanceLoader {
    public:
        [[nodiscard]] InstanceData load(const UserConfig& config) const;

    private:
        [[nodiscard]] std::string buildInstanceDirectory() const;
        [[nodiscard]] InstanceData loadCaseInstance(const std::string& instanceDir, int instId) const;

        void validateInstance(const InstanceData& instance) const;
        void buildServiceMatrices(InstanceData& instance) const;
        void buildAssignmentCostMatrix(InstanceData& instance) const;
        void buildDepotNodes(InstanceData& instance) const;
    };

} // namespace mrta