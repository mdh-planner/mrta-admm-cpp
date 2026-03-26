#pragma once

#include "../domain/CommonTypes.h"

namespace mrta {

    class PrecedenceGraph {
    public:
        PrecedenceGraph(int nTasks, std::vector<PrecedenceEdge> edges);

        VecDouble pushForward(const VecDouble& tIn, const VecDouble& durTask) const;

    private:
        int n_{};
        std::vector<PrecedenceEdge> edges_;
    };

} // namespace mrta