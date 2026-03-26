#include "PrecedenceGraph.h"

#include <stdexcept>
#include <vector>

namespace mrta {

    PrecedenceGraph::PrecedenceGraph(int nTasks, std::vector<PrecedenceEdge> edges)
        : n_(nTasks), edges_(std::move(edges)) {
    }

    VecDouble PrecedenceGraph::pushForward(const VecDouble& tIn, const VecDouble& durTask) const {
        if (static_cast<int>(tIn.size()) != n_ || static_cast<int>(durTask.size()) != n_) {
            throw std::invalid_argument("PrecedenceGraph::pushForward size mismatch.");
        }

        VecDouble tOut = tIn;
        if (edges_.empty()) {
            return tOut;
        }

        std::vector<int> indeg(n_, 0);
        std::vector<std::vector<int>> adj(n_);

        for (const auto& e : edges_) {
            adj[e.from].push_back(e.to);
            ++indeg[e.to];
        }

        std::vector<int> q;
        q.reserve(n_);
        for (int i = 0; i < n_; ++i) {
            if (indeg[i] == 0) {
                q.push_back(i);
            }
        }

        std::size_t head = 0;
        int visited = 0;

        while (head < q.size()) {
            const int u = q[head++];
            ++visited;

            const double uEnd = tOut[u] + durTask[u];
            for (int v : adj[u]) {
                if (tOut[v] < uEnd) {
                    tOut[v] = uEnd;
                }
                --indeg[v];
                if (indeg[v] == 0) {
                    q.push_back(v);
                }
            }
        }

        if (visited != n_) {
            throw std::runtime_error("Precedence graph has a cycle.");
        }

        return tOut;
    }

} // namespace mrta