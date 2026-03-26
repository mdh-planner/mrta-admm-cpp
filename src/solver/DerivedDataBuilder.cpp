#include "DerivedDataBuilder.h"
#include "PrecedenceGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mrta {

    namespace {

        double meanOfFiniteMatrix(const MatrixDouble& M) {
            double sum = 0.0;
            std::size_t count = 0;

            for (const auto& row : M) {
                for (double v : row) {
                    if (std::isfinite(v)) {
                        sum += v;
                        ++count;
                    }
                }
            }

            return count == 0 ? 0.0 : sum / static_cast<double>(count);
        }

        MatrixDouble buildServiceEstimateMatrix(
            const MatrixDouble& cap,
            const BoolVec& isMR,
            const BoolVec& isVirtual,
            const MatrixDouble& svcPhysSJ,
            const MatrixDouble& svcVirtSJ
        ) {
            const int m = static_cast<int>(cap.size());
            const int n = m == 0 ? 0 : static_cast<int>(cap.front().size());

            MatrixDouble svcEst(m, VecDouble(n, std::numeric_limits<double>::infinity()));

            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    if (cap[s][j] < 0.5) {
                        svcEst[s][j] = std::numeric_limits<double>::infinity();
                        continue;
                    }

                    if (isVirtual[j]) {
                        double d = svcVirtSJ[s][j];
                        if (!std::isfinite(d) || d < 0.0) {
                            d = 0.0;
                        }
                        svcEst[s][j] = d;
                    }
                    else if (isMR[j]) {
                        double best = 0.0;
                        bool found = false;
                        for (int r = 0; r < m; ++r) {
                            const double d = svcPhysSJ[r][j];
                            if (std::isfinite(d) && d >= 0.0) {
                                best = found ? std::max(best, d) : d;
                                found = true;
                            }
                        }
                        svcEst[s][j] = found ? best : 0.0;
                    }
                    else {
                        double d = svcPhysSJ[s][j];
                        if (!std::isfinite(d) || d < 0.0) {
                            d = 0.0;
                        }
                        svcEst[s][j] = d;
                    }
                }
            }

            return svcEst;
        }

        VecDouble buildTaskFlex(const MatrixDouble& cap) {
            const int m = static_cast<int>(cap.size());
            const int n = m == 0 ? 0 : static_cast<int>(cap.front().size());

            VecDouble out(n, 0.0);
            for (int j = 0; j < n; ++j) {
                int count = 0;
                for (int s = 0; s < m; ++s) {
                    if (cap[s][j] > 0.5) {
                        ++count;
                    }
                }
                out[j] = static_cast<double>(count);
            }
            return out;
        }

        VecDouble buildScarcityRobot(const MatrixDouble& cap) {
            const int m = static_cast<int>(cap.size());
            const int n = m == 0 ? 0 : static_cast<int>(cap.front().size());

            VecDouble out(m, 1.0);
            for (int s = 0; s < m; ++s) {
                int count = 0;
                for (int j = 0; j < n; ++j) {
                    if (cap[s][j] > 0.5) {
                        ++count;
                    }
                }
                out[s] = 1.0 / static_cast<double>(std::max(count, 1));
            }
            return out;
        }

        MatrixDouble buildMrTravel(const MatrixDouble& Tstart) {
            MatrixDouble out = Tstart;
            const double denom = std::max(1e-9, meanOfFiniteMatrix(Tstart));
            for (auto& row : out) {
                for (double& v : row) {
                    v /= denom;
                }
            }
            return out;
        }

    } // namespace

    DerivedData DerivedDataBuilder::build(const InstanceData& instance) const {
        DerivedData out{};

        out.svcEst = buildServiceEstimateMatrix(
            instance.cap,
            instance.isMR,
            instance.isVirtual,
            instance.svcPhysSJ,
            instance.svcVirtSJ
        );

        out.taskFlex = buildTaskFlex(instance.cap);
        out.scarcityRobot = buildScarcityRobot(instance.cap);
        out.mrTravel = buildMrTravel(instance.Tstart);

        PrecedenceGraph graph(instance.n, instance.predPairs);

        out.EST0 = graph.pushForward(VecDouble(instance.n, 0.0), instance.durTask);
        out.tauBar = buildTauBar(instance, out.EST0);
        out.t0 = buildT0(instance, out.EST0);
        out.t0 = graph.pushForward(out.t0, instance.durTask);

        return out;
    }

    MatrixDouble DerivedDataBuilder::buildTauBar(
        const InstanceData& instance,
        const VecDouble& est0
    ) const {
        MatrixDouble tauBar(instance.m, VecDouble(instance.n, 0.0));

        for (int s = 0; s < instance.m; ++s) {
            for (int j = 0; j < instance.n; ++j) {
                tauBar[s][j] = est0[j] + 0.2 * instance.W[s][instance.m + j];
            }
        }

        return tauBar;
    }

    VecDouble DerivedDataBuilder::buildT0(
        const InstanceData& instance,
        const VecDouble& est0
    ) const {
        VecDouble t0 = est0;

        for (int j = 0; j < instance.n; ++j) {
            double best = std::numeric_limits<double>::infinity();
            for (int s = 0; s < instance.m; ++s) {
                best = std::min(best, instance.W[s][instance.m + j]);
            }
            if (!std::isfinite(best)) {
                best = 0.0;
            }
            t0[j] += 0.2 * best;
        }

        return t0;
    }

} // namespace mrta