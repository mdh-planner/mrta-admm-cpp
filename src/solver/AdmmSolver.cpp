#include "AdmmSolver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mrta {

    AdmmResult AdmmSolver::run(
        const InstanceData& inst,
        const DerivedData& data,
        const SolverParameters& params) const
    {
        const auto& AP = params.admm;

        AdmmResult result = runLoop(
            inst,
            data,
            params,
            AP.maxIter,
            AP.rhoSync,
            AP.dualRelax,
            AP.thetaRelax,
            AP.tolSync
        );

        if (!result.converged) {
            std::cout << "  ADMM did not converge. Retrying with fallback damping...\n";

            result = runLoop(
                inst,
                data,
                params,
                AP.fallbackMaxIter,
                AP.fallbackRhoSync,
                AP.fallbackDualRelax,
                AP.fallbackThetaRelax,
                AP.fallbackTolSync
            );

            if (!result.converged) {
                std::cout << "  ADMM fallback also did not converge. Proceeding with best solution.\n";
            }
        }

        return result;
    }

    AdmmResult AdmmSolver::runLoop(
        const InstanceData& inst,
        const DerivedData& data,
        const SolverParameters& params,
        int maxIter,
        double rhoSync,
        double dualRelax,
        double thetaRelax,
        double tolSync) const
    {
        const int m = inst.m;
        const int n = inst.n;

        const double BIG = 1e6;
        const auto& AP = params.admm;

        MatrixDouble y(m, VecDouble(n, 0.0));
        MatrixDouble z(m, VecDouble(n, 0.0));
        MatrixDouble u(m, VecDouble(n, 0.0));
        MatrixDouble nu(m, VecDouble(n, 0.0));

        MatrixDouble tau = data.tauBar;
        const MatrixDouble& tauBar = data.tauBar;

        VecDouble theta(n, 0.0);

        VecDouble histCard(maxIter, 0.0);
        VecDouble histSync(maxIter, 0.0);

        const double lambdaScarce = 0.35;
        const double lambdaLoadDyn = 0.15;
        const bool useCapNormalizedBalance = true;

        VecDouble taskFlex(n, 0.0);
        VecDouble capCount(m, 1.0);
        VecDouble scarcityRobot(m, 1.0);

        for (int j = 0; j < n; ++j) {
            int cnt = 0;
            for (int s = 0; s < m; ++s) {
                if (inst.cap[s][j] > 0.5) {
                    ++cnt;
                }
            }
            taskFlex[j] = static_cast<double>(cnt);
        }

        for (int s = 0; s < m; ++s) {
            int cnt = 0;
            for (int j = 0; j < n; ++j) {
                if (inst.cap[s][j] > 0.5) {
                    ++cnt;
                }
            }
            capCount[s] = static_cast<double>(std::max(cnt, 1));
            scarcityRobot[s] = 1.0 / capCount[s];
        }

        for (int s = 0; s < m; ++s) {
            for (int j = 0; j < n; ++j) {
                tau[s][j] = std::max(tauBar[s][j], 0.0);
            }
        }

        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int s = 0; s < m; ++s) {
                sum += tau[s][j];
            }
            theta[j] = sum / static_cast<double>(std::max(m, 1));
        }

        for (int j = 0; j < n; ++j) {
            std::vector<int> idxCap;
            idxCap.reserve(m);

            for (int s = 0; s < m; ++s) {
                if (inst.cap[s][j] > 0.5) {
                    idxCap.push_back(s);
                }
            }

            if (idxCap.empty()) {
                throw std::runtime_error("AdmmSolver::runLoop: task has no capable robots.");
            }

            const double initVal =
                std::min(1.0, static_cast<double>(inst.k[j]) / static_cast<double>(idxCap.size()));

            for (int s : idxCap) {
                y[s][j] = initVal;
            }
        }

        z = y;
        MatrixDouble cYeff = inst.cY;

        bool converged = false;
        int lastIter = maxIter;

        for (int it = 0; it < maxIter; ++it) {
            // -------------------------------------------------
            // Periodic shaping
            // -------------------------------------------------
            if (it == 0 || ((it + 1) % AP.shapingPeriod == 0)) {
                VecDouble workLoad(m, 0.0);
                for (int s = 0; s < m; ++s) {
                    for (int j = 0; j < n; ++j) {
                        const double svc = data.svcEst[s][j];
                        if (std::isfinite(svc)) {
                            workLoad[s] += z[s][j] * svc;
                        }
                    }
                }

                double workTarget = 0.0;
                for (double w : workLoad) {
                    workTarget += w;
                }
                workTarget /= static_cast<double>(std::max(m, 1));

                VecDouble overWork(m, 0.0);
                for (int s = 0; s < m; ++s) {
                    overWork[s] = std::max(0.0, (workLoad[s] - workTarget) / std::max(workTarget, 1e-9));
                }

                VecDouble taskScarcity(n, 0.0);
                for (int j = 0; j < n; ++j) {
                    taskScarcity[j] = 1.0 / std::max(taskFlex[j], 1.0);
                }

                cYeff = inst.cY;
                for (int s = 0; s < m; ++s) {
                    for (int j = 0; j < n; ++j) {
                        cYeff[s][j] += AP.lambdaLoad * overWork[s];
                        cYeff[s][j] -= lambdaScarce * (scarcityRobot[s] * taskScarcity[j]);

                        if (inst.isMR[j]) {
                            cYeff[s][j] += AP.lambdaMr * data.mrTravel[s][j];
                        }

                        if (inst.cap[s][j] < 0.5) {
                            cYeff[s][j] = BIG;
                        }
                    }
                }
            }

            // -------------------------------------------------
            // Local y and tau updates
            // -------------------------------------------------
            VecDouble loadSNow(m, 0.0);
            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    loadSNow[s] += z[s][j];
                }
            }

            VecDouble loadMetricNow(m, 0.0);
            for (int s = 0; s < m; ++s) {
                loadMetricNow[s] = useCapNormalizedBalance
                    ? loadSNow[s] / capCount[s]
                    : loadSNow[s];
            }

            double targetMetricNow = 0.0;
            for (double v : loadMetricNow) {
                targetMetricNow += v;
            }
            targetMetricNow /= static_cast<double>(std::max(m, 1));

            VecDouble overloadNow(m, 0.0);
            for (int s = 0; s < m; ++s) {
                overloadNow[s] = std::max(0.0, loadMetricNow[s] - targetMetricNow);
            }

            for (int s = 0; s < m; ++s) {
                const double balPenS = lambdaLoadDyn * overloadNow[s];

                for (int j = 0; j < n; ++j) {
                    if (inst.cap[s][j] < 0.5) {
                        y[s][j] = 0.0;
                        tau[s][j] = 0.0;
                    }
                    else {
                        const double yStar =
                            (AP.rhoCard * (z[s][j] - u[s][j]) - cYeff[s][j] - balPenS) /
                            (2.0 * AP.alphaY + AP.rhoCard);

                        y[s][j] = std::clamp(yStar, 0.0, 1.0);

                        const double tb = std::max(0.0, tauBar[s][j]);

                        if (inst.isMR[j]) {
                            const double w = rhoSync * z[s][j];
                            const double denom = 2.0 * AP.alphaTau + w;
                            const double numer =
                                2.0 * AP.alphaTau * tb + w * (theta[j] - nu[s][j]);

                            tau[s][j] = std::max(0.0, numer / denom);
                        }
                        else {
                            tau[s][j] = tb;
                        }
                    }
                }
            }

            // -------------------------------------------------
            // z projection
            // -------------------------------------------------
            for (int j = 0; j < n; ++j) {
                VecDouble vj(m, 0.0);
                VecDouble lo(m, 0.0);
                VecDouble hi(m, 0.0);

                for (int s = 0; s < m; ++s) {
                    vj[s] = y[s][j] + u[s][j];
                    lo[s] = 0.0;
                    hi[s] = inst.cap[s][j];
                }

                const VecDouble proj = projectBoxedSimplex(
                    vj,
                    static_cast<double>(inst.k[j]),
                    lo,
                    hi
                );

                for (int s = 0; s < m; ++s) {
                    z[s][j] = proj[s];
                }
            }

            // -------------------------------------------------
            // Dual update
            // -------------------------------------------------
            for (int s = 0; s < m; ++s) {
                for (int j = 0; j < n; ++j) {
                    u[s][j] += (y[s][j] - z[s][j]);
                }
            }

            // -------------------------------------------------
            // MR theta consensus
            // -------------------------------------------------
            for (int j = 0; j < n; ++j) {
                if (!inst.isMR[j]) {
                    continue;
                }

                double wsum = 1e-12;
                double weightedSum = 0.0;

                for (int s = 0; s < m; ++s) {
                    const double w = z[s][j];
                    wsum += w;
                    weightedSum += w * (tau[s][j] + nu[s][j]);
                }

                const double thetaNew = weightedSum / wsum;
                theta[j] = (1.0 - thetaRelax) * theta[j] + thetaRelax * thetaNew;

                for (int s = 0; s < m; ++s) {
                    const double w = z[s][j];
                    nu[s][j] += dualRelax * w * (tau[s][j] - theta[j]);
                }
            }

            // -------------------------------------------------
            // Residuals
            // -------------------------------------------------
            double rCard = 0.0;
            for (int j = 0; j < n; ++j) {
                double sumz = 0.0;
                for (int s = 0; s < m; ++s) {
                    sumz += z[s][j];
                }
                rCard = std::max(rCard, std::abs(sumz - static_cast<double>(inst.k[j])));
            }

            double rSync = 0.0;
            for (int j = 0; j < n; ++j) {
                if (!inst.isMR[j]) {
                    continue;
                }
                for (int s = 0; s < m; ++s) {
                    const double val = z[s][j] * (tau[s][j] - theta[j]);
                    rSync = std::max(rSync, std::abs(val));
                }
            }

            histCard[it] = rCard;
            histSync[it] = rSync;

            if (((it + 1) % 25) == 0 || it == 0) {
                VecDouble rowLoads(m, 0.0);
                for (int s = 0; s < m; ++s) {
                    for (int j = 0; j < n; ++j) {
                        rowLoads[s] += z[s][j];
                    }
                }

                double meanLoad = 0.0;
                double maxLoad = 0.0;
                for (double v : rowLoads) {
                    meanLoad += v;
                    maxLoad = std::max(maxLoad, v);
                }
                meanLoad /= static_cast<double>(std::max(m, 1));

                std::cout
                    << "iter " << (it + 1)
                    << " | r_card=" << rCard
                    << " | r_sync=" << rSync
                    << " | mean load=" << meanLoad
                    << " | max load=" << maxLoad
                    << '\n';
            }

            if (rCard < AP.tolCard && rSync < tolSync) {
                converged = true;
                lastIter = it + 1;
                std::cout << "Converged at iter " << lastIter << '\n';
                break;
            }
        }

        histCard.resize(lastIter);
        histSync.resize(lastIter);

        return AdmmResult{ z, theta, converged, histCard, histSync };
    }

    VecDouble AdmmSolver::projectBoxedSimplex(
        const VecDouble& v,
        double k,
        const VecDouble& lo,
        const VecDouble& hi) const
    {
        if (v.size() != lo.size() || v.size() != hi.size()) {
            throw std::invalid_argument("projectBoxedSimplex: size mismatch.");
        }

        const std::size_t n = v.size();

        double sumLo = 0.0;
        double sumHi = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (lo[i] > hi[i]) {
                throw std::invalid_argument("projectBoxedSimplex: lo[i] > hi[i].");
            }
            sumLo += lo[i];
            sumHi += hi[i];
        }

        k = std::min(std::max(k, sumLo), sumHi);

        double lamLow = std::numeric_limits<double>::infinity();
        double lamHigh = -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < n; ++i) {
            lamLow = std::min(lamLow, v[i] - hi[i]);
            lamHigh = std::max(lamHigh, v[i] - lo[i]);
        }

        lamLow -= 1.0;
        lamHigh += 1.0;

        VecDouble x(n, 0.0);

        for (int it = 0; it < 80; ++it) {
            const double lam = 0.5 * (lamLow + lamHigh);

            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = std::min(hi[i], std::max(lo[i], v[i] - lam));
                s += x[i];
            }

            if (s > k) {
                lamLow = lam;
            }
            else {
                lamHigh = lam;
            }
        }

        const double lam = 0.5 * (lamLow + lamHigh);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::min(hi[i], std::max(lo[i], v[i] - lam));
        }

        return x;
    }

} // namespace mrta