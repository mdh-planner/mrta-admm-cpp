#include "InstanceLoader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mrta {

    namespace {

        using EquipSets = std::vector<VecInt>;

        struct ParsedDimensions {
            int m{};
            int n{};
        };

        struct ParsedTasks {
            VecInt k;
            VecInt reqEquip;
            BoolVec isVirtual;
            std::vector<PrecedenceEdge> predPairs;
            MatrixDouble Rpar;
            VecDouble durTask;
        };

        std::string trim(const std::string& s) {
            const auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                return "";
            }
            const auto last = s.find_last_not_of(" \t\r\n");
            return s.substr(first, last - first + 1);
        }

        std::vector<std::string> splitWhitespace(const std::string& s) {
            std::istringstream iss(s);
            std::vector<std::string> out;
            std::string token;
            while (iss >> token) {
                out.push_back(token);
            }
            return out;
        }

        std::vector<std::string> splitByChar(const std::string& s, char delim) {
            std::vector<std::string> out;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, delim)) {
                out.push_back(item);
            }
            return out;
        }

        double parseDoubleStrict(const std::string& token) {
            std::size_t pos = 0;
            const double value = std::stod(token, &pos);
            if (pos != token.size()) {
                throw std::runtime_error("Failed to parse numeric token: " + token);
            }
            return value;
        }

        int parseIntStrict(const std::string& token) {
            std::size_t pos = 0;
            const int value = std::stoi(token, &pos);
            if (pos != token.size()) {
                throw std::runtime_error("Failed to parse integer token: " + token);
            }
            return value;
        }

        bool isSentinelMinusOne(double x) {
            return std::abs(x + 1.0) <= 1e-12;
        }

        double finiteMeanBelowBig(const MatrixDouble& M, double big) {
            double sum = 0.0;
            std::size_t count = 0;

            for (const auto& row : M) {
                for (double v : row) {
                    if (std::isfinite(v) && v < big) {
                        sum += v;
                        ++count;
                    }
                }
            }

            return count == 0 ? 0.0 : sum / static_cast<double>(count);
        }

        double finiteMean(const MatrixDouble& M) {
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

        ParsedDimensions parseDimensionsFromName(const std::string& fileName) {
            static const std::regex pattern(R"(_(\d+)_(\d+)\.txt$)");
            std::smatch match;
            if (!std::regex_search(fileName, match, pattern)) {
                throw std::runtime_error("Could not parse m,n from filename: " + fileName);
            }

            return ParsedDimensions{
                .m = std::stoi(match[1].str()),
                .n = std::stoi(match[2].str())
            };
        }

        std::filesystem::path findSingleMatchingFile(
            const std::filesystem::path& dir,
            const std::regex& pattern,
            const std::string& label
        ) {
            std::vector<std::filesystem::path> matches;

            if (!std::filesystem::exists(dir)) {
                throw std::runtime_error("Instance directory not found: " + dir.string());
            }

            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::string name = entry.path().filename().string();
                if (std::regex_match(name, pattern)) {
                    matches.push_back(entry.path());
                }
            }

            if (matches.empty()) {
                throw std::runtime_error("No " + label + " file found in: " + dir.string());
            }
            if (matches.size() != 1) {
                throw std::runtime_error(
                    "Expected exactly 1 " + label + " file in: " + dir.string() +
                    ", found " + std::to_string(matches.size())
                );
            }

            return matches.front();
        }

        EquipSets readAgentsEquipment(const std::filesystem::path& path, int m) {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("Failed to open agents file: " + path.string());
            }

            std::vector<std::string> lines;
            std::string line;
            while (std::getline(in, line)) {
                line = trim(line);
                if (!line.empty()) {
                    lines.push_back(line);
                }
            }

            if (static_cast<int>(lines.size()) < m) {
                throw std::runtime_error(
                    "Agents file has " + std::to_string(lines.size()) +
                    " lines but expected at least " + std::to_string(m) + "."
                );
            }

            EquipSets equipSets(m);

            for (int i = 0; i < m; ++i) {
                const auto parts = splitWhitespace(lines[i]);
                if (parts.size() < 2) {
                    throw std::runtime_error("Bad agents line: \"" + lines[i] + "\"");
                }

                std::string eqStr;
                for (std::size_t p = 1; p < parts.size(); ++p) {
                    eqStr += parts[p];
                }

                auto tokens = splitByChar(eqStr, ',');
                VecInt eq;
                for (const auto& tokRaw : tokens) {
                    const auto tok = trim(tokRaw);
                    if (tok.empty()) {
                        continue;
                    }
                    eq.push_back(parseIntStrict(tok));
                }

                std::sort(eq.begin(), eq.end());
                eq.erase(std::unique(eq.begin(), eq.end()), eq.end());
                equipSets[i] = std::move(eq);
            }

            return equipSets;
        }

        MatrixDouble readNumericMatrix(const std::filesystem::path& path) {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("Failed to open numeric matrix file: " + path.string());
            }

            MatrixDouble M;
            std::string line;

            while (std::getline(in, line)) {
                line = trim(line);
                if (line.empty()) {
                    continue;
                }

                auto parts = splitWhitespace(line);
                VecDouble row;
                row.reserve(parts.size());

                for (const auto& token : parts) {
                    row.push_back(parseDoubleStrict(token));
                }

                if (!M.empty() && row.size() != M.front().size()) {
                    throw std::runtime_error("Non-rectangular numeric matrix in file: " + path.string());
                }

                M.push_back(std::move(row));
            }

            if (M.empty()) {
                throw std::runtime_error("Numeric matrix file is empty: " + path.string());
            }

            return M;
        }

        ParsedTasks readTasks(const std::filesystem::path& path, int n) {
            const MatrixDouble M = readNumericMatrix(path);

            if (static_cast<int>(M.size()) < n) {
                throw std::runtime_error(
                    "Tasks file has " + std::to_string(M.size()) +
                    " rows but expected at least " + std::to_string(n) + "."
                );
            }
            if (M.front().size() < 6) {
                throw std::runtime_error(
                    "Tasks file must have at least 6 columns, got " +
                    std::to_string(M.front().size()) + "."
                );
            }

            std::vector<double> taskIdxFile(n);
            std::unordered_map<long long, int> mapFileIdToLocal;

            for (int r = 0; r < n; ++r) {
                const double taskId = M[r][0];
                taskIdxFile[r] = taskId;

                const long long key = static_cast<long long>(std::llround(taskId));
                if (mapFileIdToLocal.contains(key)) {
                    throw std::runtime_error("Task indices are not unique.");
                }
                mapFileIdToLocal[key] = r;
            }

            ParsedTasks out;
            out.k.assign(n, 0);
            out.reqEquip.assign(n, 0);
            out.isVirtual.assign(n, false);
            out.durTask.assign(n, 0.0);
            out.predPairs.clear();
            out.Rpar.assign(n, VecDouble(n, 0.0));

            for (int r = 0; r < n; ++r) {
                const auto& row = M[r];
                const int j = r;

                out.k[j] = static_cast<int>(std::llround(row[1]));
                out.reqEquip[j] = static_cast<int>(std::llround(row[2]));
                out.isVirtual[j] = (std::llround(row[3]) != 0);
                const double predFile = row[4];
                out.durTask[j] = row[5];

                if (!std::isnan(predFile) && !isSentinelMinusOne(predFile)) {
                    const long long predKey = static_cast<long long>(std::llround(predFile));
                    if (!mapFileIdToLocal.contains(predKey)) {
                        throw std::runtime_error(
                            "Task predecessor not found among task indices in " + path.string()
                        );
                    }

                    const int predIdx = mapFileIdToLocal[predKey];
                    // MATLAB stores [j, i] meaning task j depends on i.
                    // Internal C++ edge is i -> j.
                    out.predPairs.push_back(PrecedenceEdge{ .from = j, .to = predIdx });
                }

                if (row.size() >= 7) {
                    for (std::size_t c = 6; c < row.size(); ++c) {
                        const double t2File = row[c];
                        if (std::isnan(t2File) || isSentinelMinusOne(t2File)) {
                            continue;
                        }

                        const long long parKey = static_cast<long long>(std::llround(t2File));
                        if (!mapFileIdToLocal.contains(parKey)) {
                            throw std::runtime_error(
                                "Parallel task reference not found among task indices in " + path.string()
                            );
                        }

                        const int jj = mapFileIdToLocal[parKey];
                        out.Rpar[j][jj] = 1.0;
                        out.Rpar[jj][j] = 1.0;
                    }
                }
            }

            return out;
        }

        MatrixDouble readWeightsMatrix(const std::filesystem::path& path) {
            MatrixDouble W = readNumericMatrix(path);
            for (auto& row : W) {
                for (double& v : row) {
                    if (std::isnan(v)) {
                        v = 0.0;
                    }
                }
            }
            return W;
        }

    } // namespace

    InstanceData InstanceLoader::load(const UserConfig& config) const {
        const std::string instDir = buildInstanceDirectory();
        InstanceData instance = loadCaseInstance(instDir, config.instId);

        buildDepotNodes(instance);
        validateInstance(instance);
        buildServiceMatrices(instance);
        buildAssignmentCostMatrix(instance);

        std::cout
            << "Loaded instance " << config.instId
            << ": m=" << instance.m
            << " robots, n=" << instance.n
            << " tasks, depots=" << instance.depotNodes.size()
            << '\n';

        if (config.useFixedEndDepot) {
            instance.depotPolicy = DepotPolicy::Fixed;
        }
        else {
            instance.depotPolicy = DepotPolicy::Free;
            instance.endDepotFixed.clear();
        }

        return instance;
    }

    std::string InstanceLoader::buildInstanceDirectory() const {
        const auto cwd = std::filesystem::current_path();
        return (cwd / "Test_Instances").string();
    }

    InstanceData InstanceLoader::loadCaseInstance(const std::string& instanceDir, int instId) const {
        const std::filesystem::path dir(instanceDir);

        const std::regex patA("^Inst\\(" + std::to_string(instId) + R"(\)_agents_.*\.txt$)");
        const std::regex patT("^Inst\\(" + std::to_string(instId) + R"(\)_tasks_.*\.txt$)");
        const std::regex patW("^Inst\\(" + std::to_string(instId) + R"(\)_weights_.*\.txt$)");

        const auto agentsPath = findSingleMatchingFile(dir, patA, "agents");
        const auto tasksPath = findSingleMatchingFile(dir, patT, "tasks");
        const auto weightsPath = findSingleMatchingFile(dir, patW, "weights");

        const auto dimA = parseDimensionsFromName(agentsPath.filename().string());
        const auto dimT = parseDimensionsFromName(tasksPath.filename().string());
        const auto dimW = parseDimensionsFromName(weightsPath.filename().string());

        if (dimA.m != dimT.m || dimA.m != dimW.m || dimA.n != dimT.n || dimA.n != dimW.n) {
            throw std::runtime_error(
                "Instance " + std::to_string(instId) + ": mismatch in (m,n) across filenames."
            );
        }

        const int m = dimA.m;
        const int n = dimA.n;

        const EquipSets agentEquipSets = readAgentsEquipment(agentsPath, m);
        const ParsedTasks taskData = readTasks(tasksPath, n);
        const MatrixDouble W = readWeightsMatrix(weightsPath);

        if (static_cast<int>(W.size()) < (m + n) || static_cast<int>(W.front().size()) < (m + n)) {
            throw std::runtime_error(
                "Weights matrix too small: got " +
                std::to_string(W.size()) + "x" + std::to_string(W.front().size()) +
                ", need at least " + std::to_string(m + n) + "x" + std::to_string(m + n) + "."
            );
        }

        MatrixDouble cap(m, VecDouble(n, 0.0));
        for (int s = 0; s < m; ++s) {
            for (int j = 0; j < n; ++j) {
                if (taskData.reqEquip[j] == -1) {
                    cap[s][j] = 1.0;
                }
                else {
                    cap[s][j] = std::binary_search(
                        agentEquipSets[s].begin(),
                        agentEquipSets[s].end(),
                        taskData.reqEquip[j]
                    ) ? 1.0 : 0.0;
                }
            }
        }

        BoolVec isMR(n, false);
        for (int j = 0; j < n; ++j) {
            isMR[j] = (!taskData.isVirtual[j] && taskData.k[j] > 1);
        }

        MatrixDouble Tstart(m, VecDouble(n, 0.0));
        for (int s = 0; s < m; ++s) {
            for (int j = 0; j < n; ++j) {
                Tstart[s][j] = W[s][m + j];
            }
        }

        MatrixDouble TT(n, VecDouble(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                TT[i][j] = W[m + i][m + j];
            }
        }

        InstanceData I;
        I.instId = instId;
        I.instDir = instanceDir;
        I.m = m;
        I.n = n;
        I.k = taskData.k;
        I.reqEquip = taskData.reqEquip;
        I.isVirtual = taskData.isVirtual;
        I.isMR = isMR;
        I.durTask = taskData.durTask;
        I.cap = std::move(cap);
        I.predPairs = taskData.predPairs;
        I.Rpar = taskData.Rpar;
        I.W = W;
        I.Tstart = std::move(Tstart);
        I.TT = std::move(TT);

        return I;
    }

    void InstanceLoader::validateInstance(const InstanceData& instance) const {
        if (instance.m <= 0 || instance.n <= 0) {
            throw std::runtime_error("Invalid instance: m and n must be positive.");
        }

        if (static_cast<int>(instance.k.size()) != instance.n) {
            throw std::runtime_error("Invalid instance: k must have length n.");
        }

        for (int j = 0; j < instance.n; ++j) {
            if (instance.isVirtual[j] && instance.k[j] > 1) {
                std::ostringstream oss;
                oss << "Invalid instance: virtual task " << (j + 1)
                    << " has k>1. Virtual tasks must have k=1.";
                throw std::runtime_error(oss.str());
            }

            if (instance.isVirtual[j] && instance.isMR[j]) {
                std::ostringstream oss;
                oss << "Invalid instance: virtual task " << (j + 1)
                    << " marked MR. Virtual tasks must be SR.";
                throw std::runtime_error(oss.str());
            }
        }
    }

    void InstanceLoader::buildDepotNodes(InstanceData& instance) const {
        const int totalRows = static_cast<int>(instance.W.size());
        const int totalCols = instance.W.empty() ? 0 : static_cast<int>(instance.W.front().size());

        if (totalRows != totalCols) {
            throw std::runtime_error("Weights matrix must be square.");
        }

        const int nDep = totalRows - (instance.m + instance.n);

        if (nDep <= 0) {
            throw std::runtime_error(
                "Weights matrix has no depots. Expected additional rows after (m+n)."
            );
        }

        instance.depotNodes.clear();
        instance.depotNodes.reserve(nDep);

        for (int node = instance.m + instance.n; node < instance.m + instance.n + nDep; ++node) {
            instance.depotNodes.push_back(node);
        }
    }

    void InstanceLoader::buildServiceMatrices(InstanceData& instance) const {
        instance.svcPhysSJ.assign(instance.m, VecDouble(instance.n, 0.0));
        instance.svcVirtSJ.assign(instance.m, VecDouble(instance.n, 0.0));

        for (int j = 0; j < instance.n; ++j) {
            for (int s = 0; s < instance.m; ++s) {
                if (instance.isVirtual[j]) {
                    instance.svcVirtSJ[s][j] = instance.durTask[j];
                    instance.svcPhysSJ[s][j] = 0.0;
                }
                else {
                    instance.svcPhysSJ[s][j] = instance.durTask[j];
                    instance.svcVirtSJ[s][j] = 0.0;
                }
            }
        }

        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int s = 0; s < instance.m; ++s) {
            for (int j = 0; j < instance.n; ++j) {
                if (instance.cap[s][j] < 0.5) {
                    instance.svcPhysSJ[s][j] = nan;
                    instance.svcVirtSJ[s][j] = nan;
                }
            }
        }
    }

    void InstanceLoader::buildAssignmentCostMatrix(InstanceData& instance) const {
        constexpr double BIG = 1e6;

        instance.cY.assign(instance.m, VecDouble(instance.n, 0.0));

        for (int s = 0; s < instance.m; ++s) {
            for (int j = 0; j < instance.n; ++j) {
                if (instance.cap[s][j] < 0.5) {
                    instance.cY[s][j] = BIG;
                }
                else if (instance.isVirtual[j]) {
                    instance.cY[s][j] = 0.1;
                }
                else {
                    instance.cY[s][j] = instance.Tstart[s][j];
                }
            }
        }

        const double meanCY = std::max(1e-9, finiteMeanBelowBig(instance.cY, BIG));
        for (auto& row : instance.cY) {
            for (double& v : row) {
                v = 5.0 * (v / meanCY);
            }
        }

        const double epsTie = 0.02 * finiteMeanBelowBig(instance.cY, BIG);
        for (int s = 0; s < instance.m; ++s) {
            for (int j = 0; j < instance.n; ++j) {
                if (instance.cap[s][j] >= 0.5) {
                    instance.cY[s][j] += epsTie * (static_cast<double>(s + 1) / instance.m);
                }
                else {
                    instance.cY[s][j] = BIG;
                }
            }
        }

        instance.mrTravel = instance.Tstart;
        const double meanMrTravel = std::max(1e-9, finiteMean(instance.mrTravel));
        for (auto& row : instance.mrTravel) {
            for (double& v : row) {
                v /= meanMrTravel;
            }
        }
    }

} // namespace mrta