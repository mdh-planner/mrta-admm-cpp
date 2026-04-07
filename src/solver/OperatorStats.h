#pragma once
// ============================================================================
// OperatorStats.h
//
// Tracks per-primitive operator performance across all ILS iterations.
//
// Usage pattern:
//   1. Each primitive in LocalSearch calls recordOp() after it returns.
//   2. IlsSolver takes a snapshot() before each LS run.
//   3. If the run produced a new global best  -> keep (do nothing).
//      If the run did NOT beat the global best -> restore(snap) to roll back.
//
// This means TotalDelta only reflects improvements on the path to the
// final answer, not wasted work on perturbed solutions that were discarded.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace mrta {

	// ---------------------------------------------------------------------------
	// Call context
	// ---------------------------------------------------------------------------
	enum class OpContext : int {
		TopLevel = 0,
		Polish = 1,
		DeepEval = 2,
		ThoroughEval = 3,
		COUNT = 4
	};

	inline const char* opContextName(OpContext c) {
		switch (c) {
		case OpContext::TopLevel:     return "TopLevel";
		case OpContext::Polish:       return "Polish";
		case OpContext::DeepEval:     return "DeepEval";
		case OpContext::ThoroughEval: return "ThoroughEval";
		default:                      return "Unknown";
		}
	}

	// ---------------------------------------------------------------------------
	// Per-primitive, per-context entry
	// ---------------------------------------------------------------------------
	struct OpEntry {
		std::string name;
		OpContext   context;
		int64_t     skips = 0;
		int64_t     calls = 0;
		int64_t     hits = 0;
		double      deltaTotal = 0.0;
	};

	// ---------------------------------------------------------------------------
	// OperatorStats
	// ---------------------------------------------------------------------------
	struct OperatorStats {

		std::vector<OpEntry> entries;
		OpContext currentContext = OpContext::TopLevel;

		// -----------------------------------------------------------------------
		// RAII context guard
		// -----------------------------------------------------------------------
		struct ScopeContext {
			OperatorStats* stats;
			OpContext      saved;
			ScopeContext(OperatorStats* s, OpContext c)
				: stats(s)
				, saved(s ? s->currentContext : OpContext::TopLevel)
			{
				if (stats) stats->currentContext = c;
			}
			~ScopeContext() { if (stats) stats->currentContext = saved; }
		};

		// -----------------------------------------------------------------------
		// record() — call after each primitive returns.
		// -----------------------------------------------------------------------
		void record(const std::string& name, bool hit, double delta)
		{
			OpEntry* e = find(name, currentContext);
			if (!e) {
				entries.push_back({ name, currentContext, 0, 0, 0, 0.0 });
				e = &entries.back();
			}
			++e->calls;
			if (hit) {
				++e->hits;
				e->deltaTotal += delta;
			}
		}

		// -----------------------------------------------------------------------
		// recordSkip()
		// -----------------------------------------------------------------------
		void recordSkip(const std::string& name)
		{
			OpEntry* e = find(name, currentContext);
			if (!e) {
				entries.push_back({ name, currentContext, 0, 0, 0, 0.0 });
				e = &entries.back();
			}
			++e->skips;
		}

		// -----------------------------------------------------------------------
		// snapshot() / restore()
		//
		// Call snapshot() in IlsSolver before each ls.run().
		// If the result beats the global best: do nothing (keep the recorded stats).
		// If the result does NOT beat the global best: call restore(snap) to undo
		// all operator stats that were recorded during that wasted LS run.
		// -----------------------------------------------------------------------
		std::vector<OpEntry> snapshot() const
		{
			return entries;
		}

		void restore(const std::vector<OpEntry>& snap)
		{
			entries = snap;
		}

		// -----------------------------------------------------------------------
		// reset()
		// -----------------------------------------------------------------------
		void reset()
		{
			entries.clear();
			currentContext = OpContext::TopLevel;
		}

		// -----------------------------------------------------------------------
		// printSummary()
		// -----------------------------------------------------------------------
		void printSummary() const
		{
			if (entries.empty()) {
				std::cout << "[OperatorStats] No data recorded.\n";
				return;
			}

			auto sorted = entries;
			std::sort(sorted.begin(), sorted.end(), [](const OpEntry& a, const OpEntry& b) {
				if (a.context != b.context)
					return static_cast<int>(a.context) < static_cast<int>(b.context);
				return a.name < b.name;
				});

			constexpr int W_NAME = 30;
			constexpr int W_CTX = 14;
			constexpr int W_NUM = 9;
			constexpr int W_PCT = 9;
			constexpr int W_AVG = 12;
			const     int W_TOT = W_NAME + W_CTX + W_NUM * 2 + W_PCT + W_AVG + W_NUM + 6;

			std::cout << "\n" << std::string(W_TOT, '=') << "\n";
			std::cout << "  OPERATOR STATS SUMMARY (only improvements on path to global best)\n";
			std::cout << std::string(W_TOT, '=') << "\n";
			std::cout
				<< std::left << std::setw(W_NAME) << "Operator"
				<< std::left << std::setw(W_CTX) << "Context"
				<< std::right << std::setw(W_NUM) << "Calls"
				<< std::right << std::setw(W_NUM) << "Hits"
				<< std::right << std::setw(W_PCT) << "Hit%"
				<< std::right << std::setw(W_AVG) << "AvgDelta"
				<< std::right << std::setw(W_NUM) << "Skips"
				<< "\n";
			std::cout << std::string(W_TOT, '-') << "\n";

			OpContext lastCtx = static_cast<OpContext>(-1);
			for (const auto& e : sorted) {
				if (e.context != lastCtx) {
					if (lastCtx != static_cast<OpContext>(-1))
						std::cout << std::string(W_TOT, '-') << "\n";
					lastCtx = e.context;
				}
				const double hitPct = e.calls > 0
					? 100.0 * static_cast<double>(e.hits) / static_cast<double>(e.calls) : 0.0;
				const double avgDelta = e.hits > 0
					? e.deltaTotal / static_cast<double>(e.hits) : 0.0;
				std::cout
					<< std::left << std::setw(W_NAME) << e.name
					<< std::left << std::setw(W_CTX) << opContextName(e.context)
					<< std::right << std::setw(W_NUM) << e.calls
					<< std::right << std::setw(W_NUM) << e.hits
					<< std::right << std::setw(W_PCT) << std::fixed << std::setprecision(1) << hitPct
					<< std::right << std::setw(W_AVG) << std::fixed << std::setprecision(2) << avgDelta
					<< std::right << std::setw(W_NUM) << e.skips
					<< "\n";
			}
			std::cout << std::string(W_TOT, '=') << "\n";

			// Totals collapsed across contexts
			const int W_TOT2 = W_NAME + W_NUM * 2 + W_PCT + W_AVG + 4;
			std::cout << "\n  TOTALS (collapsed across contexts)\n";
			std::cout << std::string(W_TOT2, '-') << "\n";
			std::cout
				<< std::left << std::setw(W_NAME) << "Operator"
				<< std::right << std::setw(W_NUM) << "Calls"
				<< std::right << std::setw(W_NUM) << "Hits"
				<< std::right << std::setw(W_PCT) << "Hit%"
				<< std::right << std::setw(W_AVG) << "TotalDelta"
				<< "\n";
			std::cout << std::string(W_TOT2, '-') << "\n";

			std::vector<std::string> names;
			for (const auto& e : sorted)
				if (std::find(names.begin(), names.end(), e.name) == names.end())
					names.push_back(e.name);
			std::sort(names.begin(), names.end());

			double grandTotal = 0.0;
			for (const auto& name : names) {
				int64_t totalCalls = 0, totalHits = 0;
				double  totalDelta = 0.0;
				for (const auto& e : sorted) {
					if (e.name == name) {
						totalCalls += e.calls;
						totalHits += e.hits;
						totalDelta += e.deltaTotal;
					}
				}
				grandTotal += totalDelta;
				const double hitPct = totalCalls > 0
					? 100.0 * static_cast<double>(totalHits) / static_cast<double>(totalCalls) : 0.0;
				std::cout
					<< std::left << std::setw(W_NAME) << name
					<< std::right << std::setw(W_NUM) << totalCalls
					<< std::right << std::setw(W_NUM) << totalHits
					<< std::right << std::setw(W_PCT) << std::fixed << std::setprecision(1) << hitPct
					<< std::right << std::setw(W_AVG) << std::fixed << std::setprecision(2) << totalDelta
					<< "\n";
			}
			std::cout << std::string(W_TOT2, '-') << "\n";
			std::cout
				<< std::left << std::setw(W_NAME) << "GRAND TOTAL"
				<< std::right << std::setw(W_NUM) << ""
				<< std::right << std::setw(W_NUM) << ""
				<< std::right << std::setw(W_PCT) << ""
				<< std::right << std::setw(W_AVG) << std::fixed << std::setprecision(2) << grandTotal
				<< "\n";
			std::cout << std::string(W_TOT2, '=') << "\n\n";
		}

	private:

		OpEntry* find(const std::string& name, OpContext ctx)
		{
			for (auto& e : entries)
				if (e.name == name && e.context == ctx) return &e;
			return nullptr;
		}
	};

} // namespace mrta