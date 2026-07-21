#pragma once

#include <optional>

namespace YirangOnnx
{
	enum class ExecutionMode
	{
		sequential,
		parallel
	};

	enum class GraphOptimization
	{
		disabled,
		basic,
		extended,
		all
	};

	struct SessionTuning
	{
		std::optional<int> intra_op_threads_;
		std::optional<int> inter_op_threads_;
		bool enable_mem_pattern_ = true;
		bool enable_cpu_mem_arena_ = true;
		ExecutionMode execution_mode_ = ExecutionMode::sequential;
		GraphOptimization graph_optimization_ = GraphOptimization::all;
	};
} // namespace YirangOnnx
