#pragma once

#include "ModelTypes.h"
#include "Tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace YirangOnnx
{
	inline constexpr size_t kMaxTensorElements = static_cast<size_t>(1) << 30;
	inline constexpr size_t kMaxTensorBytes = static_cast<size_t>(1) << 32;

	struct ResolvedShape
	{
		std::vector<int64_t> dims_;
		std::vector<std::string> notes_;
	};

	auto random_generation_supports(int32_t elem_type) -> bool;

	auto random_element_byte_width(int32_t elem_type) -> size_t;

	auto resolve_input_shape(const ValueInfo& input, const std::map<std::string, int64_t>& dim_overrides)
		-> std::tuple<std::optional<ResolvedShape>, std::optional<std::string>>;

	auto make_random_tensor(const std::string& name, int32_t elem_type, const std::vector<int64_t>& shape, std::optional<uint64_t> seed)
		-> std::tuple<std::optional<Tensor>, std::optional<std::string>>;
} // namespace YirangOnnx
