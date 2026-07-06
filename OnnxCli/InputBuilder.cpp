#include "InputBuilder.h"

#include "onnx.pb.h"

#include <charconv>
#include <cstring>
#include <format>
#include <random>
#include <utility>

namespace YirangOnnx
{
	auto resolve_input_shape(const ValueInfo& input, const std::map<std::string, int64_t>& dim_overrides)
		-> std::tuple<std::optional<ResolvedShape>, std::optional<std::string>>
	{
		ResolvedShape resolved;
		for (const auto& dim : input.shape_)
		{
			int64_t value = 0;
			auto [ptr, ec] = std::from_chars(dim.data(), dim.data() + dim.size(), value);
			if (ec == std::errc() && ptr == dim.data() + dim.size())
			{
				if (value < 1)
				{
					return { std::nullopt, std::format("input '{}': dimension '{}' is not positive", input.name_, dim) };
				}
				resolved.dims_.push_back(value);
				continue;
			}

			if (auto found = dim_overrides.find(dim); found != dim_overrides.end())
			{
				resolved.dims_.push_back(found->second);
				continue;
			}

			resolved.dims_.push_back(1);
			resolved.notes_.push_back(std::format("input '{}': symbolic dimension '{}' -> 1 (set dim_overrides to change)", input.name_, dim));
		}
		return { std::move(resolved), std::nullopt };
	}

	auto make_random_tensor(const std::string& name, int32_t elem_type, const std::vector<int64_t>& shape, std::optional<uint64_t> seed)
		-> std::tuple<std::optional<Tensor>, std::optional<std::string>>
	{
		size_t count = 1;
		for (int64_t dim : shape)
		{
			if (dim < 1)
			{
				return { std::nullopt, std::format("input '{}': non-positive dimension {}", name, dim) };
			}
			count *= static_cast<size_t>(dim);
		}

		Tensor tensor;
		tensor.name_ = name;
		tensor.elem_type_ = elem_type;
		tensor.shape_ = shape;

		std::mt19937_64 rng(seed.has_value() ? seed.value() : std::random_device{}());

		switch (elem_type)
		{
		case onnx::TensorProto::FLOAT:
		{
			std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
			tensor.data_.resize(count * sizeof(float));
			for (size_t i = 0; i < count; ++i)
			{
				const float value = dist(rng);
				std::memcpy(tensor.data_.data() + i * sizeof(float), &value, sizeof(float));
			}
			break;
		}
		case onnx::TensorProto::DOUBLE:
		{
			std::uniform_real_distribution<double> dist(-1.0, 1.0);
			tensor.data_.resize(count * sizeof(double));
			for (size_t i = 0; i < count; ++i)
			{
				const double value = dist(rng);
				std::memcpy(tensor.data_.data() + i * sizeof(double), &value, sizeof(double));
			}
			break;
		}
		case onnx::TensorProto::INT32:
		{
			std::uniform_int_distribution<int32_t> dist(0, 9);
			tensor.data_.resize(count * sizeof(int32_t));
			for (size_t i = 0; i < count; ++i)
			{
				const int32_t value = dist(rng);
				std::memcpy(tensor.data_.data() + i * sizeof(int32_t), &value, sizeof(int32_t));
			}
			break;
		}
		case onnx::TensorProto::INT64:
		{
			std::uniform_int_distribution<int64_t> dist(0, 9);
			tensor.data_.resize(count * sizeof(int64_t));
			for (size_t i = 0; i < count; ++i)
			{
				const int64_t value = dist(rng);
				std::memcpy(tensor.data_.data() + i * sizeof(int64_t), &value, sizeof(int64_t));
			}
			break;
		}
		case onnx::TensorProto::BOOL:
		{
			std::bernoulli_distribution dist(0.5);
			tensor.data_.resize(count);
			for (size_t i = 0; i < count; ++i)
			{
				tensor.data_[i] = dist(rng) ? 1 : 0;
			}
			break;
		}
		default:
			return { std::nullopt, std::format("input '{}': random generation unsupported for {}", name, data_type_name(elem_type)) };
		}

		return { std::move(tensor), std::nullopt };
	}
} // namespace YirangOnnx
