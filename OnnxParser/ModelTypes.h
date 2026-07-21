#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace YirangOnnx
{
	struct OpsetInfo
	{
		std::string domain_;
		int64_t version_ = 0;
	};

	struct MetadataEntry
	{
		std::string key_;
		std::string value_;
	};

	struct ModelMetadata
	{
		int64_t ir_version_ = 0;
		std::string producer_name_;
		std::string producer_version_;
		std::string domain_;
		int64_t model_version_ = 0;
		std::string doc_string_;
		std::string graph_name_;
		std::vector<OpsetInfo> opset_imports_;
		std::vector<MetadataEntry> metadata_props_;
		size_t function_count_ = 0;
	};

	enum class TensorDataSource : uint8_t
	{
		None,
		Inline,
		External
	};

	struct TensorInfo
	{
		std::string name_;
		std::string data_type_;
		std::vector<int64_t> dims_;
		TensorDataSource source_ = TensorDataSource::None;
		size_t byte_size_ = 0;
		std::vector<MetadataEntry> external_data_;

		auto element_count(void) const -> size_t
		{
			size_t count = 1;
			for (int64_t dim : dims_)
			{
				if (dim <= 0)
				{
					return 0;
				}
				const size_t magnitude = static_cast<size_t>(dim);
				if (count > (std::numeric_limits<size_t>::max)() / magnitude)
				{
					return (std::numeric_limits<size_t>::max)();
				}
				count *= magnitude;
			}
			return count;
		}
	};

	struct ValueInfo
	{
		std::string name_;
		std::string data_type_;
		int32_t elem_type_ = 0;
		std::vector<std::string> shape_;
	};

	struct AttributeInfo
	{
		std::string name_;
		std::string type_;
		std::string value_;
		std::vector<int64_t> ints_;
		std::vector<double> floats_;
		std::vector<std::string> strings_;
	};

	struct NodeInfo
	{
		std::string name_;
		std::string op_type_;
		std::string domain_;
		std::vector<std::string> inputs_;
		std::vector<std::string> outputs_;
		std::vector<AttributeInfo> attributes_;
		size_t subgraph_node_count_ = 0;
	};

	auto data_type_name(int32_t data_type) -> std::string;
	auto data_type_id(const std::string& name) -> std::optional<int32_t>;
} // namespace YirangOnnx
