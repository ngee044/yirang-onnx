#pragma once

#include <cstdint>
#include <string>
#include <utility>
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
	};

	struct ValueInfo
	{
		std::string name_;
		std::string data_type_;
		std::vector<std::string> shape_;
	};

	struct NodeInfo
	{
		std::string name_;
		std::string op_type_;
		std::string domain_;
		std::vector<std::string> inputs_;
		std::vector<std::string> outputs_;
		std::vector<std::string> attribute_names_;
	};

	auto data_type_name(int32_t data_type) -> std::string;
}
