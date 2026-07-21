#include "OnnxModel.h"

#include "File.h"
#include "JsonTool.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <boost/json.hpp>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
		constexpr size_t attribute_preview_limit = 16;
		constexpr size_t weight_export_element_limit = static_cast<size_t>(1) << 24;

		auto to_value_info(const onnx::ValueInfoProto& value) -> ValueInfo
		{
			ValueInfo info;
			info.name_ = value.name();
			if (value.has_type() && value.type().has_tensor_type())
			{
				const auto& tensor = value.type().tensor_type();
				info.data_type_ = data_type_name(tensor.elem_type());
				info.elem_type_ = tensor.elem_type();
				if (tensor.has_shape())
				{
					const auto& shape = tensor.shape();
					for (int i = 0; i < shape.dim_size(); ++i)
					{
						const auto& dim = shape.dim(i);
						if (dim.has_dim_value())
						{
							info.shape_.push_back(std::to_string(dim.dim_value()));
						}
						else if (dim.has_dim_param())
						{
							info.shape_.push_back(dim.dim_param());
						}
						else
						{
							info.shape_.push_back("?");
						}
					}
				}
			}
			else if (value.has_type())
			{
				switch (value.type().value_case())
				{
				case onnx::TypeProto::kSequenceType:
					if (value.type().sequence_type().has_elem_type() && value.type().sequence_type().elem_type().has_tensor_type())
					{
						info.data_type_ = std::format("SEQUENCE({})", data_type_name(value.type().sequence_type().elem_type().tensor_type().elem_type()));
					}
					else
					{
						info.data_type_ = "SEQUENCE";
					}
					break;
				case onnx::TypeProto::kMapType:
					info.data_type_ = "MAP";
					break;
				case onnx::TypeProto::kOptionalType:
					info.data_type_ = "OPTIONAL";
					break;
				case onnx::TypeProto::kSparseTensorType:
					info.data_type_ = "SPARSE_TENSOR";
					break;
				default:
					break;
				}
			}
			return info;
		}

		auto int32_field_logical_bytes(int32_t data_type, size_t count) -> size_t
		{
			switch (data_type)
			{
			case onnx::TensorProto::INT16:
			case onnx::TensorProto::UINT16:
			case onnx::TensorProto::FLOAT16:
			case onnx::TensorProto::BFLOAT16:
				return count * 2;
			case onnx::TensorProto::INT8:
			case onnx::TensorProto::UINT8:
			case onnx::TensorProto::BOOL:
			case onnx::TensorProto::FLOAT8E4M3FN:
			case onnx::TensorProto::FLOAT8E4M3FNUZ:
			case onnx::TensorProto::FLOAT8E5M2:
			case onnx::TensorProto::FLOAT8E5M2FNUZ:
				return count;
			case onnx::TensorProto::INT4:
			case onnx::TensorProto::UINT4:
				return count;
			default:
				return count * sizeof(int32_t);
			}
		}

		auto inline_byte_size(const onnx::TensorProto& tensor) -> size_t
		{
			if (tensor.has_raw_data())
			{
				return tensor.raw_data().size();
			}

			size_t bytes = 0;
			bytes += static_cast<size_t>(tensor.float_data_size()) * sizeof(float);
			bytes += int32_field_logical_bytes(tensor.data_type(), static_cast<size_t>(tensor.int32_data_size()));
			bytes += static_cast<size_t>(tensor.int64_data_size()) * sizeof(int64_t);
			bytes += static_cast<size_t>(tensor.double_data_size()) * sizeof(double);
			bytes += static_cast<size_t>(tensor.uint64_data_size()) * (tensor.data_type() == onnx::TensorProto::UINT32 ? sizeof(uint32_t) : sizeof(uint64_t));
			for (int i = 0; i < tensor.string_data_size(); ++i)
			{
				bytes += tensor.string_data(i).size();
			}
			return bytes;
		}

		auto is_valid_utf8(const std::string& text) -> bool
		{
			size_t i = 0;
			while (i < text.size())
			{
				const auto lead = static_cast<unsigned char>(text[i]);
				size_t length = 0;
				if (lead < 0x80)
				{
					length = 1;
				}
				else if ((lead >> 5) == 0x06)
				{
					length = 2;
				}
				else if ((lead >> 4) == 0x0E)
				{
					length = 3;
				}
				else if ((lead >> 3) == 0x1E)
				{
					length = 4;
				}
				else
				{
					return false;
				}
				if (i + length > text.size())
				{
					return false;
				}
				for (size_t j = 1; j < length; ++j)
				{
					if ((static_cast<unsigned char>(text[i + j]) >> 6) != 0x02)
					{
						return false;
					}
				}
				i += length;
			}
			return true;
		}

		auto printable_or_hex(const std::string& text) -> std::string
		{
			if (is_valid_utf8(text))
			{
				return text;
			}
			constexpr size_t hex_preview_limit = 16;
			std::string hex = "0x";
			for (size_t i = 0; i < text.size() && i < hex_preview_limit; ++i)
			{
				hex += std::format("{:02x}", static_cast<unsigned char>(text[i]));
			}
			if (text.size() > hex_preview_limit)
			{
				hex += "...";
			}
			return std::format("{} ({} bytes, non-UTF8)", hex, text.size());
		}

		auto escape_dot(const std::string& text) -> std::string
		{
			std::string out;
			out.reserve(text.size());
			for (char ch : text)
			{
				if (ch == '"' || ch == '\\')
				{
					out.push_back('\\');
				}
				out.push_back(ch);
			}
			return out;
		}

		auto human_bytes(size_t bytes) -> std::string
		{
			constexpr double kib = 1024.0;
			const double value = static_cast<double>(bytes);
			if (value >= kib * kib * kib)
			{
				return std::format("{:.2f} GiB", value / (kib * kib * kib));
			}
			if (value >= kib * kib)
			{
				return std::format("{:.2f} MiB", value / (kib * kib));
			}
			if (value >= kib)
			{
				return std::format("{:.2f} KiB", value / kib);
			}
			return std::format("{} B", bytes);
		}

		template <typename Container, typename Formatter> auto join_values(const Container& values, Formatter&& format_value) -> std::string
		{
			std::string out = "[";
			size_t index = 0;
			for (const auto& value : values)
			{
				if (index >= attribute_preview_limit)
				{
					out += std::format(", … ({} total)", static_cast<size_t>(values.size()));
					break;
				}
				out += std::format("{}{}", (index == 0 ? "" : ", "), format_value(value));
				++index;
			}
			out += "]";
			return out;
		}

		auto tensor_dims_string(const onnx::TensorProto& tensor) -> std::string
		{
			std::string joined;
			for (int i = 0; i < tensor.dims_size(); ++i)
			{
				joined += std::format("{}{}", (i == 0 ? "" : ", "), tensor.dims(i));
			}
			return std::format("[{}]", joined);
		}

		auto attribute_type_name(onnx::AttributeProto::AttributeType type) -> std::string
		{
			switch (type)
			{
			case onnx::AttributeProto::FLOAT:
				return "FLOAT";
			case onnx::AttributeProto::INT:
				return "INT";
			case onnx::AttributeProto::STRING:
				return "STRING";
			case onnx::AttributeProto::TENSOR:
				return "TENSOR";
			case onnx::AttributeProto::GRAPH:
				return "GRAPH";
			case onnx::AttributeProto::SPARSE_TENSOR:
				return "SPARSE_TENSOR";
			case onnx::AttributeProto::TYPE_PROTO:
				return "TYPE_PROTO";
			case onnx::AttributeProto::FLOATS:
				return "FLOATS";
			case onnx::AttributeProto::INTS:
				return "INTS";
			case onnx::AttributeProto::STRINGS:
				return "STRINGS";
			case onnx::AttributeProto::TENSORS:
				return "TENSORS";
			case onnx::AttributeProto::GRAPHS:
				return "GRAPHS";
			case onnx::AttributeProto::SPARSE_TENSORS:
				return "SPARSE_TENSORS";
			case onnx::AttributeProto::TYPE_PROTOS:
				return "TYPE_PROTOS";
			default:
				return "UNDEFINED";
			}
		}

		auto to_attribute_info(const onnx::AttributeProto& attribute) -> AttributeInfo
		{
			AttributeInfo info;
			info.name_ = attribute.name();
			info.type_ = attribute_type_name(attribute.type());

			if (!attribute.ref_attr_name().empty())
			{
				info.value_ = std::format("ref '{}'", attribute.ref_attr_name());
				return info;
			}

			switch (attribute.type())
			{
			case onnx::AttributeProto::FLOAT:
				info.floats_.push_back(static_cast<double>(attribute.f()));
				info.value_ = std::format("{}", attribute.f());
				break;
			case onnx::AttributeProto::INT:
				info.ints_.push_back(attribute.i());
				info.value_ = std::format("{}", attribute.i());
				break;
			case onnx::AttributeProto::STRING:
				info.strings_.push_back(printable_or_hex(attribute.s()));
				info.value_ = printable_or_hex(attribute.s());
				break;
			case onnx::AttributeProto::TENSOR:
				info.value_ = std::format("tensor {} {}", data_type_name(attribute.t().data_type()), tensor_dims_string(attribute.t()));
				break;
			case onnx::AttributeProto::GRAPH:
				info.value_ = std::format("graph '{}' ({} nodes)", attribute.g().name(), attribute.g().node_size());
				break;
			case onnx::AttributeProto::FLOATS:
				for (float value : attribute.floats())
				{
					info.floats_.push_back(static_cast<double>(value));
				}
				info.value_ = join_values(attribute.floats(), [](float value) { return std::format("{}", value); });
				break;
			case onnx::AttributeProto::INTS:
				for (int64_t value : attribute.ints())
				{
					info.ints_.push_back(value);
				}
				info.value_ = join_values(attribute.ints(), [](int64_t value) { return std::format("{}", value); });
				break;
			case onnx::AttributeProto::STRINGS:
				for (const auto& value : attribute.strings())
				{
					info.strings_.push_back(printable_or_hex(value));
				}
				info.value_ = join_values(attribute.strings(), [](const std::string& value) { return std::format("'{}'", printable_or_hex(value)); });
				break;
			case onnx::AttributeProto::TENSORS:
				info.value_ = std::format("{} tensors", attribute.tensors_size());
				break;
			case onnx::AttributeProto::GRAPHS:
			{
				size_t nested = 0;
				for (const auto& graph : attribute.graphs())
				{
					nested += static_cast<size_t>(graph.node_size());
				}
				info.value_ = std::format("{} graphs ({} nodes)", attribute.graphs_size(), nested);
				break;
			}
			case onnx::AttributeProto::SPARSE_TENSOR:
				info.value_ = "sparse tensor";
				break;
			case onnx::AttributeProto::SPARSE_TENSORS:
				info.value_ = std::format("{} sparse tensors", attribute.sparse_tensors_size());
				break;
			case onnx::AttributeProto::TYPE_PROTO:
				info.value_ = "type proto";
				break;
			case onnx::AttributeProto::TYPE_PROTOS:
				info.value_ = std::format("{} type protos", attribute.type_protos_size());
				break;
			default:
				break;
			}
			return info;
		}

		auto graph_node_count_recursive(const onnx::GraphProto& graph) -> size_t
		{
			size_t total = static_cast<size_t>(graph.node_size());
			for (const auto& node : graph.node())
			{
				for (const auto& attribute : node.attribute())
				{
					if (attribute.has_g())
					{
						total += graph_node_count_recursive(attribute.g());
					}
					for (const auto& nested : attribute.graphs())
					{
						total += graph_node_count_recursive(nested);
					}
				}
			}
			return total;
		}

		auto node_subgraph_count(const onnx::NodeProto& node) -> size_t
		{
			size_t total = 0;
			for (const auto& attribute : node.attribute())
			{
				if (attribute.has_g())
				{
					total += graph_node_count_recursive(attribute.g());
				}
				for (const auto& nested : attribute.graphs())
				{
					total += graph_node_count_recursive(nested);
				}
			}
			return total;
		}

		auto accumulate_operator_counts(const onnx::GraphProto& graph, std::map<std::string, size_t>& counts) -> void
		{
			for (const auto& node : graph.node())
			{
				++counts[node.op_type()];
				for (const auto& attribute : node.attribute())
				{
					if (attribute.has_g())
					{
						accumulate_operator_counts(attribute.g(), counts);
					}
					for (const auto& nested : attribute.graphs())
					{
						accumulate_operator_counts(nested, counts);
					}
				}
			}
		}

		auto initializer_data_json(const onnx::TensorProto& tensor) -> std::optional<boost::json::array>
		{
			namespace json = boost::json;
			json::array values;
			const std::string& raw = tensor.raw_data();
			switch (tensor.data_type())
			{
			case onnx::TensorProto::FLOAT:
				for (int i = 0; i < tensor.float_data_size(); ++i)
				{
					values.push_back(json::value(static_cast<double>(tensor.float_data(i))));
				}
				if (tensor.float_data_size() == 0 && tensor.has_raw_data())
				{
					for (size_t i = 0; i + sizeof(float) <= raw.size(); i += sizeof(float))
					{
						float v;
						std::memcpy(&v, raw.data() + i, sizeof(float));
						values.push_back(json::value(static_cast<double>(v)));
					}
				}
				return values;
			case onnx::TensorProto::DOUBLE:
				for (int i = 0; i < tensor.double_data_size(); ++i)
				{
					values.push_back(json::value(tensor.double_data(i)));
				}
				if (tensor.double_data_size() == 0 && tensor.has_raw_data())
				{
					for (size_t i = 0; i + sizeof(double) <= raw.size(); i += sizeof(double))
					{
						double v;
						std::memcpy(&v, raw.data() + i, sizeof(double));
						values.push_back(json::value(v));
					}
				}
				return values;
			case onnx::TensorProto::INT32:
				for (int i = 0; i < tensor.int32_data_size(); ++i)
				{
					values.push_back(json::value(static_cast<int64_t>(tensor.int32_data(i))));
				}
				if (tensor.int32_data_size() == 0 && tensor.has_raw_data())
				{
					for (size_t i = 0; i + sizeof(int32_t) <= raw.size(); i += sizeof(int32_t))
					{
						int32_t v;
						std::memcpy(&v, raw.data() + i, sizeof(int32_t));
						values.push_back(json::value(static_cast<int64_t>(v)));
					}
				}
				return values;
			case onnx::TensorProto::INT64:
				for (int i = 0; i < tensor.int64_data_size(); ++i)
				{
					values.push_back(json::value(tensor.int64_data(i)));
				}
				if (tensor.int64_data_size() == 0 && tensor.has_raw_data())
				{
					for (size_t i = 0; i + sizeof(int64_t) <= raw.size(); i += sizeof(int64_t))
					{
						int64_t v;
						std::memcpy(&v, raw.data() + i, sizeof(int64_t));
						values.push_back(json::value(v));
					}
				}
				return values;
			default:
				return std::nullopt;
			}
		}
	} // namespace

	auto data_type_name(int32_t data_type) -> std::string
	{
		switch (data_type)
		{
		case onnx::TensorProto::UNDEFINED:
			return "UNDEFINED";
		case onnx::TensorProto::FLOAT:
			return "FLOAT";
		case onnx::TensorProto::UINT8:
			return "UINT8";
		case onnx::TensorProto::INT8:
			return "INT8";
		case onnx::TensorProto::UINT16:
			return "UINT16";
		case onnx::TensorProto::INT16:
			return "INT16";
		case onnx::TensorProto::INT32:
			return "INT32";
		case onnx::TensorProto::INT64:
			return "INT64";
		case onnx::TensorProto::STRING:
			return "STRING";
		case onnx::TensorProto::BOOL:
			return "BOOL";
		case onnx::TensorProto::FLOAT16:
			return "FLOAT16";
		case onnx::TensorProto::DOUBLE:
			return "DOUBLE";
		case onnx::TensorProto::UINT32:
			return "UINT32";
		case onnx::TensorProto::UINT64:
			return "UINT64";
		case onnx::TensorProto::COMPLEX64:
			return "COMPLEX64";
		case onnx::TensorProto::COMPLEX128:
			return "COMPLEX128";
		case onnx::TensorProto::BFLOAT16:
			return "BFLOAT16";
		case onnx::TensorProto::FLOAT8E4M3FN:
			return "FLOAT8E4M3FN";
		case onnx::TensorProto::FLOAT8E4M3FNUZ:
			return "FLOAT8E4M3FNUZ";
		case onnx::TensorProto::FLOAT8E5M2:
			return "FLOAT8E5M2";
		case onnx::TensorProto::FLOAT8E5M2FNUZ:
			return "FLOAT8E5M2FNUZ";
		case onnx::TensorProto::UINT4:
			return "UINT4";
		case onnx::TensorProto::INT4:
			return "INT4";
		default:
			return std::format("UNKNOWN({})", data_type);
		}
	}

	auto data_type_id(const std::string& name) -> std::optional<int32_t>
	{
		std::string upper = name;
		std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });

		for (int32_t candidate = onnx::TensorProto::UNDEFINED; candidate <= onnx::TensorProto::INT4; ++candidate)
		{
			if (data_type_name(candidate) == upper)
			{
				return candidate;
			}
		}
		return std::nullopt;
	}

	auto OnnxModel::load(const std::string& path) -> std::tuple<std::optional<OnnxModel>, std::optional<std::string>>
	{
		// Utilities::File can throw (it touches the parent path on open); never let it escape.
		try
		{
			File file;
			if (auto opened = file.open(path, std::ios::in | std::ios::binary); !opened)
			{
				return { std::nullopt, std::format("cannot open '{}': {}", path, opened.error()) };
			}

			auto read = file.read_bytes();
			if (!read)
			{
				return { std::nullopt, std::format("cannot read '{}': {}", path, read.error()) };
			}

			return parse(read.value());
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("cannot load '{}': {}", path, e.what()) };
		}
	}

	auto OnnxModel::parse(const std::vector<uint8_t>& bytes) -> std::tuple<std::optional<OnnxModel>, std::optional<std::string>>
	{
		if (bytes.empty())
		{
			return { std::nullopt, "empty ONNX buffer" };
		}

		if (bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			return { std::nullopt, "ONNX buffer exceeds the 2GB protobuf message limit" };
		}

		try
		{
			OnnxModel model;
			if (!model.model_.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
			{
				return { std::nullopt, "failed to parse ONNX protobuf (not a valid ModelProto?)" };
			}

			return { std::move(model), std::nullopt };
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("failed to parse ONNX protobuf: {}", e.what()) };
		}
	}

	auto OnnxModel::metadata(void) const -> ModelMetadata
	{
		ModelMetadata meta;
		meta.ir_version_ = model_.ir_version();
		meta.producer_name_ = model_.producer_name();
		meta.producer_version_ = model_.producer_version();
		meta.domain_ = model_.domain();
		meta.model_version_ = model_.model_version();
		meta.doc_string_ = model_.doc_string();
		if (model_.has_graph())
		{
			meta.graph_name_ = model_.graph().name();
		}

		for (int i = 0; i < model_.opset_import_size(); ++i)
		{
			const auto& opset = model_.opset_import(i);
			meta.opset_imports_.push_back(OpsetInfo{ opset.domain(), opset.version() });
		}

		for (int i = 0; i < model_.metadata_props_size(); ++i)
		{
			const auto& entry = model_.metadata_props(i);
			meta.metadata_props_.push_back(MetadataEntry{ entry.key(), entry.value() });
		}

		meta.function_count_ = static_cast<size_t>(model_.functions_size());
		return meta;
	}

	auto OnnxModel::inputs(void) const -> std::vector<ValueInfo>
	{
		std::vector<ValueInfo> result;
		if (!model_.has_graph())
		{
			return result;
		}

		const auto& graph = model_.graph();
		for (int i = 0; i < graph.input_size(); ++i)
		{
			result.push_back(to_value_info(graph.input(i)));
		}
		return result;
	}

	auto OnnxModel::outputs(void) const -> std::vector<ValueInfo>
	{
		std::vector<ValueInfo> result;
		if (!model_.has_graph())
		{
			return result;
		}

		const auto& graph = model_.graph();
		for (int i = 0; i < graph.output_size(); ++i)
		{
			result.push_back(to_value_info(graph.output(i)));
		}
		return result;
	}

	auto OnnxModel::nodes(void) const -> std::vector<NodeInfo>
	{
		std::vector<NodeInfo> result;
		if (!model_.has_graph())
		{
			return result;
		}

		const auto& graph = model_.graph();
		for (int i = 0; i < graph.node_size(); ++i)
		{
			const auto& node = graph.node(i);
			NodeInfo info;
			info.name_ = node.name();
			info.op_type_ = node.op_type();
			info.domain_ = node.domain();
			for (int j = 0; j < node.input_size(); ++j)
			{
				info.inputs_.push_back(node.input(j));
			}
			for (int j = 0; j < node.output_size(); ++j)
			{
				info.outputs_.push_back(node.output(j));
			}
			for (int j = 0; j < node.attribute_size(); ++j)
			{
				info.attributes_.push_back(to_attribute_info(node.attribute(j)));
			}
			info.subgraph_node_count_ = node_subgraph_count(node);
			result.push_back(std::move(info));
		}
		return result;
	}

	auto OnnxModel::initializers(void) const -> std::vector<TensorInfo>
	{
		std::vector<TensorInfo> result;
		if (!model_.has_graph())
		{
			return result;
		}

		const auto& graph = model_.graph();
		for (int i = 0; i < graph.initializer_size(); ++i)
		{
			const auto& tensor = graph.initializer(i);
			TensorInfo info;
			info.name_ = tensor.name();
			info.data_type_ = data_type_name(tensor.data_type());
			for (int j = 0; j < tensor.dims_size(); ++j)
			{
				info.dims_.push_back(tensor.dims(j));
			}

			if (tensor.has_data_location() && tensor.data_location() == onnx::TensorProto::EXTERNAL)
			{
				info.source_ = TensorDataSource::External;
				info.byte_size_ = 0;
				for (const auto& entry : tensor.external_data())
				{
					info.external_data_.push_back(MetadataEntry{ entry.key(), entry.value() });
					if (entry.key() == "length")
					{
						int64_t length = 0;
						auto [ptr, ec] = std::from_chars(entry.value().data(), entry.value().data() + entry.value().size(), length);
						if (ec == std::errc() && length > 0)
						{
							info.byte_size_ = static_cast<size_t>(length);
						}
					}
				}
			}
			else
			{
				info.source_ = TensorDataSource::Inline;
				info.byte_size_ = inline_byte_size(tensor);
			}
			result.push_back(std::move(info));
		}
		return result;
	}

	auto OnnxModel::operator_histogram(void) const -> std::vector<std::pair<std::string, size_t>>
	{
		std::map<std::string, size_t> counts;
		if (model_.has_graph())
		{
			accumulate_operator_counts(model_.graph(), counts);
		}

		std::vector<std::pair<std::string, size_t>> result(counts.begin(), counts.end());
		std::sort(result.begin(), result.end(),
				  [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b)
				  {
					  if (a.second != b.second)
					  {
						  return a.second > b.second;
					  }
					  return a.first < b.first;
				  });
		return result;
	}

	auto OnnxModel::proto(void) const -> const onnx::ModelProto& { return model_; }

	auto OnnxModel::to_json(bool include_initializer_data) const -> std::string
	{
		namespace json = boost::json;
		json::object root;

		const auto meta = metadata();
		json::object meta_obj;
		meta_obj["ir_version"] = meta.ir_version_;
		meta_obj["producer_name"] = printable_or_hex(meta.producer_name_);
		meta_obj["producer_version"] = printable_or_hex(meta.producer_version_);
		meta_obj["domain"] = printable_or_hex(meta.domain_);
		meta_obj["model_version"] = meta.model_version_;
		meta_obj["doc_string"] = printable_or_hex(meta.doc_string_);
		meta_obj["graph_name"] = printable_or_hex(meta.graph_name_);
		meta_obj["function_count"] = static_cast<int64_t>(meta.function_count_);

		json::array opsets;
		for (const auto& opset : meta.opset_imports_)
		{
			json::object entry;
			entry["domain"] = printable_or_hex(opset.domain_);
			entry["version"] = opset.version_;
			opsets.push_back(std::move(entry));
		}
		meta_obj["opset_import"] = std::move(opsets);

		json::array props;
		for (const auto& prop : meta.metadata_props_)
		{
			json::object entry;
			entry["key"] = printable_or_hex(prop.key_);
			entry["value"] = printable_or_hex(prop.value_);
			props.push_back(std::move(entry));
		}
		meta_obj["metadata_props"] = std::move(props);
		root["metadata"] = std::move(meta_obj);

		const auto value_array = [](const std::vector<ValueInfo>& values) -> json::array
		{
			json::array arr;
			for (const auto& value : values)
			{
				json::object entry;
				entry["name"] = printable_or_hex(value.name_);
				entry["data_type"] = value.data_type_;
				json::array shape;
				for (const auto& dim : value.shape_)
				{
					shape.push_back(json::value(dim));
				}
				entry["shape"] = std::move(shape);
				arr.push_back(std::move(entry));
			}
			return arr;
		};
		root["inputs"] = value_array(inputs());
		root["outputs"] = value_array(outputs());

		json::array operators;
		for (const auto& [name, count] : operator_histogram())
		{
			json::object entry;
			entry["op_type"] = name;
			entry["count"] = static_cast<int64_t>(count);
			operators.push_back(std::move(entry));
		}
		root["operators"] = std::move(operators);

		const auto node_list = nodes();
		size_t subgraph_nodes = 0;

		json::array node_array;
		for (const auto& node : node_list)
		{
			subgraph_nodes += node.subgraph_node_count_;

			json::object entry;
			entry["name"] = printable_or_hex(node.name_);
			entry["op_type"] = printable_or_hex(node.op_type_);
			entry["domain"] = printable_or_hex(node.domain_);

			json::array ins;
			for (const auto& name : node.inputs_)
			{
				ins.push_back(json::value(printable_or_hex(name)));
			}
			entry["inputs"] = std::move(ins);

			json::array outs;
			for (const auto& name : node.outputs_)
			{
				outs.push_back(json::value(printable_or_hex(name)));
			}
			entry["outputs"] = std::move(outs);

			json::array attrs;
			for (const auto& attribute : node.attributes_)
			{
				json::object attr;
				attr["name"] = attribute.name_;
				attr["type"] = attribute.type_;
				if (attribute.type_ == "INT" && attribute.ints_.size() == 1)
				{
					attr["value"] = attribute.ints_[0];
				}
				else if (attribute.type_ == "FLOAT" && attribute.floats_.size() == 1)
				{
					attr["value"] = attribute.floats_[0];
				}
				else if (attribute.type_ == "STRING" && attribute.strings_.size() == 1)
				{
					attr["value"] = attribute.strings_[0];
				}
				else if (attribute.type_ == "INTS")
				{
					json::array values;
					for (int64_t value : attribute.ints_)
					{
						values.push_back(json::value(value));
					}
					attr["value"] = std::move(values);
				}
				else if (attribute.type_ == "FLOATS")
				{
					json::array values;
					for (double value : attribute.floats_)
					{
						values.push_back(json::value(value));
					}
					attr["value"] = std::move(values);
				}
				else if (attribute.type_ == "STRINGS")
				{
					json::array values;
					for (const auto& value : attribute.strings_)
					{
						values.push_back(json::value(value));
					}
					attr["value"] = std::move(values);
				}
				else
				{
					attr["value"] = attribute.value_;
				}
				attrs.push_back(std::move(attr));
			}
			entry["attributes"] = std::move(attrs);

			if (node.subgraph_node_count_ > 0)
			{
				entry["subgraph_node_count"] = static_cast<int64_t>(node.subgraph_node_count_);
			}
			node_array.push_back(std::move(entry));
		}
		root["nodes"] = std::move(node_array);
		root["subgraph_node_count"] = static_cast<int64_t>(subgraph_nodes);

		json::array init_array;
		const auto init_list = initializers();
		size_t total_parameters = 0;
		size_t total_initializer_bytes = 0;
		for (size_t i = 0; i < init_list.size(); ++i)
		{
			const auto& tensor = init_list[i];
			total_parameters += tensor.element_count();
			total_initializer_bytes += tensor.byte_size_;

			json::object entry;
			entry["name"] = printable_or_hex(tensor.name_);
			entry["data_type"] = tensor.data_type_;
			json::array dims;
			for (auto dim : tensor.dims_)
			{
				dims.push_back(json::value(dim));
			}
			entry["dims"] = std::move(dims);
			entry["source"] = (tensor.source_ == TensorDataSource::External) ? "external" : "inline";
			entry["byte_size"] = static_cast<int64_t>(tensor.byte_size_);

			if (!tensor.external_data_.empty())
			{
				json::object external;
				for (const auto& kv : tensor.external_data_)
				{
					external[printable_or_hex(kv.key_)] = printable_or_hex(kv.value_);
				}
				entry["external_data"] = std::move(external);
			}

			if (include_initializer_data && model_.has_graph())
			{
				const auto& proto = model_.graph().initializer(static_cast<int>(i));
				if (proto.has_data_location() && proto.data_location() == onnx::TensorProto::EXTERNAL)
				{
					entry["data_omitted"] = "external data (not loaded)";
				}
				else if (tensor.element_count() > weight_export_element_limit)
				{
					entry["data_omitted"] = std::format("too large to export ({} elements > {} limit)", tensor.element_count(), weight_export_element_limit);
				}
				else if (auto data = initializer_data_json(proto); !data.has_value())
				{
					entry["data_omitted"] = "value export unsupported for this dtype";
				}
				else if (data->empty() && tensor.element_count() > 0)
				{
					entry["data_omitted"] = "no inline data";
				}
				else
				{
					entry["data"] = std::move(data.value());
				}
			}
			init_array.push_back(std::move(entry));
		}
		root["initializers"] = std::move(init_array);
		root["total_parameters"] = static_cast<int64_t>(total_parameters);
		root["total_initializer_bytes"] = static_cast<int64_t>(total_initializer_bytes);

		return JsonTool::pretty_format(root);
	}

	auto OnnxModel::to_dot(void) const -> std::string
	{
		const auto node_list = nodes();
		const auto input_list = inputs();
		const auto output_list = outputs();

		std::unordered_set<std::string> initializer_names;
		for (const auto& tensor : initializers())
		{
			initializer_names.insert(tensor.name_);
		}

		const auto value_label = [](const ValueInfo& value) -> std::string
		{
			std::string label = escape_dot(value.name_.empty() ? "?" : value.name_);
			if (!value.data_type_.empty())
			{
				std::string shape;
				for (size_t i = 0; i < value.shape_.size(); ++i)
				{
					shape += std::format("{}{}", (i == 0 ? "" : ", "), value.shape_[i]);
				}
				label += std::format("\\n{} [{}]", escape_dot(value.data_type_), escape_dot(shape));
			}
			return label;
		};

		std::ostringstream out;
		out << "digraph onnx_model\n{\n";
		out << "\trankdir=TB;\n";
		out << "\tnode [shape=box, style=rounded];\n";

		std::unordered_map<std::string, std::string> input_node_ids;
		for (size_t i = 0; i < input_list.size(); ++i)
		{
			if (initializer_names.contains(input_list[i].name_))
			{
				continue;
			}
			out << std::format("\tgi{} [label=\"{}\", shape=ellipse, style=filled, fillcolor=lightblue];\n", i, value_label(input_list[i]));
			input_node_ids[input_list[i].name_] = std::format("gi{}", i);
		}

		for (size_t i = 0; i < node_list.size(); ++i)
		{
			const auto& node = node_list[i];
			// Escape parts separately so the "\n" DOT line-break is not itself escaped.
			std::string label = escape_dot(node.op_type_.empty() ? "?" : node.op_type_);
			if (!node.name_.empty())
			{
				label += "\\n" + escape_dot(node.name_);
			}
			out << std::format("\tn{} [label=\"{}\"];\n", i, label);
		}

		for (size_t i = 0; i < output_list.size(); ++i)
		{
			out << std::format("\tgo{} [label=\"{}\", shape=ellipse, style=filled, fillcolor=palegreen];\n", i, value_label(output_list[i]));
		}

		std::unordered_map<std::string, size_t> producer;
		for (size_t i = 0; i < node_list.size(); ++i)
		{
			for (const auto& output : node_list[i].outputs_)
			{
				if (output.empty())
				{
					continue;
				}
				producer[output] = i;
			}
		}

		for (size_t i = 0; i < node_list.size(); ++i)
		{
			for (const auto& input : node_list[i].inputs_)
			{
				if (input.empty())
				{
					continue;
				}
				if (auto found = producer.find(input); found != producer.end())
				{
					out << std::format("\tn{} -> n{} [label=\"{}\"];\n", found->second, i, escape_dot(input));
				}
				else if (auto graph_input = input_node_ids.find(input); graph_input != input_node_ids.end())
				{
					out << std::format("\t{} -> n{} [label=\"{}\"];\n", graph_input->second, i, escape_dot(input));
				}
			}
		}

		for (size_t i = 0; i < output_list.size(); ++i)
		{
			if (auto found = producer.find(output_list[i].name_); found != producer.end())
			{
				out << std::format("\tn{} -> go{} [label=\"{}\"];\n", found->second, i, escape_dot(output_list[i].name_));
			}
		}

		out << "}\n";
		return out.str();
	}

	auto OnnxModel::to_summary(void) const -> std::string
	{
		const auto meta = metadata();
		const auto node_list = nodes();
		const auto init_list = initializers();

		size_t subgraph_nodes = 0;
		for (const auto& node : node_list)
		{
			subgraph_nodes += node.subgraph_node_count_;
		}

		size_t total_parameters = 0;
		size_t total_initializer_bytes = 0;
		for (const auto& tensor : init_list)
		{
			total_parameters += tensor.element_count();
			total_initializer_bytes += tensor.byte_size_;
		}

		std::ostringstream out;
		out << "ONNX model summary\n";
		out << std::format("  ir_version    : {}\n", meta.ir_version_);
		out << std::format("  producer      : {} {}\n", meta.producer_name_, meta.producer_version_);
		out << std::format("  domain        : {}\n", meta.domain_.empty() ? "(default)" : meta.domain_);
		out << std::format("  model_version : {}\n", meta.model_version_);
		out << std::format("  graph         : {}\n", meta.graph_name_);
		out << "  opset_import  :\n";
		for (const auto& opset : meta.opset_imports_)
		{
			out << std::format("      {} v{}\n", opset.domain_.empty() ? "ai.onnx" : opset.domain_, opset.version_);
		}
		out << std::format("  nodes         : {}{}\n", node_list.size(), (subgraph_nodes > 0) ? std::format(" (+{} in subgraphs)", subgraph_nodes) : "");
		out << std::format("  initializers  : {}\n", init_list.size());
		out << std::format("  parameters    : {} ({})\n", total_parameters, human_bytes(total_initializer_bytes));
		out << std::format("  inputs        : {}\n", inputs().size());
		out << std::format("  outputs       : {}\n", outputs().size());
		out << "  operators     :\n";
		for (const auto& [name, count] : operator_histogram())
		{
			out << std::format("      {:>4}  {}\n", count, name);
		}
		return out.str();
	}
} // namespace YirangOnnx
