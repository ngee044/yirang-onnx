#include "OnnxModel.h"

#include "File.h"
#include "JsonTool.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <boost/json.hpp>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
		auto to_value_info(const onnx::ValueInfoProto& value) -> ValueInfo
		{
			ValueInfo info;
			info.name_ = value.name();
			if (value.has_type() && value.type().has_tensor_type())
			{
				const auto& tensor = value.type().tensor_type();
				info.data_type_ = data_type_name(tensor.elem_type());
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
			return info;
		}

		auto inline_byte_size(const onnx::TensorProto& tensor) -> size_t
		{
			if (tensor.has_raw_data())
			{
				return tensor.raw_data().size();
			}

			size_t bytes = 0;
			bytes += static_cast<size_t>(tensor.float_data_size()) * sizeof(float);
			bytes += static_cast<size_t>(tensor.int32_data_size()) * sizeof(int32_t);
			bytes += static_cast<size_t>(tensor.int64_data_size()) * sizeof(int64_t);
			bytes += static_cast<size_t>(tensor.double_data_size()) * sizeof(double);
			bytes += static_cast<size_t>(tensor.uint64_data_size()) * sizeof(uint64_t);
			for (int i = 0; i < tensor.string_data_size(); ++i)
			{
				bytes += tensor.string_data(i).size();
			}
			return bytes;
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
	}

	auto data_type_name(int32_t data_type) -> std::string
	{
		switch (data_type)
		{
		case 0:
			return "UNDEFINED";
		case 1:
			return "FLOAT";
		case 2:
			return "UINT8";
		case 3:
			return "INT8";
		case 4:
			return "UINT16";
		case 5:
			return "INT16";
		case 6:
			return "INT32";
		case 7:
			return "INT64";
		case 8:
			return "STRING";
		case 9:
			return "BOOL";
		case 10:
			return "FLOAT16";
		case 11:
			return "DOUBLE";
		case 12:
			return "UINT32";
		case 13:
			return "UINT64";
		case 14:
			return "COMPLEX64";
		case 15:
			return "COMPLEX128";
		case 16:
			return "BFLOAT16";
		case 17:
			return "FLOAT8E4M3FN";
		case 18:
			return "FLOAT8E4M3FNUZ";
		case 19:
			return "FLOAT8E5M2";
		case 20:
			return "FLOAT8E5M2FNUZ";
		case 21:
			return "UINT4";
		case 22:
			return "INT4";
		default:
			return std::format("UNKNOWN({})", data_type);
		}
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

		OnnxModel model;
		if (!model.model_.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
		{
			return { std::nullopt, "failed to parse ONNX protobuf (not a valid ModelProto?)" };
		}

		return { std::move(model), std::nullopt };
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
				info.attribute_names_.push_back(node.attribute(j).name());
			}
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
			const auto& graph = model_.graph();
			for (int i = 0; i < graph.node_size(); ++i)
			{
				++counts[graph.node(i).op_type()];
			}
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

	auto OnnxModel::to_json(void) const -> std::string
	{
		namespace json = boost::json;
		json::object root;

		const auto meta = metadata();
		json::object meta_obj;
		meta_obj["ir_version"] = meta.ir_version_;
		meta_obj["producer_name"] = meta.producer_name_;
		meta_obj["producer_version"] = meta.producer_version_;
		meta_obj["domain"] = meta.domain_;
		meta_obj["model_version"] = meta.model_version_;
		meta_obj["doc_string"] = meta.doc_string_;
		meta_obj["graph_name"] = meta.graph_name_;
		meta_obj["function_count"] = static_cast<int64_t>(meta.function_count_);

		json::array opsets;
		for (const auto& opset : meta.opset_imports_)
		{
			json::object entry;
			entry["domain"] = opset.domain_;
			entry["version"] = opset.version_;
			opsets.push_back(std::move(entry));
		}
		meta_obj["opset_import"] = std::move(opsets);

		json::array props;
		for (const auto& prop : meta.metadata_props_)
		{
			json::object entry;
			entry["key"] = prop.key_;
			entry["value"] = prop.value_;
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
				entry["name"] = value.name_;
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

		json::array node_array;
		for (const auto& node : nodes())
		{
			json::object entry;
			entry["name"] = node.name_;
			entry["op_type"] = node.op_type_;
			entry["domain"] = node.domain_;

			json::array ins;
			for (const auto& name : node.inputs_)
			{
				ins.push_back(json::value(name));
			}
			entry["inputs"] = std::move(ins);

			json::array outs;
			for (const auto& name : node.outputs_)
			{
				outs.push_back(json::value(name));
			}
			entry["outputs"] = std::move(outs);

			json::array attrs;
			for (const auto& name : node.attribute_names_)
			{
				attrs.push_back(json::value(name));
			}
			entry["attributes"] = std::move(attrs);
			node_array.push_back(std::move(entry));
		}
		root["nodes"] = std::move(node_array);

		json::array init_array;
		for (const auto& tensor : initializers())
		{
			json::object entry;
			entry["name"] = tensor.name_;
			entry["data_type"] = tensor.data_type_;
			json::array dims;
			for (auto dim : tensor.dims_)
			{
				dims.push_back(json::value(dim));
			}
			entry["dims"] = std::move(dims);
			entry["source"] = (tensor.source_ == TensorDataSource::External) ? "external" : "inline";
			entry["byte_size"] = static_cast<int64_t>(tensor.byte_size_);
			init_array.push_back(std::move(entry));
		}
		root["initializers"] = std::move(init_array);

		return JsonTool::pretty_format(root);
	}

	auto OnnxModel::to_dot(void) const -> std::string
	{
		const auto node_list = nodes();

		std::ostringstream out;
		out << "digraph onnx_model\n{\n";
		out << "\trankdir=TB;\n";
		out << "\tnode [shape=box, style=rounded];\n";

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

		std::unordered_map<std::string, size_t> producer;
		for (size_t i = 0; i < node_list.size(); ++i)
		{
			for (const auto& output : node_list[i].outputs_)
			{
				producer[output] = i;
			}
		}

		for (size_t i = 0; i < node_list.size(); ++i)
		{
			for (const auto& input : node_list[i].inputs_)
			{
				auto found = producer.find(input);
				if (found != producer.end())
				{
					out << std::format("\tn{} -> n{} [label=\"{}\"];\n", found->second, i, escape_dot(input));
				}
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
		out << std::format("  nodes         : {}\n", node_list.size());
		out << std::format("  initializers  : {}\n", init_list.size());
		out << std::format("  inputs        : {}\n", inputs().size());
		out << std::format("  outputs       : {}\n", outputs().size());
		out << "  operators     :\n";
		for (const auto& [name, count] : operator_histogram())
		{
			out << std::format("      {:>4}  {}\n", count, name);
		}
		return out.str();
	}
}
