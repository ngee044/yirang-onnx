#include "RunCommand.h"

#include "File.h"
#include "InferenceEngine.h"
#include "Logger.h"
#include "ModelTypes.h"
#include "Tensor.h"

#include "onnx.pb.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
		auto tensor_from_proto(const onnx::TensorProto& proto) -> Tensor
		{
			Tensor tensor;
			tensor.name_ = proto.name();
			tensor.elem_type_ = proto.data_type();
			for (int i = 0; i < proto.dims_size(); ++i)
			{
				tensor.shape_.push_back(proto.dims(i));
			}

			if (proto.has_raw_data())
			{
				const std::string& raw = proto.raw_data();
				tensor.data_.assign(raw.begin(), raw.end());
				return tensor;
			}

			const auto append = [&tensor](const void* source, size_t bytes)
			{
				const uint8_t* begin = reinterpret_cast<const uint8_t*>(source);
				tensor.data_.insert(tensor.data_.end(), begin, begin + bytes);
			};
			switch (proto.data_type())
			{
			case 1: // FLOAT
				for (int i = 0; i < proto.float_data_size(); ++i)
				{
					float v = proto.float_data(i);
					append(&v, sizeof(v));
				}
				break;
			case 11: // DOUBLE
				for (int i = 0; i < proto.double_data_size(); ++i)
				{
					double v = proto.double_data(i);
					append(&v, sizeof(v));
				}
				break;
			case 6: // INT32
				for (int i = 0; i < proto.int32_data_size(); ++i)
				{
					int32_t v = proto.int32_data(i);
					append(&v, sizeof(v));
				}
				break;
			case 7: // INT64
				for (int i = 0; i < proto.int64_data_size(); ++i)
				{
					int64_t v = proto.int64_data(i);
					append(&v, sizeof(v));
				}
				break;
			default:
				break;
			}
			return tensor;
		}

		auto proto_from_tensor(const Tensor& tensor) -> onnx::TensorProto
		{
			onnx::TensorProto proto;
			proto.set_name(tensor.name_);
			proto.set_data_type(tensor.elem_type_);
			for (int64_t dim : tensor.shape_)
			{
				proto.add_dims(dim);
			}
			proto.set_raw_data(tensor.data_.data(), tensor.data_.size());
			return proto;
		}

		auto shape_string(const std::vector<int64_t>& shape) -> std::string
		{
			std::string out = "[";
			for (size_t i = 0; i < shape.size(); ++i)
			{
				out += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
			}
			out += "]";
			return out;
		}
	}

	auto run_inference(const std::string& model_path, const std::vector<std::string>& input_paths, const std::string& output_dir) -> int
	{
		std::vector<Tensor> inputs;
		for (const auto& path : input_paths)
		{
			File file;
			if (auto opened = file.open(path, std::ios::in | std::ios::binary); !opened)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot open input tensor '{}': {}", path, opened.error()));
				return 1;
			}

			auto bytes = file.read_bytes();
			if (!bytes)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot read input tensor '{}': {}", path, bytes.error()));
				return 1;
			}

			onnx::TensorProto proto;
			if (!proto.ParseFromArray(bytes.value().data(), static_cast<int>(bytes.value().size())))
			{
				Logger::handle().write(LogTypes::Error, std::format("'{}' is not a valid TensorProto", path));
				return 1;
			}
			if (proto.name().empty())
			{
				Logger::handle().write(LogTypes::Error, std::format("input tensor '{}' has no name (must match a graph input)", path));
				return 1;
			}
			inputs.push_back(tensor_from_proto(proto));
		}

		const InferenceEngine engine;
		auto [outputs, error] = engine.run(model_path, inputs);
		if (!outputs.has_value())
		{
			Logger::handle().write(LogTypes::Error, error.value_or("inference failed"));
			return 1;
		}

		// Utilities::File::open creates the parent directory, so no explicit mkdir.
		const std::filesystem::path dir = output_dir.empty() ? std::filesystem::path(".") : std::filesystem::path(output_dir);

		for (const auto& output : outputs.value())
		{
			std::string safe_name = output.name_;
			for (char& ch : safe_name)
			{
				if (ch == '/' || ch == '\\')
				{
					ch = '_';
				}
			}
			const std::filesystem::path out_path = dir / (safe_name + ".pb");

			const onnx::TensorProto proto = proto_from_tensor(output);
			std::string serialized;
			proto.SerializeToString(&serialized);

			File out;
			if (auto opened = out.open(out_path.string(), std::ios::out | std::ios::binary | std::ios::trunc); !opened)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), opened.error()));
				return 1;
			}
			if (auto written = out.write_bytes(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()); !written)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot write output '{}': {}", out_path.string(), written.error()));
				return 1;
			}

			std::cout << "output " << output.name_ << " " << shape_string(output.shape_) << " " << data_type_name(output.elem_type_) << " -> " << out_path.string()
					  << '\n';
		}
		return 0;
	}
}
