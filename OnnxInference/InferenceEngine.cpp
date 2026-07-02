#include "InferenceEngine.h"

#include <format>

#include <onnxruntime_cxx_api.h>

namespace YirangOnnx
{
	namespace
	{
		auto element_size(int32_t elem_type) -> size_t
		{
			switch (elem_type)
			{
			case 1:		  // FLOAT
			case 6:		  // INT32
			case 12:	  // UINT32
				return 4;
			case 2:		  // UINT8
			case 3:		  // INT8
			case 9:		  // BOOL
				return 1;
			case 4:		  // UINT16
			case 5:		  // INT16
			case 10:	  // FLOAT16
			case 16:	  // BFLOAT16
				return 2;
			case 7:		  // INT64
			case 11:	  // DOUBLE
			case 13:	  // UINT64
				return 8;
			default:
				return 0; // unsupported (e.g. STRING)
			}
		}
	}

	auto InferenceEngine::run(const std::string& model_path, const std::vector<Tensor>& inputs) const
		-> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>
	{
		// ORT's ONNXTensorElementDataType enum shares the numbering of ONNX
		// TensorProto.DataType, so elem_type_ casts across directly.
		try
		{
			Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yirang-onnx");
			Ort::SessionOptions session_options;
			Ort::Session session(env, model_path.c_str(), session_options);
			Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			std::vector<Ort::Value> input_values;
			std::vector<const char*> input_names;
			input_values.reserve(inputs.size());
			input_names.reserve(inputs.size());
			for (const auto& tensor : inputs)
			{
				if (element_size(tensor.elem_type_) == 0)
				{
					return { std::nullopt, std::format("input '{}': unsupported data_type {}", tensor.name_, tensor.elem_type_) };
				}
				input_values.push_back(Ort::Value::CreateTensor(memory_info, const_cast<uint8_t*>(tensor.data_.data()), tensor.data_.size(), tensor.shape_.data(),
																tensor.shape_.size(), static_cast<ONNXTensorElementDataType>(tensor.elem_type_)));
				input_names.push_back(tensor.name_.c_str());
			}

			Ort::AllocatorWithDefaultOptions allocator;
			const size_t output_count = session.GetOutputCount();
			std::vector<std::string> output_names_str;
			std::vector<const char*> output_names;
			for (size_t i = 0; i < output_count; ++i)
			{
				output_names_str.push_back(session.GetOutputNameAllocated(i, allocator).get());
			}
			for (const auto& name : output_names_str)
			{
				output_names.push_back(name.c_str());
			}

			auto results
				= session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_values.data(), input_values.size(), output_names.data(), output_names.size());

			std::vector<Tensor> outputs;
			outputs.reserve(results.size());
			for (size_t i = 0; i < results.size(); ++i)
			{
				const auto info = results[i].GetTensorTypeAndShapeInfo();
				const auto shape = info.GetShape();
				const int32_t elem_type = static_cast<int32_t>(info.GetElementType());
				const size_t elem = element_size(elem_type);

				Tensor out;
				out.name_ = output_names_str[i];
				out.elem_type_ = elem_type;
				out.shape_.assign(shape.begin(), shape.end());
				const size_t bytes = info.GetElementCount() * elem;
				const uint8_t* data = results[i].GetTensorData<uint8_t>();
				out.data_.assign(data, data + bytes);
				outputs.push_back(std::move(out));
			}
			return { std::move(outputs), std::nullopt };
		}
		catch (const Ort::Exception& e)
		{
			return { std::nullopt, std::format("onnxruntime error: {}", e.what()) };
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("inference failed: {}", e.what()) };
		}
	}
}
