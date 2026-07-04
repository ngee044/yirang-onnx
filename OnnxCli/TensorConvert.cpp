#include "TensorConvert.h"

#include "ModelTypes.h"

#include <cstdint>
#include <format>
#include <string>

namespace YirangOnnx
{
	auto tensor_from_proto(const onnx::TensorProto& proto) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>
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
			return { std::move(tensor), std::nullopt };
		}

		const auto append = [&tensor](const void* source, size_t bytes)
		{
			const uint8_t* begin = reinterpret_cast<const uint8_t*>(source);
			tensor.data_.insert(tensor.data_.end(), begin, begin + bytes);
		};
		switch (proto.data_type())
		{
		case onnx::TensorProto::FLOAT:
			for (int i = 0; i < proto.float_data_size(); ++i)
			{
				float v = proto.float_data(i);
				append(&v, sizeof(v));
			}
			break;
		case onnx::TensorProto::DOUBLE:
			for (int i = 0; i < proto.double_data_size(); ++i)
			{
				double v = proto.double_data(i);
				append(&v, sizeof(v));
			}
			break;
		case onnx::TensorProto::INT32:
			for (int i = 0; i < proto.int32_data_size(); ++i)
			{
				int32_t v = proto.int32_data(i);
				append(&v, sizeof(v));
			}
			break;
		case onnx::TensorProto::INT64:
			for (int i = 0; i < proto.int64_data_size(); ++i)
			{
				int64_t v = proto.int64_data(i);
				append(&v, sizeof(v));
			}
			break;
		default:
			return { std::nullopt,
					 std::format("tensor '{}': typed-data decoding unsupported for {} (store values in raw_data)", proto.name(), data_type_name(proto.data_type())) };
		}
		return { std::move(tensor), std::nullopt };
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
} // namespace YirangOnnx
