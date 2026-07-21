#include "TensorConvert.h"

#include "ModelTypes.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>

namespace YirangOnnx
{
	namespace
	{
		auto element_byte_width(int32_t elem_type) -> size_t
		{
			switch (elem_type)
			{
			case onnx::TensorProto::FLOAT:
				return sizeof(float);
			case onnx::TensorProto::DOUBLE:
				return sizeof(double);
			case onnx::TensorProto::INT32:
				return sizeof(int32_t);
			case onnx::TensorProto::INT64:
				return sizeof(int64_t);
			case onnx::TensorProto::BOOL:
				return 1;
			default:
				return 0;
			}
		}

		auto verify_data_size(const onnx::TensorProto& proto, const Tensor& tensor) -> std::optional<std::string>
		{
			const size_t width = element_byte_width(proto.data_type());
			if (width == 0)
			{
				return std::nullopt;
			}

			size_t count = 1;
			for (int i = 0; i < proto.dims_size(); ++i)
			{
				const int64_t dim = proto.dims(i);
				if (dim < 0)
				{
					return std::format("tensor '{}': negative dimension {}", proto.name(), dim);
				}
				const size_t magnitude = static_cast<size_t>(dim);
				if (magnitude != 0 && count > std::numeric_limits<size_t>::max() / magnitude)
				{
					return std::format("tensor '{}': dimension product overflows", proto.name());
				}
				count *= magnitude;
			}

			if (tensor.data_.size() != count * width)
			{
				return std::format("tensor '{}': data size {} bytes does not match shape ({} elements x {} bytes)", proto.name(), tensor.data_.size(), count, width);
			}
			return std::nullopt;
		}
	} // namespace

	auto tensor_from_proto(const onnx::TensorProto& proto) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>
	{
		if (proto.has_data_location() && proto.data_location() == onnx::TensorProto::EXTERNAL)
		{
			return { std::nullopt, std::format("tensor '{}': external data is not loaded (provide values in raw_data)", proto.name()) };
		}

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
			if (auto mismatch = verify_data_size(proto, tensor); mismatch.has_value())
			{
				return { std::nullopt, mismatch };
			}
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
		if (auto mismatch = verify_data_size(proto, tensor); mismatch.has_value())
		{
			return { std::nullopt, mismatch };
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
