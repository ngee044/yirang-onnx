#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace YirangOnnx
{
	// Plain, protobuf-free tensor contract shared between the CLI (which reads/writes
	// ONNX TensorProto .pb files) and the inference engine (which links onnxruntime).
	// data_ holds raw little-endian element bytes; elem_type_ is an ONNX
	// TensorProto.DataType value.
	struct Tensor
	{
		std::string name_;
		int32_t elem_type_ = 0;
		std::vector<int64_t> shape_;
		std::vector<uint8_t> data_;
	};
}
