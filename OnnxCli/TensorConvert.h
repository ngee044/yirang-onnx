#pragma once

#include "Tensor.h"

#include "onnx.pb.h"

#include <optional>
#include <string>
#include <tuple>

namespace YirangOnnx
{
	auto tensor_from_proto(const onnx::TensorProto& proto) -> std::tuple<std::optional<Tensor>, std::optional<std::string>>;
	auto proto_from_tensor(const Tensor& tensor) -> onnx::TensorProto;
} // namespace YirangOnnx
