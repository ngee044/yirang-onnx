#include "TensorConvert.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace YirangOnnx;

namespace
{
	auto make_proto(const std::string& name, int32_t data_type, const std::vector<int64_t>& dims) -> onnx::TensorProto
	{
		onnx::TensorProto proto;
		proto.set_name(name);
		proto.set_data_type(data_type);
		for (int64_t dim : dims)
		{
			proto.add_dims(dim);
		}
		return proto;
	}
} // namespace

TEST(TensorConvertTest, RoundTripsRawData)
{
	auto proto = make_proto("X", onnx::TensorProto::FLOAT, { 1, 2 });
	const std::vector<float> values{ 1.5f, -2.0f };
	proto.set_raw_data(values.data(), values.size() * sizeof(float));

	auto [tensor, error] = tensor_from_proto(proto);
	ASSERT_TRUE(tensor.has_value()) << error.value_or("");
	EXPECT_EQ(tensor->name_, "X");
	EXPECT_EQ(tensor->elem_type_, onnx::TensorProto::FLOAT);
	ASSERT_EQ(tensor->shape_.size(), 2u);
	EXPECT_EQ(tensor->shape_[1], 2);
	ASSERT_EQ(tensor->data_.size(), values.size() * sizeof(float));
	EXPECT_EQ(std::memcmp(tensor->data_.data(), values.data(), tensor->data_.size()), 0);

	const auto round = proto_from_tensor(tensor.value());
	EXPECT_EQ(round.name(), proto.name());
	EXPECT_EQ(round.data_type(), proto.data_type());
	ASSERT_EQ(round.dims_size(), proto.dims_size());
	EXPECT_EQ(round.raw_data(), proto.raw_data());
}

TEST(TensorConvertTest, DecodesTypedFloatData)
{
	auto proto = make_proto("X", onnx::TensorProto::FLOAT, { 3 });
	proto.add_float_data(1.0f);
	proto.add_float_data(2.0f);
	proto.add_float_data(3.0f);

	auto [tensor, error] = tensor_from_proto(proto);
	ASSERT_TRUE(tensor.has_value()) << error.value_or("");
	ASSERT_EQ(tensor->data_.size(), 3u * sizeof(float));

	float decoded[3];
	std::memcpy(decoded, tensor->data_.data(), sizeof(decoded));
	EXPECT_FLOAT_EQ(decoded[0], 1.0f);
	EXPECT_FLOAT_EQ(decoded[2], 3.0f);
}

TEST(TensorConvertTest, DecodesTypedInt64Data)
{
	auto proto = make_proto("I", onnx::TensorProto::INT64, { 2 });
	proto.add_int64_data(-7);
	proto.add_int64_data(42);

	auto [tensor, error] = tensor_from_proto(proto);
	ASSERT_TRUE(tensor.has_value()) << error.value_or("");
	ASSERT_EQ(tensor->data_.size(), 2u * sizeof(int64_t));

	int64_t decoded[2];
	std::memcpy(decoded, tensor->data_.data(), sizeof(decoded));
	EXPECT_EQ(decoded[0], -7);
	EXPECT_EQ(decoded[1], 42);
}

TEST(TensorConvertTest, RejectsUnsupportedTypedData)
{
	auto proto = make_proto("U", onnx::TensorProto::UINT8, { 2 });
	proto.add_int32_data(1);
	proto.add_int32_data(2);

	auto [tensor, error] = tensor_from_proto(proto);
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("UINT8"), std::string::npos);
}

TEST(TensorConvertTest, SerializesTensorToProto)
{
	Tensor tensor;
	tensor.name_ = "Z";
	tensor.elem_type_ = onnx::TensorProto::INT32;
	tensor.shape_ = { 2, 1 };
	const std::vector<int32_t> values{ 5, -9 };
	tensor.data_.resize(values.size() * sizeof(int32_t));
	std::memcpy(tensor.data_.data(), values.data(), tensor.data_.size());

	const auto proto = proto_from_tensor(tensor);
	EXPECT_EQ(proto.name(), "Z");
	EXPECT_EQ(proto.data_type(), onnx::TensorProto::INT32);
	ASSERT_EQ(proto.dims_size(), 2);
	EXPECT_EQ(proto.dims(0), 2);
	ASSERT_EQ(proto.raw_data().size(), tensor.data_.size());
	EXPECT_EQ(std::memcmp(proto.raw_data().data(), values.data(), proto.raw_data().size()), 0);
}

TEST(TensorConvertTest, RejectsExternalDataTensor)
{
	auto proto = make_proto("E", onnx::TensorProto::FLOAT, { 2 });
	proto.set_data_location(onnx::TensorProto::EXTERNAL);

	auto [tensor, error] = tensor_from_proto(proto);
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("external data"), std::string::npos);
}

TEST(TensorConvertTest, RejectsRawDataSizeMismatch)
{
	auto proto = make_proto("M", onnx::TensorProto::FLOAT, { 2 });
	const float value = 1.0f;
	proto.set_raw_data(&value, sizeof(value));

	auto [tensor, error] = tensor_from_proto(proto);
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("does not match shape"), std::string::npos);
}
