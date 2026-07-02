#include "OnnxModel.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace YirangOnnx;

namespace
{
	auto build_sample_model(void) -> std::vector<uint8_t>
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		model.set_producer_name("yirang-test");
		model.set_producer_version("0.1.0");

		auto* opset = model.add_opset_import();
		opset->set_domain("");
		opset->set_version(18);

		auto* graph = model.mutable_graph();
		graph->set_name("test_graph");

		auto* input = graph->add_input();
		input->set_name("X");
		auto* input_tensor = input->mutable_type()->mutable_tensor_type();
		input_tensor->set_elem_type(1); // FLOAT
		auto* input_shape = input_tensor->mutable_shape();
		input_shape->add_dim()->set_dim_value(1);
		input_shape->add_dim()->set_dim_param("N");

		auto* weight = graph->add_initializer();
		weight->set_name("W");
		weight->set_data_type(1); // FLOAT
		weight->add_dims(2);
		weight->add_dims(2);
		weight->add_float_data(1.0f);
		weight->add_float_data(2.0f);
		weight->add_float_data(3.0f);
		weight->add_float_data(4.0f);

		auto* matmul = graph->add_node();
		matmul->set_op_type("MatMul");
		matmul->set_name("matmul0");
		matmul->add_input("X");
		matmul->add_input("W");
		matmul->add_output("Y");

		auto* relu = graph->add_node();
		relu->set_op_type("Relu");
		relu->add_input("Y");
		relu->add_output("Z");

		auto* output = graph->add_output();
		output->set_name("Z");
		output->mutable_type()->mutable_tensor_type()->set_elem_type(1);

		std::string serialized;
		model.SerializeToString(&serialized);
		return std::vector<uint8_t>(serialized.begin(), serialized.end());
	}
}

TEST(OnnxModelTest, ParsesMetadata)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value()) << error.value_or("");

	const auto meta = model->metadata();
	EXPECT_EQ(meta.ir_version_, 9);
	EXPECT_EQ(meta.producer_name_, "yirang-test");
	EXPECT_EQ(meta.graph_name_, "test_graph");
	ASSERT_EQ(meta.opset_imports_.size(), 1u);
	EXPECT_EQ(meta.opset_imports_[0].version_, 18);
}

TEST(OnnxModelTest, ExtractsNodesAndOperators)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value());

	const auto nodes = model->nodes();
	ASSERT_EQ(nodes.size(), 2u);
	EXPECT_EQ(nodes[0].op_type_, "MatMul");
	EXPECT_EQ(nodes[0].inputs_.size(), 2u);
	EXPECT_EQ(nodes[1].op_type_, "Relu");

	const auto histogram = model->operator_histogram();
	ASSERT_EQ(histogram.size(), 2u);
	EXPECT_EQ(histogram[0].second, 1u);
}

TEST(OnnxModelTest, ExtractsInitializersAndIo)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value());

	const auto initializers = model->initializers();
	ASSERT_EQ(initializers.size(), 1u);
	EXPECT_EQ(initializers[0].name_, "W");
	EXPECT_EQ(initializers[0].data_type_, "FLOAT");
	ASSERT_EQ(initializers[0].dims_.size(), 2u);
	EXPECT_EQ(initializers[0].byte_size_, 4u * sizeof(float));
	EXPECT_EQ(initializers[0].source_, TensorDataSource::Inline);

	EXPECT_EQ(model->inputs().size(), 1u);
	EXPECT_EQ(model->outputs().size(), 1u);
	EXPECT_EQ(model->inputs()[0].data_type_, "FLOAT");
	ASSERT_EQ(model->inputs()[0].shape_.size(), 2u);
	EXPECT_EQ(model->inputs()[0].shape_[1], "N");
}

TEST(OnnxModelTest, RendersJsonAndDot)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value());

	const auto json = model->to_json();
	EXPECT_NE(json.find("MatMul"), std::string::npos);
	EXPECT_NE(json.find("test_graph"), std::string::npos);

	const auto dot = model->to_dot();
	EXPECT_NE(dot.find("digraph"), std::string::npos);
	EXPECT_NE(dot.find("->"), std::string::npos);
}

TEST(OnnxModelTest, RejectsEmptyBuffer)
{
	auto [model, error] = OnnxModel::parse(std::vector<uint8_t>{});
	EXPECT_FALSE(model.has_value());
	EXPECT_TRUE(error.has_value());
}

TEST(OnnxModelTest, LoadsFromFileRoundTrip)
{
	// YIRANG_SAMPLE_OUT: write the sample .onnx there for CLI smoke-testing.
	const auto bytes = build_sample_model();

	std::filesystem::path path;
	if (const char* out = std::getenv("YIRANG_SAMPLE_OUT"); out != nullptr)
	{
		path = out;
	}
	else
	{
		path = std::filesystem::temp_directory_path() / "yirang_sample_roundtrip.onnx";
	}

	{
		std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
		ASSERT_TRUE(file.is_open());
		file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}

	auto [model, error] = OnnxModel::load(path.string());
	ASSERT_TRUE(model.has_value()) << error.value_or("");
	EXPECT_EQ(model->nodes().size(), 2u);
	EXPECT_EQ(model->metadata().graph_name_, "test_graph");

	auto [missing, missing_error] = OnnxModel::load("/no/such/path/does-not-exist.onnx");
	EXPECT_FALSE(missing.has_value());
	EXPECT_TRUE(missing_error.has_value());
}
