#include "File.h"
#include "OnnxModel.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace Utilities;

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

	auto build_attribute_model(void) -> std::vector<uint8_t>
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* graph = model.mutable_graph();
		graph->set_name("attribute_graph");

		auto* conv = graph->add_node();
		conv->set_op_type("Conv");
		conv->set_name("conv0");
		conv->add_input("X");
		conv->add_output("Y");

		auto* kernel = conv->add_attribute();
		kernel->set_name("kernel_shape");
		kernel->set_type(onnx::AttributeProto::INTS);
		kernel->add_ints(3);
		kernel->add_ints(3);

		auto* alpha = conv->add_attribute();
		alpha->set_name("alpha");
		alpha->set_type(onnx::AttributeProto::FLOAT);
		alpha->set_f(0.5f);

		auto* mode = conv->add_attribute();
		mode->set_name("mode");
		mode->set_type(onnx::AttributeProto::STRING);
		mode->set_s("constant");

		std::string serialized;
		model.SerializeToString(&serialized);
		return std::vector<uint8_t>(serialized.begin(), serialized.end());
	}

	auto build_subgraph_model(void) -> std::vector<uint8_t>
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* graph = model.mutable_graph();
		graph->set_name("subgraph_graph");

		auto* if_node = graph->add_node();
		if_node->set_op_type("If");
		if_node->set_name("if0");
		if_node->add_input("cond");
		if_node->add_output("out");

		auto* then_branch = if_node->add_attribute();
		then_branch->set_name("then_branch");
		then_branch->set_type(onnx::AttributeProto::GRAPH);
		auto* then_graph = then_branch->mutable_g();
		then_graph->set_name("then_graph");
		auto* then_relu = then_graph->add_node();
		then_relu->set_op_type("Relu");
		then_relu->add_input("A");
		then_relu->add_output("B");

		auto* else_branch = if_node->add_attribute();
		else_branch->set_name("else_branch");
		else_branch->set_type(onnx::AttributeProto::GRAPH);
		auto* else_graph = else_branch->mutable_g();
		else_graph->set_name("else_graph");
		auto* else_sigmoid = else_graph->add_node();
		else_sigmoid->set_op_type("Sigmoid");
		else_sigmoid->add_input("A");
		else_sigmoid->add_output("B");

		std::string serialized;
		model.SerializeToString(&serialized);
		return std::vector<uint8_t>(serialized.begin(), serialized.end());
	}

	auto build_external_model(void) -> std::vector<uint8_t>
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* graph = model.mutable_graph();
		graph->set_name("external_graph");

		auto* weight = graph->add_initializer();
		weight->set_name("W_ext");
		weight->set_data_type(1); // FLOAT
		weight->add_dims(4);
		weight->set_data_location(onnx::TensorProto::EXTERNAL);
		auto* location = weight->add_external_data();
		location->set_key("location");
		location->set_value("weights.bin");
		auto* offset = weight->add_external_data();
		offset->set_key("offset");
		offset->set_value("0");
		auto* length = weight->add_external_data();
		length->set_key("length");
		length->set_value("16");

		std::string serialized;
		model.SerializeToString(&serialized);
		return std::vector<uint8_t>(serialized.begin(), serialized.end());
	}
} // namespace

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
	EXPECT_EQ(initializers[0].element_count(), 4u);

	EXPECT_EQ(model->inputs().size(), 1u);
	EXPECT_EQ(model->outputs().size(), 1u);
	EXPECT_EQ(model->inputs()[0].data_type_, "FLOAT");
	EXPECT_EQ(model->inputs()[0].elem_type_, 1);
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
	EXPECT_NE(json.find("total_parameters"), std::string::npos);

	const auto dot = model->to_dot();
	EXPECT_NE(dot.find("digraph"), std::string::npos);
	EXPECT_NE(dot.find("->"), std::string::npos);
}

TEST(OnnxModelTest, DotIncludesGraphIo)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value());

	const auto dot = model->to_dot();
	EXPECT_NE(dot.find("gi0"), std::string::npos);
	EXPECT_NE(dot.find("go0"), std::string::npos);
	EXPECT_NE(dot.find("gi0 -> n0"), std::string::npos);
	EXPECT_NE(dot.find("n1 -> go0"), std::string::npos);
	EXPECT_NE(dot.find("FLOAT [1, N]"), std::string::npos);
}

TEST(OnnxModelTest, SummaryHasParameterTotals)
{
	auto [model, error] = OnnxModel::parse(build_sample_model());
	ASSERT_TRUE(model.has_value());

	const auto summary = model->to_summary();
	EXPECT_NE(summary.find("parameters    : 4 (16 B)"), std::string::npos);
}

TEST(OnnxModelTest, ExtractsAttributeValues)
{
	auto [model, error] = OnnxModel::parse(build_attribute_model());
	ASSERT_TRUE(model.has_value()) << error.value_or("");

	const auto nodes = model->nodes();
	ASSERT_EQ(nodes.size(), 1u);
	ASSERT_EQ(nodes[0].attributes_.size(), 3u);

	const auto& kernel = nodes[0].attributes_[0];
	EXPECT_EQ(kernel.name_, "kernel_shape");
	EXPECT_EQ(kernel.type_, "INTS");
	ASSERT_EQ(kernel.ints_.size(), 2u);
	EXPECT_EQ(kernel.ints_[0], 3);
	EXPECT_EQ(kernel.value_, "[3, 3]");

	const auto& alpha = nodes[0].attributes_[1];
	EXPECT_EQ(alpha.type_, "FLOAT");
	ASSERT_EQ(alpha.floats_.size(), 1u);
	EXPECT_DOUBLE_EQ(alpha.floats_[0], 0.5);

	const auto& mode = nodes[0].attributes_[2];
	EXPECT_EQ(mode.type_, "STRING");
	EXPECT_EQ(mode.value_, "constant");

	const auto json = model->to_json();
	EXPECT_NE(json.find("kernel_shape"), std::string::npos);
	EXPECT_NE(json.find("constant"), std::string::npos);
}

TEST(OnnxModelTest, AggregatesSubgraphNodes)
{
	auto [model, error] = OnnxModel::parse(build_subgraph_model());
	ASSERT_TRUE(model.has_value()) << error.value_or("");

	const auto nodes = model->nodes();
	ASSERT_EQ(nodes.size(), 1u);
	EXPECT_EQ(nodes[0].subgraph_node_count_, 2u);
	ASSERT_EQ(nodes[0].attributes_.size(), 2u);
	EXPECT_EQ(nodes[0].attributes_[0].type_, "GRAPH");
	EXPECT_NE(nodes[0].attributes_[0].value_.find("then_graph"), std::string::npos);

	const auto histogram = model->operator_histogram();
	ASSERT_EQ(histogram.size(), 3u);

	const auto summary = model->to_summary();
	EXPECT_NE(summary.find("(+2 in subgraphs)"), std::string::npos);
}

TEST(OnnxModelTest, ReportsExternalInitializerMetadata)
{
	auto [model, error] = OnnxModel::parse(build_external_model());
	ASSERT_TRUE(model.has_value()) << error.value_or("");

	const auto initializers = model->initializers();
	ASSERT_EQ(initializers.size(), 1u);
	EXPECT_EQ(initializers[0].source_, TensorDataSource::External);
	EXPECT_EQ(initializers[0].byte_size_, 16u);
	ASSERT_EQ(initializers[0].external_data_.size(), 3u);
	EXPECT_EQ(initializers[0].external_data_[0].key_, "location");
	EXPECT_EQ(initializers[0].external_data_[0].value_, "weights.bin");

	const auto json = model->to_json();
	EXPECT_NE(json.find("weights.bin"), std::string::npos);
}

TEST(OnnxModelTest, MapsDataTypeNamesToIds)
{
	EXPECT_EQ(data_type_id("FLOAT"), std::optional<int32_t>(1));
	EXPECT_EQ(data_type_id("int64"), std::optional<int32_t>(7));
	EXPECT_EQ(data_type_id("BFLOAT16"), std::optional<int32_t>(16));
	EXPECT_FALSE(data_type_id("NOPE").has_value());
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
		File file;
		ASSERT_TRUE(file.open(path.string(), std::ios::out | std::ios::binary | std::ios::trunc).has_value());
		ASSERT_TRUE(file.write_bytes(bytes).has_value());
	}

	auto [model, error] = OnnxModel::load(path.string());
	ASSERT_TRUE(model.has_value()) << error.value_or("");
	EXPECT_EQ(model->nodes().size(), 2u);
	EXPECT_EQ(model->metadata().graph_name_, "test_graph");

	auto [missing, missing_error] = OnnxModel::load("/no/such/path/does-not-exist.onnx");
	EXPECT_FALSE(missing.has_value());
	EXPECT_TRUE(missing_error.has_value());
}
