#include "InferenceEngine.h"
#include "Tensor.h"

#include "File.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace Utilities;

using namespace YirangOnnx;

namespace
{
	auto write_sample_model(const std::filesystem::path& path) -> void
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* opset = model.add_opset_import();
		opset->set_domain("");
		opset->set_version(18);

		auto* graph = model.mutable_graph();
		graph->set_name("engine_graph");

		auto* input = graph->add_input();
		input->set_name("X");
		auto* input_tensor = input->mutable_type()->mutable_tensor_type();
		input_tensor->set_elem_type(1);
		auto* input_shape = input_tensor->mutable_shape();
		input_shape->add_dim()->set_dim_value(1);
		input_shape->add_dim()->set_dim_value(2);

		auto* weight = graph->add_initializer();
		weight->set_name("W");
		weight->set_data_type(1);
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

		File file;
		ASSERT_TRUE(file.open(path.string(), std::ios::out | std::ios::binary | std::ios::trunc).has_value());
		ASSERT_TRUE(file.write_bytes(std::vector<uint8_t>(serialized.begin(), serialized.end())).has_value());
	}

	auto make_input(void) -> Tensor
	{
		Tensor tensor;
		tensor.name_ = "X";
		tensor.elem_type_ = 1;
		tensor.shape_ = { 1, 2 };
		const float values[2] = { 1.0f, 1.0f };
		tensor.data_.resize(sizeof(values));
		std::memcpy(tensor.data_.data(), values, sizeof(values));
		return tensor;
	}
} // namespace

TEST(InferenceEngineTest, RunsModelEndToEndAndReusesSession)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_engine_e2e.onnx";
	write_sample_model(path);

	InferenceEngine engine;
	EXPECT_FALSE(engine.loaded());

	auto loaded = engine.load(path.string());
	ASSERT_TRUE(loaded.has_value()) << (loaded.has_value() ? "" : loaded.error());
	EXPECT_TRUE(engine.loaded());

	for (int round = 0; round < 2; ++round)
	{
		auto [outputs, error] = engine.run({ make_input() });
		ASSERT_TRUE(outputs.has_value()) << error.value_or("");
		ASSERT_EQ(outputs->size(), 1u);

		const auto& z = outputs->front();
		EXPECT_EQ(z.name_, "Z");
		EXPECT_EQ(z.elem_type_, 1);
		ASSERT_EQ(z.shape_.size(), 2u);
		EXPECT_EQ(z.shape_[1], 2);
		ASSERT_EQ(z.data_.size(), 2u * sizeof(float));

		float values[2];
		std::memcpy(values, z.data_.data(), sizeof(values));
		EXPECT_FLOAT_EQ(values[0], 4.0f);
		EXPECT_FLOAT_EQ(values[1], 6.0f);
	}

	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, RunsWithSessionTuning)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_engine_tuned.onnx";
	write_sample_model(path);

	SessionTuning tuning;
	tuning.intra_op_threads_ = 1;
	tuning.inter_op_threads_ = 1;
	tuning.enable_mem_pattern_ = false;
	tuning.enable_cpu_mem_arena_ = false;
	tuning.execution_mode_ = ExecutionMode::parallel;
	tuning.graph_optimization_ = GraphOptimization::disabled;

	InferenceEngine engine;
	auto loaded = engine.load(path.string(), tuning);
	ASSERT_TRUE(loaded.has_value()) << (loaded.has_value() ? "" : loaded.error());

	auto [outputs, error] = engine.run({ make_input() });
	ASSERT_TRUE(outputs.has_value()) << error.value_or("");
	ASSERT_EQ(outputs->size(), 1u);

	const auto& z = outputs->front();
	ASSERT_EQ(z.data_.size(), 2u * sizeof(float));
	float values[2];
	std::memcpy(values, z.data_.data(), sizeof(values));
	EXPECT_FLOAT_EQ(values[0], 4.0f);
	EXPECT_FLOAT_EQ(values[1], 6.0f);

	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, RunWithoutLoadFails)
{
	const InferenceEngine engine;
	auto [outputs, error] = engine.run({});
	EXPECT_FALSE(outputs.has_value());
	EXPECT_TRUE(error.has_value());
}

TEST(InferenceEngineTest, LoadFailsForMissingModel)
{
	InferenceEngine engine;
	auto loaded = engine.load("/no/such/model.onnx");
	EXPECT_FALSE(loaded.has_value());
	EXPECT_FALSE(engine.loaded());
}
