#include "InferenceEngine.h"
#include "Tensor.h"

#include "File.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
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

	auto write_dynamic_model(const std::filesystem::path& path, bool unknown_rank) -> void
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		model.add_opset_import()->set_version(18);
		auto* graph = model.mutable_graph();
		graph->set_name("dynamic_outputs");
		auto* input = graph->add_input();
		input->set_name("X");
		input->mutable_type()->mutable_tensor_type()->set_elem_type(1);
		if (!unknown_rank)
		{
			input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);
			auto* nonzero = graph->add_node();
			nonzero->set_op_type("NonZero");
			nonzero->add_input("X");
			nonzero->add_output("indices");
			auto* output = graph->add_output();
			output->set_name("indices");
			auto* type = output->mutable_type()->mutable_tensor_type();
			type->set_elem_type(7);
			type->mutable_shape()->add_dim()->set_dim_value(1);
			type->mutable_shape()->add_dim()->set_dim_param("count");
		}
		auto* identity = graph->add_node();
		identity->set_op_type("Identity");
		identity->add_input("X");
		identity->add_output("echo");
		auto* output = graph->add_output();
		output->set_name("echo");
		output->mutable_type()->mutable_tensor_type()->set_elem_type(1);
		if (!unknown_rank)
		{
			output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);
		}

		std::string serialized;
		ASSERT_TRUE(model.SerializeToString(&serialized));
		File file;
		ASSERT_TRUE(file.open(path.string(), std::ios::out | std::ios::binary | std::ios::trunc).has_value());
		ASSERT_TRUE(file.write_bytes(std::vector<uint8_t>(serialized.begin(), serialized.end())).has_value());
	}

	auto expect_same_outputs(const std::vector<Tensor>& actual, const std::vector<Tensor>& expected) -> void
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (size_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_EQ(actual[i].name_, expected[i].name_);
			EXPECT_EQ(actual[i].elem_type_, expected[i].elem_type_);
			EXPECT_EQ(actual[i].shape_, expected[i].shape_);
			EXPECT_EQ(actual[i].data_, expected[i].data_);
		}
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

TEST(InferenceEngineTest, RejectsInputDataSizeMismatch)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_engine_mismatch.onnx";
	write_sample_model(path);

	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()).has_value());

	Tensor bad;
	bad.name_ = "X";
	bad.elem_type_ = 1;
	bad.shape_ = { 1, 2 };
	bad.data_.resize(sizeof(float)); // one float, but shape wants two

	auto [outputs, error] = engine.run({ bad });
	EXPECT_FALSE(outputs.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("does not match shape"), std::string::npos);

	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedRunsMatchOrdinaryRunsAndObserveInputUpdates)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_reuse.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	SessionTuning tuning;
	tuning.intra_op_threads_ = 1;
	tuning.inter_op_threads_ = 1;
	ASSERT_TRUE(engine.load(path.string(), tuning));
	std::vector<Tensor> inputs{ make_input() };
	ASSERT_TRUE(engine.prepare(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	std::vector<Tensor> first;
	for (int i = 1; i <= 3; ++i)
	{
		const float values[2] = { static_cast<float>(i), 1.0f };
		std::memcpy(inputs[0].data_.data(), values, sizeof(values));
		ASSERT_TRUE(engine.run_prepared(inputs));
		auto [actual, error] = engine.collect_outputs();
		ASSERT_TRUE(actual.has_value()) << error.value_or("");
		auto [expected, run_error] = engine.run(inputs);
		ASSERT_TRUE(expected.has_value()) << run_error.value_or("");
		expect_same_outputs(actual.value(), expected.value());
		if (i == 1)
		{
			first = std::move(actual.value());
		}
	}
	float first_values[2];
	ASSERT_EQ(first.size(), 1u);
	ASSERT_EQ(first[0].data_.size(), sizeof(first_values));
	std::memcpy(first_values, first[0].data_.data(), sizeof(first_values));
	EXPECT_FLOAT_EQ(first_values[0], 4.0f);
	EXPECT_FLOAT_EQ(first_values[1], 6.0f);
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedRunRejectsChangedBindingsUntilPreparedAgain)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_bindings.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	ASSERT_TRUE(engine.prepare(inputs));
	ASSERT_TRUE(engine.run_prepared(inputs));
	inputs[0].shape_ = { 2, 1 };
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	inputs[0].shape_ = { 1, 2 };
	inputs[0].name_ = "renamed";
	EXPECT_FALSE(engine.run_prepared(inputs));
	inputs[0].name_ = "X";
	inputs[0].elem_type_ = 6;
	EXPECT_FALSE(engine.run_prepared(inputs));
	inputs[0].elem_type_ = 1;
	inputs[0].data_.resize(4);
	EXPECT_FALSE(engine.run_prepared(inputs));
	inputs[0] = make_input();
	ASSERT_TRUE(engine.prepare(inputs));
	auto replacement = inputs;
	EXPECT_FALSE(engine.run_prepared(replacement));
	EXPECT_FALSE(engine.run_prepared({}));
	ASSERT_TRUE(engine.prepare(replacement));
	ASSERT_TRUE(engine.run_prepared(replacement));
	EXPECT_TRUE(std::get<0>(engine.collect_outputs()).has_value());
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedDynamicOutputsChangeWithInputData)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_dynamic.onnx";
	write_dynamic_model(path, false);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	inputs[0].shape_ = { 4 };
	inputs[0].data_.resize(4 * sizeof(float));
	ASSERT_TRUE(engine.prepare(inputs));
	for (int count : { 1, 3, 0, 2 })
	{
		float values[4] = {};
		for (int i = 0; i < count; ++i)
		{
			values[i] = 1.0f;
		}
		std::memcpy(inputs[0].data_.data(), values, sizeof(values));
		auto executed = engine.run_prepared(inputs);
		ASSERT_TRUE(executed) << executed.error();
		auto [actual, error] = engine.collect_outputs();
		ASSERT_TRUE(actual.has_value()) << error.value_or("");
		auto [expected, run_error] = engine.run(inputs);
		ASSERT_TRUE(expected.has_value()) << run_error.value_or("");
		expect_same_outputs(actual.value(), expected.value());
		ASSERT_EQ(actual->size(), 2u);
		EXPECT_EQ(actual->front().shape_, (std::vector<int64_t>{ 1, count }));
	}
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedUnknownRankHandlesRebindingAndScalars)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_unknown_rank.onnx";
	write_dynamic_model(path, true);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	for (const auto& shape : std::vector<std::vector<int64_t>>{ { 1, 2 }, { 2 }, {}, { 0 } })
	{
		inputs[0].shape_ = shape;
		inputs[0].data_.resize(shape.empty() ? sizeof(float) : (shape.front() == 0 ? 0 : 2 * sizeof(float)));
		ASSERT_TRUE(engine.prepare(inputs));
		auto executed = engine.run_prepared(inputs);
		ASSERT_TRUE(executed) << executed.error();
		auto [actual, error] = engine.collect_outputs();
		ASSERT_TRUE(actual.has_value()) << error.value_or("");
		ASSERT_EQ(actual->size(), 1u);
		EXPECT_EQ(actual->front().shape_, shape);
		EXPECT_EQ(actual->front().data_, inputs[0].data_);
	}
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedStateIsClearedByPreparationFailureAndReload)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_lifecycle.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	std::vector<Tensor> inputs{ make_input() };
	EXPECT_FALSE(engine.prepare(inputs));
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	ASSERT_TRUE(engine.load(path.string()));
	ASSERT_TRUE(engine.prepare(inputs));
	ASSERT_TRUE(engine.run_prepared(inputs));
	auto bad = inputs;
	bad[0].data_.clear();
	EXPECT_FALSE(engine.prepare(bad));
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	ASSERT_TRUE(engine.prepare(inputs));
	ASSERT_TRUE(engine.run_prepared(inputs));
	ASSERT_TRUE(engine.load(path.string()));
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	ASSERT_TRUE(engine.prepare(inputs));
	EXPECT_FALSE(engine.load("/no/such/prepared-model.onnx"));
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedRunFailureDoesNotExposeOldOutputs)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_failure.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	ASSERT_TRUE(engine.prepare(inputs));
	ASSERT_TRUE(engine.run_prepared(inputs));
	inputs[0].shape_ = { 2, 1 };
	ASSERT_TRUE(engine.prepare(inputs));
	EXPECT_FALSE(engine.run_prepared(inputs));
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	inputs[0].shape_ = { 1, 2 };
	ASSERT_TRUE(engine.prepare(inputs));
	ASSERT_TRUE(engine.run_prepared(inputs));
	EXPECT_TRUE(std::get<0>(engine.collect_outputs()).has_value());
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparedStateMovesAndStaysIndependentBetweenEngines)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_move.onnx";
	write_sample_model(path);
	InferenceEngine first;
	InferenceEngine second;
	ASSERT_TRUE(first.load(path.string()));
	ASSERT_TRUE(second.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	auto other = inputs;
	const float values[2] = { 2.0f, 3.0f };
	std::memcpy(other[0].data_.data(), values, sizeof(values));
	ASSERT_TRUE(first.prepare(inputs));
	ASSERT_TRUE(second.prepare(other));
	InferenceEngine moved(std::move(first));
	EXPECT_FALSE(first.run_prepared(inputs));
	ASSERT_TRUE(moved.run_prepared(inputs));
	ASSERT_TRUE(second.run_prepared(other));
	auto [actual, error] = moved.collect_outputs();
	auto [expected, run_error] = moved.run(inputs);
	ASSERT_TRUE(actual.has_value()) << error.value_or("");
	ASSERT_TRUE(expected.has_value()) << run_error.value_or("");
	expect_same_outputs(actual.value(), expected.value());
	std::tie(actual, error) = second.collect_outputs();
	std::tie(expected, run_error) = second.run(other);
	ASSERT_TRUE(actual.has_value()) << error.value_or("");
	ASSERT_TRUE(expected.has_value()) << run_error.value_or("");
	expect_same_outputs(actual.value(), expected.value());
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, RejectsOverflowingTensorBytesInBothExecutionPaths)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_overflow.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	std::vector<Tensor> inputs{ make_input() };
	inputs[0].shape_ = { std::numeric_limits<int64_t>::max(), 2 };
	const auto prepared = engine.prepare(inputs);
	ASSERT_FALSE(prepared);
	EXPECT_NE(prepared.error().find("overflows"), std::string::npos);
	auto [outputs, error] = engine.run(inputs);
	EXPECT_FALSE(outputs.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("overflows"), std::string::npos);
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparationRejectsDuplicateInputBindings)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_duplicate.onnx";
	write_sample_model(path);
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	const auto prepared = engine.prepare({ make_input(), make_input() });
	ASSERT_FALSE(prepared);
	EXPECT_NE(prepared.error().find("duplicate input name"), std::string::npos);
	std::error_code ec;
	std::filesystem::remove(path, ec);
}

TEST(InferenceEngineTest, PreparationRejectsNonTensorOutputs)
{
	const auto path = std::filesystem::temp_directory_path() / "yirang_prepared_sequence.onnx";
	onnx::ModelProto model;
	model.set_ir_version(9);
	model.add_opset_import()->set_version(18);
	auto* graph = model.mutable_graph();
	graph->set_name("sequence_output");
	auto* input = graph->add_input();
	input->set_name("X");
	input->mutable_type()->mutable_tensor_type()->set_elem_type(1);
	auto* sequence = graph->add_node();
	sequence->set_op_type("SequenceConstruct");
	sequence->add_input("X");
	sequence->add_output("sequence");
	auto* output = graph->add_output();
	output->set_name("sequence");
	output->mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(1);
	std::string serialized;
	ASSERT_TRUE(model.SerializeToString(&serialized));
	{
		File file;
		ASSERT_TRUE(file.open(path.string(), std::ios::out | std::ios::binary | std::ios::trunc).has_value());
		ASSERT_TRUE(file.write_bytes(std::vector<uint8_t>(serialized.begin(), serialized.end())).has_value());
	}
	InferenceEngine engine;
	ASSERT_TRUE(engine.load(path.string()));
	const auto prepared = engine.prepare({ make_input() });
	ASSERT_FALSE(prepared);
	EXPECT_NE(prepared.error().find("only tensor outputs"), std::string::npos);
	EXPECT_FALSE(std::get<0>(engine.collect_outputs()).has_value());
	std::error_code ec;
	std::filesystem::remove(path, ec);
}
