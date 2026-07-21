#include "OnnxModel.h"
#include "RunSupport.h"
#include "TensorConvert.h"

#include "onnx.pb.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace YirangOnnx;

namespace
{
	auto make_float_tensor(const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& values) -> Tensor
	{
		Tensor tensor;
		tensor.name_ = name;
		tensor.elem_type_ = onnx::TensorProto::FLOAT;
		tensor.shape_ = shape;
		tensor.data_.resize(values.size() * sizeof(float));
		std::memcpy(tensor.data_.data(), values.data(), tensor.data_.size());
		return tensor;
	}

	auto build_add_model(void) -> OnnxModel
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* opset = model.add_opset_import();
		opset->set_domain("");
		opset->set_version(18);

		auto* graph = model.mutable_graph();
		graph->set_name("run_support_graph");

		auto* x = graph->add_input();
		x->set_name("X");
		auto* x_type = x->mutable_type()->mutable_tensor_type();
		x_type->set_elem_type(onnx::TensorProto::FLOAT);
		x_type->mutable_shape()->add_dim()->set_dim_value(1);
		x_type->mutable_shape()->add_dim()->set_dim_value(2);

		auto* w_input = graph->add_input();
		w_input->set_name("W");
		auto* w_type = w_input->mutable_type()->mutable_tensor_type();
		w_type->set_elem_type(onnx::TensorProto::FLOAT);
		w_type->mutable_shape()->add_dim()->set_dim_value(2);

		auto* w = graph->add_initializer();
		w->set_name("W");
		w->set_data_type(onnx::TensorProto::FLOAT);
		w->add_dims(2);
		w->add_float_data(1.0F);
		w->add_float_data(2.0F);

		auto* node = graph->add_node();
		node->set_op_type("Add");
		node->add_input("X");
		node->add_input("W");
		node->add_output("Z");

		auto* z = graph->add_output();
		z->set_name("Z");
		z->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::FLOAT);

		std::string bytes;
		model.SerializeToString(&bytes);
		auto [parsed, error] = OnnxModel::parse(std::vector<uint8_t>(bytes.begin(), bytes.end()));
		EXPECT_TRUE(parsed.has_value()) << error.value_or("");
		return std::move(parsed.value());
	}

	auto build_two_data_input_model(void) -> OnnxModel
	{
		onnx::ModelProto model;
		model.set_ir_version(9);
		auto* opset = model.add_opset_import();
		opset->set_domain("");
		opset->set_version(18);

		auto* graph = model.mutable_graph();
		graph->set_name("two_input_graph");

		for (const auto* name : { "X", "Y" })
		{
			auto* input = graph->add_input();
			input->set_name(name);
			auto* type = input->mutable_type()->mutable_tensor_type();
			type->set_elem_type(onnx::TensorProto::FLOAT);
			type->mutable_shape()->add_dim()->set_dim_value(1);
		}

		auto* node = graph->add_node();
		node->set_op_type("Add");
		node->add_input("X");
		node->add_input("Y");
		node->add_output("Z");

		auto* z = graph->add_output();
		z->set_name("Z");
		z->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::FLOAT);

		std::string bytes;
		model.SerializeToString(&bytes);
		auto [parsed, error] = OnnxModel::parse(std::vector<uint8_t>(bytes.begin(), bytes.end()));
		EXPECT_TRUE(parsed.has_value()) << error.value_or("");
		return std::move(parsed.value());
	}

	class ScopedTempFile
	{
	public:
		explicit ScopedTempFile(const std::string& name) : path_(std::filesystem::temp_directory_path() / name) {}
		~ScopedTempFile(void)
		{
			std::error_code ec;
			std::filesystem::remove(path_, ec);
		}

		auto path(void) const -> const std::filesystem::path& { return path_; }

	private:
		std::filesystem::path path_;
	};
} // namespace

TEST(RunSupportTest, SafeFileNameReplacesSeparators)
{
	EXPECT_EQ(safe_file_name("a/b\\c"), "a_b_c");
	EXPECT_EQ(safe_file_name("plain"), "plain");
}

TEST(RunSupportTest, WriteFileBytesWritesAndVerifies)
{
	ScopedTempFile file(::testing::UnitTest::GetInstance()->current_test_info()->name() + std::string(".bin"));
	const std::vector<uint8_t> payload = { 1, 2, 3, 4, 5 };

	auto error = write_file_bytes(file.path().string(), payload.data(), payload.size());
	ASSERT_FALSE(error.has_value()) << error.value_or("");

	std::ifstream in(file.path(), std::ios::binary);
	std::vector<uint8_t> read_back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	EXPECT_EQ(read_back, payload);
}

TEST(RunSupportTest, LoadTensorFileFailsForMissingFile)
{
	auto [tensor, error] = load_tensor_file("/nonexistent/tensor.pb");
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("cannot open"), std::string::npos);
}

TEST(RunSupportTest, LoadTensorFileRejectsUnnamedTensor)
{
	onnx::TensorProto proto;
	proto.set_data_type(onnx::TensorProto::FLOAT);
	proto.add_dims(1);
	proto.add_float_data(1.0F);
	std::string serialized;
	ASSERT_TRUE(proto.SerializeToString(&serialized));

	ScopedTempFile file(::testing::UnitTest::GetInstance()->current_test_info()->name() + std::string(".pb"));
	std::ofstream out(file.path(), std::ios::binary);
	out.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
	out.close();

	auto [tensor, error] = load_tensor_file(file.path().string());
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("has no name"), std::string::npos);
}

TEST(RunSupportTest, StatsLineComputesMinMaxMean)
{
	const Tensor tensor = make_float_tensor("Z", { 3 }, { 1.0F, 2.0F, 3.0F });

	auto line = tensor_stats_line(tensor);
	ASSERT_TRUE(line.has_value());
	EXPECT_NE(line->find("min 1"), std::string::npos);
	EXPECT_NE(line->find("max 3"), std::string::npos);
	EXPECT_NE(line->find("mean 2"), std::string::npos);
}

TEST(RunSupportTest, StatsLineUnsupportedDtypeReturnsNullopt)
{
	Tensor tensor;
	tensor.name_ = "bytes";
	tensor.elem_type_ = onnx::TensorProto::UINT8;
	tensor.shape_ = { 2 };
	tensor.data_ = { 1, 0 };

	EXPECT_FALSE(tensor_stats_line(tensor).has_value());
}

TEST(RunSupportTest, StatsAndJsonSupportBool)
{
	Tensor tensor;
	tensor.name_ = "flags";
	tensor.elem_type_ = onnx::TensorProto::BOOL;
	tensor.shape_ = { 3 };
	tensor.data_ = { 1, 0, 1 };

	auto line = tensor_stats_line(tensor);
	ASSERT_TRUE(line.has_value());
	EXPECT_NE(line->find("min 0"), std::string::npos);
	EXPECT_NE(line->find("max 1"), std::string::npos);

	auto dumped = tensor_json_dump(tensor);
	ASSERT_TRUE(dumped.has_value());
	EXPECT_NE(dumped->find("true"), std::string::npos);
	EXPECT_NE(dumped->find("false"), std::string::npos);
}

TEST(RunSupportTest, JsonDumpContainsShapeAndValues)
{
	Tensor tensor;
	tensor.name_ = "counts";
	tensor.elem_type_ = onnx::TensorProto::INT64;
	tensor.shape_ = { 2 };
	const std::vector<int64_t> values = { 7, 9 };
	tensor.data_.resize(values.size() * sizeof(int64_t));
	std::memcpy(tensor.data_.data(), values.data(), tensor.data_.size());

	auto dumped = tensor_json_dump(tensor);
	ASSERT_TRUE(dumped.has_value());
	EXPECT_NE(dumped->find("\"INT64\""), std::string::npos);
	EXPECT_NE(dumped->find('7'), std::string::npos);
	EXPECT_NE(dumped->find('9'), std::string::npos);
}

TEST(RunSupportTest, ResolveAutoRandomSkipsInitializerInputs)
{
	const OnnxModel model = build_add_model();
	InferenceJob job;

	auto [inputs, error] = resolve_job_inputs(model, job);
	ASSERT_TRUE(inputs.has_value()) << error.value_or("");
	ASSERT_EQ(inputs->size(), 1U);
	EXPECT_EQ(inputs->front().name_, "X");
	EXPECT_EQ(inputs->front().data_.size(), 2U * sizeof(float));
}

TEST(RunSupportTest, ResolveRejectsUnknownInputName)
{
	const OnnxModel model = build_add_model();
	InferenceJob job;
	InputSpec spec;
	spec.name_ = "nope";
	job.inputs_.push_back(spec);

	auto [inputs, error] = resolve_job_inputs(model, job);
	EXPECT_FALSE(inputs.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("not found among graph inputs"), std::string::npos);
}

TEST(RunSupportTest, ProcessOutputsWritesPbAndJson)
{
	const auto dir = std::filesystem::temp_directory_path() / "yirang_run_support_outputs";
	std::filesystem::remove_all(dir);

	OutputSpec spec;
	spec.dir_ = dir.string();
	spec.save_ = true;
	spec.dump_json_ = true;
	spec.stats_ = true;

	const std::vector<Tensor> outputs = { make_float_tensor("Z", { 1, 2 }, { 4.0F, 6.0F }) };
	EXPECT_EQ(process_outputs(outputs, spec), 0);
	EXPECT_TRUE(std::filesystem::exists(dir / "Z.pb"));
	EXPECT_TRUE(std::filesystem::exists(dir / "Z.json"));

	std::filesystem::remove_all(dir);
}

TEST(RunSupportTest, ResolveRejectsFileTensorUnknownName)
{
	onnx::TensorProto proto;
	proto.set_name("Q");
	proto.set_data_type(onnx::TensorProto::FLOAT);
	proto.add_dims(1);
	const float value = 1.0f;
	proto.set_raw_data(&value, sizeof(value));
	std::string serialized;
	ASSERT_TRUE(proto.SerializeToString(&serialized));

	ScopedTempFile file(::testing::UnitTest::GetInstance()->current_test_info()->name() + std::string(".pb"));
	std::ofstream out(file.path(), std::ios::binary);
	out.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
	out.close();

	const OnnxModel model = build_add_model();
	InferenceJob job;
	InputSpec spec;
	spec.path_ = file.path().string();
	job.inputs_.push_back(spec);

	auto [inputs, error] = resolve_job_inputs(model, job);
	EXPECT_FALSE(inputs.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("does not match any graph input"), std::string::npos);
}

TEST(RunSupportTest, ResolveRejectsMissingRequiredInput)
{
	const OnnxModel model = build_two_data_input_model();
	InferenceJob job;
	InputSpec spec;
	spec.name_ = "X";
	spec.random_ = RandomInputSpec{};
	job.inputs_.push_back(spec);

	auto [inputs, error] = resolve_job_inputs(model, job);
	EXPECT_FALSE(inputs.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("missing required graph input(s): 'Y'"), std::string::npos);
}
