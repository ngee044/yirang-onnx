#include "InputProject.h"

#include <gtest/gtest.h>

#include <string>

using namespace YirangOnnx;

TEST(InputProjectTest, ParsesFullSchema)
{
	const std::string script = R"({
		"model": "models/denoiser.onnx",
		"inspect": { "format": "json", "out": "report.json", "weights": true },
		"inputs": [
			{ "name": "audio", "path": "tensors/audio.pb" },
			{ "name": "mask", "random": { "data_type": "FLOAT", "shape": [1, 257], "seed": 42 } }
		],
		"dim_overrides": { "N": 4 },
		"run": { "repeat": 10, "warmup": 2 },
		"outputs": { "dir": "outputs", "save": false, "dump_json": true, "stats": false }
	})";

	auto [project, error] = InputProject::parse(script, "test");
	ASSERT_TRUE(project.has_value()) << error.value_or("");

	EXPECT_EQ(project->model_path(), "models/denoiser.onnx");
	EXPECT_TRUE(project->infer_requested());

	ASSERT_TRUE(project->inspect().has_value());
	EXPECT_EQ(project->inspect()->format_, "json");
	EXPECT_EQ(project->inspect()->out_path_, "report.json");
	EXPECT_TRUE(project->inspect()->include_weights_);

	const auto inputs = project->inputs();
	ASSERT_EQ(inputs.size(), 2u);
	EXPECT_EQ(inputs[0].name_, "audio");
	EXPECT_EQ(inputs[0].path_, "tensors/audio.pb");
	EXPECT_FALSE(inputs[0].random_.has_value());
	ASSERT_TRUE(inputs[1].random_.has_value());
	EXPECT_EQ(inputs[1].random_->data_type_, "FLOAT");
	ASSERT_EQ(inputs[1].random_->shape_.size(), 2u);
	EXPECT_EQ(inputs[1].random_->shape_[1], 257);
	EXPECT_EQ(inputs[1].random_->seed_, std::optional<uint64_t>(42));

	const auto overrides = project->dim_overrides();
	ASSERT_EQ(overrides.size(), 1u);
	EXPECT_EQ(overrides.at("N"), 4);

	EXPECT_EQ(project->run().repeat_, 10u);
	EXPECT_EQ(project->run().warmup_, 2u);

	EXPECT_EQ(project->outputs().dir_, "outputs");
	EXPECT_FALSE(project->outputs().save_);
	EXPECT_TRUE(project->outputs().dump_json_);
	EXPECT_FALSE(project->outputs().stats_);
}

TEST(InputProjectTest, AppliesDefaults)
{
	auto [project, error] = InputProject::parse(R"({ "model": "m.onnx" })", "test");
	ASSERT_TRUE(project.has_value()) << error.value_or("");

	EXPECT_FALSE(project->infer_requested());
	EXPECT_FALSE(project->inspect().has_value());
	EXPECT_TRUE(project->inputs().empty());
	EXPECT_EQ(project->run().repeat_, 1u);
	EXPECT_EQ(project->run().warmup_, 0u);
	EXPECT_TRUE(project->outputs().save_);
	EXPECT_FALSE(project->outputs().dump_json_);
	EXPECT_TRUE(project->outputs().stats_);
}

TEST(InputProjectTest, SupportsInputShorthands)
{
	auto [project, error] = InputProject::parse(R"({ "inputs": ["a.pb", { "name": "X" }] })", "test");
	ASSERT_TRUE(project.has_value()) << error.value_or("");

	EXPECT_TRUE(project->infer_requested());
	const auto inputs = project->inputs();
	ASSERT_EQ(inputs.size(), 2u);
	EXPECT_EQ(inputs[0].path_, "a.pb");
	EXPECT_TRUE(inputs[1].random_.has_value());
	EXPECT_EQ(inputs[1].name_, "X");
}

TEST(InputProjectTest, RunSectionAloneRequestsInference)
{
	auto [project, error] = InputProject::parse(R"({ "model": "m.onnx", "run": {} })", "test");
	ASSERT_TRUE(project.has_value()) << error.value_or("");
	EXPECT_TRUE(project->infer_requested());
	EXPECT_TRUE(project->inputs().empty());
}

TEST(InputProjectTest, RejectsInvalidScripts)
{
	{
		auto [project, error] = InputProject::parse("{ broken", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"([1, 2])", "test");
		EXPECT_FALSE(project.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inspect": { "format": "yaml" } })", "test");
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("yaml"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "random": {} } ] })", "test");
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("name"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ {} ] })", "test");
		EXPECT_FALSE(project.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "X", "path": "a.pb", "random": {} } ] })", "test");
		EXPECT_FALSE(project.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"({ "dim_overrides": { "N": 0 } })", "test");
		EXPECT_FALSE(project.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"({ "run": { "repeat": 0 } })", "test");
		EXPECT_FALSE(project.has_value());
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "X", "random": { "shape": [0] } } ] })", "test");
		EXPECT_FALSE(project.has_value());
	}
}

TEST(InputProjectTest, RejectsUnknownKeys)
{
	{
		auto [project, error] = InputProject::parse(R"({ "modl": "m.onnx" })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("modl"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "run": { "reepeat": 5 } })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("reepeat"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "audio", "pat": "audio.pb" } ] })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("pat"), std::string::npos);
	}
}

TEST(InputProjectTest, AcceptsUnderscoreCommentKeys)
{
	auto [project, error] = InputProject::parse(R"({ "_comment": "note", "_note": 1, "model": "m.onnx" })", "test");
	ASSERT_TRUE(project.has_value()) << error.value_or("");
	EXPECT_EQ(project->model_path(), "m.onnx");
}

TEST(InputProjectTest, RejectsInvalidRandomDataType)
{
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "X", "random": { "data_type": "flaot" } } ] })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("flaot"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "X", "random": { "data_type": "FLOAT16" } } ] })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("FLOAT16"), std::string::npos);
	}
}

TEST(InputProjectTest, RejectsOversizedShapeAndRunBounds)
{
	{
		auto [project, error] = InputProject::parse(R"({ "inputs": [ { "name": "X", "random": { "shape": [1000000000, 1000000000] } } ] })", "test");
		EXPECT_FALSE(project.has_value());
		ASSERT_TRUE(error.has_value());
		EXPECT_NE(error->find("exceeds limit"), std::string::npos);
	}
	{
		auto [project, error] = InputProject::parse(R"({ "run": { "repeat": 2000000 } })", "test");
		EXPECT_FALSE(project.has_value());
	}
}

TEST(InputProjectTest, LoadFailsForMissingFile)
{
	auto [project, error] = InputProject::load("/no/such/input_project.json");
	EXPECT_FALSE(project.has_value());
	EXPECT_TRUE(error.has_value());
}
