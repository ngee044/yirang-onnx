#include "Configurations.h"

#include "ArgumentParser.h"
#include "File.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace Utilities;
using namespace YirangOnnx;

namespace
{
	constexpr const char* missing_config = "onnx_parser_tests_missing_configurations.json";

	auto make_parser(std::vector<std::string> args) -> ArgumentParser
	{
		args.insert(args.begin(), "onnx_parser_tests");
		std::vector<char*> argv;
		argv.reserve(args.size());
		for (auto& argument : args)
		{
			argv.push_back(argument.data());
		}
		return ArgumentParser(static_cast<int32_t>(argv.size()), argv.data());
	}

	class ScopedConfigFile
	{
	public:
		ScopedConfigFile(const std::string& file_name, const std::string& content)
		{
			path_ = std::filesystem::path(make_parser({}).program_folder()) / file_name;
			File file;
			if (auto opened = file.open(path_.string(), std::ios::out | std::ios::binary | std::ios::trunc); !opened)
			{
				ADD_FAILURE() << "cannot open " << path_.string() << ": " << opened.error();
				return;
			}
			if (auto written = file.write_bytes(std::vector<uint8_t>(content.begin(), content.end())); !written)
			{
				ADD_FAILURE() << "cannot write " << path_.string() << ": " << written.error();
			}
		}

		~ScopedConfigFile(void)
		{
			std::error_code ec;
			std::filesystem::remove(path_, ec);
		}

	private:
		std::filesystem::path path_;
	};
} // namespace

TEST(ConfigurationsTest, AppliesDefaultsAndRequiresModel)
{
	Configurations configurations(make_parser({}), missing_config);

	EXPECT_EQ(configurations.output_format(), "summary");
	EXPECT_EQ(configurations.app_title(), "yirang-onnx");
	EXPECT_FALSE(configurations.include_weights());
	EXPECT_TRUE(configurations.input_paths().empty());
	EXPECT_EQ(configurations.write_interval(), 1000);
	EXPECT_FALSE(configurations.load_warning().has_value());

	ASSERT_TRUE(configurations.invalid_reason().has_value());
	EXPECT_NE(configurations.invalid_reason()->find("--model"), std::string::npos);
}

TEST(ConfigurationsTest, ParsesCliArguments)
{
	Configurations configurations(
		make_parser({ "--model", "model.onnx", "--format", "dot", "--weights", "true", "--input", "a.pb,b.pb", "--out-dir", "outputs", "--write_interval", "50" }),
		missing_config);

	EXPECT_FALSE(configurations.invalid_reason().has_value());
	EXPECT_EQ(configurations.model_path(), "model.onnx");
	EXPECT_EQ(configurations.output_format(), "dot");
	EXPECT_TRUE(configurations.include_weights());
	ASSERT_EQ(configurations.input_paths().size(), 2u);
	EXPECT_EQ(configurations.input_paths()[1], "b.pb");
	EXPECT_EQ(configurations.output_dir(), "outputs");
	EXPECT_EQ(configurations.write_interval(), 100);
}

TEST(ConfigurationsTest, RejectsUnknownFormat)
{
	Configurations configurations(make_parser({ "--model", "model.onnx", "--format", "yaml" }), missing_config);

	ASSERT_TRUE(configurations.invalid_reason().has_value());
	EXPECT_NE(configurations.invalid_reason()->find("yaml"), std::string::npos);
}

TEST(ConfigurationsTest, LoadsEngineSettingsAndIgnoresJobKeys)
{
	const std::string config_name = "onnx_parser_tests_configurations.json";
	ScopedConfigFile config(config_name, R"({ "app_title": "from-json", "write_interval": 500, "output_format": "dot", "model_path": "ignored.onnx" })");

	Configurations from_json(make_parser({ "--model", "model.onnx" }), config_name);
	EXPECT_EQ(from_json.app_title(), "from-json");
	EXPECT_EQ(from_json.write_interval(), 500);
	EXPECT_EQ(from_json.output_format(), "summary");
	EXPECT_EQ(from_json.model_path(), "model.onnx");
	EXPECT_FALSE(from_json.load_warning().has_value());

	Configurations overridden(make_parser({ "--model", "model.onnx", "--title", "from-cli" }), config_name);
	EXPECT_EQ(overridden.app_title(), "from-cli");
}

TEST(ConfigurationsTest, AcceptsInputScriptWithoutModel)
{
	Configurations configurations(make_parser({ "--input-script", "input_project.json" }), missing_config);

	EXPECT_FALSE(configurations.invalid_reason().has_value());
	EXPECT_EQ(configurations.input_script(), "input_project.json");
	EXPECT_TRUE(configurations.model_path().empty());
}

TEST(ConfigurationsTest, HelpAndVersionSkipValidation)
{
	Configurations help(make_parser({ "--help" }), missing_config);
	EXPECT_TRUE(help.show_help());
	EXPECT_FALSE(help.invalid_reason().has_value());

	Configurations version(make_parser({ "--version" }), missing_config);
	EXPECT_TRUE(version.show_version());
	EXPECT_FALSE(version.invalid_reason().has_value());
}

TEST(ConfigurationsTest, WarnsOnBrokenJson)
{
	const std::string config_name = "onnx_parser_tests_broken_configurations.json";
	ScopedConfigFile config(config_name, "{ this is not json");

	Configurations configurations(make_parser({ "--model", "model.onnx" }), config_name);

	ASSERT_TRUE(configurations.load_warning().has_value());
	EXPECT_NE(configurations.load_warning()->find(config_name), std::string::npos);
	EXPECT_FALSE(configurations.invalid_reason().has_value());
}
