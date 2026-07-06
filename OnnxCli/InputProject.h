#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace YirangOnnx
{
	struct RandomInputSpec
	{
		std::string data_type_;
		std::vector<int64_t> shape_;
		std::optional<uint64_t> seed_;
	};

	struct InputSpec
	{
		std::string name_;
		std::string path_;
		std::optional<RandomInputSpec> random_;
	};

	struct RunSpec
	{
		uint32_t repeat_ = 1;
		uint32_t warmup_ = 0;
	};

	struct OutputSpec
	{
		std::string dir_;
		bool save_ = true;
		bool dump_json_ = false;
		bool stats_ = true;
	};

	struct InspectSpec
	{
		std::string format_ = "summary";
		std::string out_path_;
		bool include_weights_ = false;
	};

	class InputProject
	{
	public:
		InputProject(void) = default;

		static auto load(const std::string& path) -> std::tuple<std::optional<InputProject>, std::optional<std::string>>;
		static auto parse(const std::string& json_text, const std::string& origin) -> std::tuple<std::optional<InputProject>, std::optional<std::string>>;

		auto model_path(void) const -> std::string;
		auto inputs(void) const -> std::vector<InputSpec>;
		auto dim_overrides(void) const -> std::map<std::string, int64_t>;
		auto run(void) const -> RunSpec;
		auto outputs(void) const -> OutputSpec;
		auto inspect(void) const -> std::optional<InspectSpec>;
		auto infer_requested(void) const -> bool;

	private:
		std::string model_path_;
		std::vector<InputSpec> inputs_;
		std::map<std::string, int64_t> dim_overrides_;
		RunSpec run_;
		OutputSpec outputs_;
		std::optional<InspectSpec> inspect_;
		bool infer_requested_ = false;
	};
} // namespace YirangOnnx
