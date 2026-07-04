#pragma once

#include "ArgumentParser.h"
#include "LogTypes.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using namespace Utilities;

namespace YirangOnnx
{
	class Configurations
	{
	public:
		Configurations(ArgumentParser&& arguments, const std::string& config_file_name = "yirang_onnx_configurations.json");
		virtual ~Configurations(void);

		auto model_path(void) const -> std::string;
		auto output_format(void) const -> std::string;
		auto output_path(void) const -> std::string;
		auto include_weights(void) const -> bool;
		auto input_paths(void) const -> std::vector<std::string>;
		auto output_dir(void) const -> std::string;
		auto app_title(void) const -> std::string;

		auto log_root_path(void) const -> std::string;
		auto write_console(void) const -> LogTypes;
		auto write_file(void) const -> LogTypes;
		auto write_interval(void) const -> uint16_t;

		auto invalid_reason(void) const -> std::optional<std::string>;
		auto load_warning(void) const -> std::optional<std::string>;

	protected:
		auto load(void) -> void;
		auto parse(ArgumentParser& arguments) -> void;
		auto validate(void) -> void;

	private:
		std::string root_path_;
		std::string config_file_name_;

		std::string model_path_;
		std::string output_format_;
		std::string output_path_;
		bool include_weights_;
		std::vector<std::string> input_paths_;
		std::string output_dir_;
		std::string app_title_;

		std::string log_root_path_;
		LogTypes write_console_;
		LogTypes write_file_;
		uint16_t write_interval_;

		std::optional<std::string> invalid_reason_;
		std::optional<std::string> load_warning_;
	};
} // namespace YirangOnnx
