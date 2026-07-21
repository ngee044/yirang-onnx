#include "Configurations.h"

#include "Converter.h"
#include "File.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <string_view>
#include <vector>

#include <boost/json.hpp>

namespace YirangOnnx
{
	namespace
	{
		auto to_log_types(int64_t value) -> LogTypes { return static_cast<LogTypes>(value); }

		auto to_int64(const boost::json::value& value) -> std::optional<int64_t>
		{
			if (value.is_int64())
			{
				return value.as_int64();
			}
			if (value.is_uint64() && value.as_uint64() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
			{
				return static_cast<int64_t>(value.as_uint64());
			}
			return std::nullopt;
		}

		auto is_valid_log_level(int64_t value) -> bool { return value >= static_cast<int64_t>(LogTypes::None) && value <= static_cast<int64_t>(LogTypes::Packet); }

		constexpr std::array<std::string_view, 14> known_cli_flags
			= { "--input-script",	   "--model",		   "--format",		   "--out",	 "--weights", "--input", "--out-dir", "--title", "--log_root_path",
				"--write_console_log", "--write_file_log", "--write_interval", "--help", "--version" };

		constexpr int64_t max_thread_count = 4096;

		auto to_execution_mode(const boost::json::value& value) -> std::optional<ExecutionMode>
		{
			if (!value.is_string())
			{
				return std::nullopt;
			}
			const std::string mode(value.as_string().c_str());
			if (mode == "sequential")
			{
				return ExecutionMode::sequential;
			}
			if (mode == "parallel")
			{
				return ExecutionMode::parallel;
			}
			return std::nullopt;
		}

		auto to_graph_optimization(const boost::json::value& value) -> std::optional<GraphOptimization>
		{
			if (!value.is_string())
			{
				return std::nullopt;
			}
			const std::string level(value.as_string().c_str());
			if (level == "disabled")
			{
				return GraphOptimization::disabled;
			}
			if (level == "basic")
			{
				return GraphOptimization::basic;
			}
			if (level == "extended")
			{
				return GraphOptimization::extended;
			}
			if (level == "all")
			{
				return GraphOptimization::all;
			}
			return std::nullopt;
		}
	} // namespace

	Configurations::Configurations(ArgumentParser&& arguments, const std::string& config_file_name)
		: config_file_name_(config_file_name)
		, output_format_("summary")
		, include_weights_(false)
		, app_title_("yirang-onnx")
		, show_help_(false)
		, show_version_(false)
		, write_console_(LogTypes::Information)
		, write_file_(LogTypes::Information)
		, write_interval_(1000)
	{
		root_path_ = arguments.program_folder();

		load();
		parse(arguments);
		validate();
	}

	Configurations::~Configurations(void) {}

	auto Configurations::model_path(void) const -> std::string { return model_path_; }
	auto Configurations::output_format(void) const -> std::string { return output_format_; }
	auto Configurations::output_path(void) const -> std::string { return output_path_; }
	auto Configurations::include_weights(void) const -> bool { return include_weights_; }
	auto Configurations::input_paths(void) const -> std::vector<std::string> { return input_paths_; }
	auto Configurations::output_dir(void) const -> std::string { return output_dir_; }
	auto Configurations::input_script(void) const -> std::string { return input_script_; }
	auto Configurations::app_title(void) const -> std::string { return app_title_; }
	auto Configurations::show_help(void) const -> bool { return show_help_; }
	auto Configurations::show_version(void) const -> bool { return show_version_; }
	auto Configurations::log_root_path(void) const -> std::string { return log_root_path_; }
	auto Configurations::write_console(void) const -> LogTypes { return write_console_; }
	auto Configurations::write_file(void) const -> LogTypes { return write_file_; }
	auto Configurations::write_interval(void) const -> uint16_t { return write_interval_; }
	auto Configurations::session_tuning(void) const -> SessionTuning { return session_tuning_; }
	auto Configurations::invalid_reason(void) const -> std::optional<std::string> { return invalid_reason_; }
	auto Configurations::load_warning(void) const -> std::optional<std::string> { return load_warning_; }

	auto Configurations::load(void) -> void
	{
		const std::filesystem::path path = std::filesystem::path(root_path_) / config_file_name_;
		if (!std::filesystem::exists(path))
		{
			return;
		}

		File source;
		if (auto opened = source.open(path.string(), std::ios::in | std::ios::binary); !opened)
		{
			load_warning_ = std::format("cannot open configuration '{}': {}", path.string(), opened.error());
			return;
		}

		auto read = source.read_bytes();
		if (!read)
		{
			load_warning_ = std::format("cannot read configuration '{}': {}", path.string(), read.error());
			return;
		}

		boost::json::value parsed;
		try
		{
			parsed = boost::json::parse(Converter::to_string(read.value()));
		}
		catch (const std::exception& e)
		{
			load_warning_ = std::format("cannot parse configuration '{}': {}", path.string(), e.what());
			return;
		}

		if (!parsed.is_object())
		{
			load_warning_ = std::format("configuration '{}' is not a JSON object; ignored", path.string());
			return;
		}
		const auto& message = parsed.as_object();

		static constexpr std::array<std::string_view, 11> known_keys{ "app_title",		  "log_root_path",		"write_console",
																	  "write_file",		  "write_interval",		"intra_op_threads",
																	  "inter_op_threads", "enable_mem_pattern", "enable_cpu_mem_arena",
																	  "execution_mode",	  "graph_optimization" };

		std::vector<std::string> warnings;

		if (const auto* value = message.if_contains("app_title"))
		{
			if (value->is_string())
			{
				app_title_ = value->as_string().c_str();
			}
			else
			{
				warnings.push_back("'app_title' must be a string; kept default");
			}
		}
		if (const auto* value = message.if_contains("log_root_path"))
		{
			if (value->is_string())
			{
				log_root_path_ = value->as_string().c_str();
			}
			else
			{
				warnings.push_back("'log_root_path' must be a string; kept default");
			}
		}
		if (const auto* value = message.if_contains("write_console"))
		{
			if (auto level = to_int64(*value); level.has_value() && is_valid_log_level(level.value()))
			{
				write_console_ = to_log_types(level.value());
			}
			else
			{
				warnings.push_back(
					std::format("'write_console' must be an integer in [{}, {}]; kept default", static_cast<int>(LogTypes::None), static_cast<int>(LogTypes::Packet)));
			}
		}
		if (const auto* value = message.if_contains("write_file"))
		{
			if (auto level = to_int64(*value); level.has_value() && is_valid_log_level(level.value()))
			{
				write_file_ = to_log_types(level.value());
			}
			else
			{
				warnings.push_back(
					std::format("'write_file' must be an integer in [{}, {}]; kept default", static_cast<int>(LogTypes::None), static_cast<int>(LogTypes::Packet)));
			}
		}
		if (const auto* value = message.if_contains("write_interval"))
		{
			if (auto interval = to_int64(*value);
				interval.has_value() && interval.value() >= 0 && interval.value() <= static_cast<int64_t>(std::numeric_limits<uint16_t>::max()))
			{
				write_interval_ = static_cast<uint16_t>(interval.value());
			}
			else
			{
				warnings.push_back(std::format("'write_interval' must be an integer in [0, {}]; kept default", std::numeric_limits<uint16_t>::max()));
			}
		}
		if (const auto* value = message.if_contains("intra_op_threads"))
		{
			if (auto threads = to_int64(*value); threads.has_value() && threads.value() >= 0 && threads.value() <= max_thread_count)
			{
				session_tuning_.intra_op_threads_ = static_cast<int>(threads.value());
			}
			else
			{
				warnings.push_back(std::format("'intra_op_threads' must be an integer in [0, {}]; kept default", max_thread_count));
			}
		}
		if (const auto* value = message.if_contains("inter_op_threads"))
		{
			if (auto threads = to_int64(*value); threads.has_value() && threads.value() >= 0 && threads.value() <= max_thread_count)
			{
				session_tuning_.inter_op_threads_ = static_cast<int>(threads.value());
			}
			else
			{
				warnings.push_back(std::format("'inter_op_threads' must be an integer in [0, {}]; kept default", max_thread_count));
			}
		}
		if (const auto* value = message.if_contains("enable_mem_pattern"))
		{
			if (value->is_bool())
			{
				session_tuning_.enable_mem_pattern_ = value->as_bool();
			}
			else
			{
				warnings.push_back("'enable_mem_pattern' must be a boolean; kept default");
			}
		}
		if (const auto* value = message.if_contains("enable_cpu_mem_arena"))
		{
			if (value->is_bool())
			{
				session_tuning_.enable_cpu_mem_arena_ = value->as_bool();
			}
			else
			{
				warnings.push_back("'enable_cpu_mem_arena' must be a boolean; kept default");
			}
		}
		if (const auto* value = message.if_contains("execution_mode"))
		{
			if (auto mode = to_execution_mode(*value); mode.has_value())
			{
				session_tuning_.execution_mode_ = mode.value();
			}
			else
			{
				warnings.push_back("'execution_mode' must be 'sequential' or 'parallel'; kept default");
			}
		}
		if (const auto* value = message.if_contains("graph_optimization"))
		{
			if (auto level = to_graph_optimization(*value); level.has_value())
			{
				session_tuning_.graph_optimization_ = level.value();
			}
			else
			{
				warnings.push_back("'graph_optimization' must be one of disabled|basic|extended|all; kept default");
			}
		}

		for (const auto& entry : message)
		{
			const std::string key(entry.key());
			if (!key.empty() && key.front() == '_')
			{
				continue;
			}
			bool known = false;
			for (const auto& candidate : known_keys)
			{
				if (key == candidate)
				{
					known = true;
					break;
				}
			}
			if (!known)
			{
				warnings.push_back(std::format("unknown key '{}' (ignored)", key));
			}
		}

		if (!warnings.empty())
		{
			std::string joined;
			for (size_t i = 0; i < warnings.size(); ++i)
			{
				joined += (i == 0 ? "" : "; ");
				joined += warnings[i];
			}
			load_warning_ = std::format("configuration '{}': {}", path.string(), joined);
		}
	}

	auto Configurations::find_unknown_flag(const std::vector<std::string>& cli_arguments) -> std::optional<std::string>
	{
		for (const auto& token : cli_arguments)
		{
			if (token.rfind("--", 0) != 0)
			{
				continue;
			}
			if (std::find(known_cli_flags.begin(), known_cli_flags.end(), token) == known_cli_flags.end())
			{
				return token;
			}
		}
		return std::nullopt;
	}

	auto Configurations::parse(ArgumentParser& arguments) -> void
	{
		const auto parse_log_level = [&arguments, this](const std::string& key) -> std::optional<LogTypes>
		{
			auto raw = arguments.to_string(key);
			if (!raw.has_value())
			{
				return std::nullopt;
			}
			int64_t level = 0;
			auto [ptr, ec] = std::from_chars(raw->data(), raw->data() + raw->size(), level);
			if (ec != std::errc() || ptr != raw->data() + raw->size() || !is_valid_log_level(level))
			{
				invalid_reason_ = std::format("invalid {} '{}' (expected an integer in [0, 8])", key, raw.value());
				return std::nullopt;
			}
			return to_log_types(level);
		};

		if (auto value = arguments.to_string("--model"); value.has_value())
		{
			model_path_ = value.value();
		}
		if (auto value = arguments.to_string("--format"); value.has_value())
		{
			output_format_ = value.value();
		}
		if (auto value = arguments.to_string("--out"); value.has_value())
		{
			output_path_ = value.value();
		}
		if (auto value = arguments.to_bool("--weights"); value.has_value())
		{
			include_weights_ = value.value();
		}
		if (auto value = arguments.to_array("--input"); value.has_value())
		{
			input_paths_ = value.value();
		}
		if (auto value = arguments.to_string("--out-dir"); value.has_value())
		{
			output_dir_ = value.value();
		}
		if (auto value = arguments.to_string("--input-script"); value.has_value())
		{
			input_script_ = value.value();
		}
		if (auto value = arguments.to_string("--title"); value.has_value())
		{
			app_title_ = value.value();
		}
		if (auto value = arguments.to_string("--log_root_path"); value.has_value())
		{
			log_root_path_ = value.value();
		}
		if (auto value = parse_log_level("--write_console_log"); value.has_value())
		{
			write_console_ = value.value();
		}
		if (auto value = parse_log_level("--write_file_log"); value.has_value())
		{
			write_file_ = value.value();
		}
		if (auto value = arguments.to_ushort("--write_interval"); value.has_value())
		{
			write_interval_ = value.value();
		}

		show_help_ = arguments.to_string("--help").has_value();
		show_version_ = arguments.to_string("--version").has_value();
	}

	auto Configurations::validate(void) -> void
	{
		if (write_interval_ < 100)
		{
			const std::string note = std::format("write_interval {} raised to the 100 ms minimum", write_interval_);
			load_warning_ = load_warning_.has_value() ? std::format("{}; {}", load_warning_.value(), note) : note;
			write_interval_ = 100;
		}

		if (show_help_ || show_version_)
		{
			return;
		}

		if (output_format_ != "summary" && output_format_ != "json" && output_format_ != "dot")
		{
			invalid_reason_ = std::format("unknown --format '{}' (expected summary|json|dot)", output_format_);
			return;
		}

		if (model_path_.empty() && input_script_.empty())
		{
			invalid_reason_ = "no --model <path.onnx> or --input-script <project.json> provided";
			return;
		}
	}
} // namespace YirangOnnx
