#include "Configurations.h"

#include "Converter.h"
#include "File.h"

#include <filesystem>
#include <format>

#include <boost/json.hpp>

namespace YirangOnnx
{
	namespace
	{
		auto to_log_types(int64_t value) -> LogTypes { return static_cast<LogTypes>(value); }
	} // namespace

	Configurations::Configurations(ArgumentParser&& arguments, const std::string& config_file_name)
		: config_file_name_(config_file_name)
		, output_format_("summary")
		, include_weights_(false)
		, app_title_("yirang-onnx")
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
	auto Configurations::app_title(void) const -> std::string { return app_title_; }
	auto Configurations::log_root_path(void) const -> std::string { return log_root_path_; }
	auto Configurations::write_console(void) const -> LogTypes { return write_console_; }
	auto Configurations::write_file(void) const -> LogTypes { return write_file_; }
	auto Configurations::write_interval(void) const -> uint16_t { return write_interval_; }
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

		if (message.contains("model_path") && message.at("model_path").is_string())
		{
			model_path_ = message.at("model_path").as_string().c_str();
		}
		if (message.contains("output_format") && message.at("output_format").is_string())
		{
			output_format_ = message.at("output_format").as_string().c_str();
		}
		if (message.contains("output_path") && message.at("output_path").is_string())
		{
			output_path_ = message.at("output_path").as_string().c_str();
		}
		if (message.contains("include_weights") && message.at("include_weights").is_bool())
		{
			include_weights_ = message.at("include_weights").as_bool();
		}
		if (message.contains("app_title") && message.at("app_title").is_string())
		{
			app_title_ = message.at("app_title").as_string().c_str();
		}
		if (message.contains("log_root_path") && message.at("log_root_path").is_string())
		{
			log_root_path_ = message.at("log_root_path").as_string().c_str();
		}
		if (message.contains("write_console") && message.at("write_console").is_number())
		{
			write_console_ = to_log_types(message.at("write_console").as_int64());
		}
		if (message.contains("write_file") && message.at("write_file").is_number())
		{
			write_file_ = to_log_types(message.at("write_file").as_int64());
		}
		if (message.contains("write_interval") && message.at("write_interval").is_number())
		{
			write_interval_ = static_cast<uint16_t>(message.at("write_interval").as_int64());
		}
	}

	auto Configurations::parse(ArgumentParser& arguments) -> void
	{
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
		if (auto value = arguments.to_string("--title"); value.has_value())
		{
			app_title_ = value.value();
		}
		if (auto value = arguments.to_string("--log_root_path"); value.has_value())
		{
			log_root_path_ = value.value();
		}
		if (auto value = arguments.to_int("--write_console_log"); value.has_value())
		{
			write_console_ = to_log_types(value.value());
		}
		if (auto value = arguments.to_int("--write_file_log"); value.has_value())
		{
			write_file_ = to_log_types(value.value());
		}
		if (auto value = arguments.to_ushort("--write_interval"); value.has_value())
		{
			write_interval_ = value.value();
		}
	}

	auto Configurations::validate(void) -> void
	{
		if (write_interval_ < 100)
		{
			write_interval_ = 100;
		}

		if (output_format_ != "summary" && output_format_ != "json" && output_format_ != "dot")
		{
			invalid_reason_ = std::format("unknown --format '{}' (expected summary|json|dot)", output_format_);
			return;
		}

		if (model_path_.empty())
		{
			invalid_reason_ = "no --model <path.onnx> provided";
			return;
		}
	}
} // namespace YirangOnnx
