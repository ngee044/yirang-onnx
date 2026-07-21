#include "Configurations.h"
#include "InputProject.h"
#include "OnnxModel.h"
#include "RunCommand.h"

#include "ArgumentParser.h"
#include "File.h"
#include "Logger.h"

#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef YIRANG_ONNX_VERSION
#define YIRANG_ONNX_VERSION "0.0.0"
#endif

using namespace Utilities;
using namespace YirangOnnx;

namespace
{
	auto print_usage(LogTypes level) -> void
	{
		Logger::handle().write(level,
							   "usage:\n"
							   "  script  : yirang-onnx --input-script <input_project.json> [--model <path.onnx>]\n"
							   "  inspect : yirang-onnx --model <path.onnx> [--format summary|json|dot] [--out <path>] [--weights true]\n"
							   "  infer   : yirang-onnx --model <path.onnx> --input <in.pb>[,<in2.pb>] [--out-dir <dir>]\n"
							   "  common  : [--title <name>] [--log_root_path <dir>] [--write_console_log <0-8>] [--write_file_log <0-8>] [--write_interval <ms>] "
							   "[--help] [--version]\n"
							   "  log level ints: 0=None 2=Error 3=Warning 4=Information (5+ debug/verbose)\n"
							   "input_project.json: { model, inspect{format,out,weights}, inputs[ path | {name,path} | {name,random{data_type,shape,seed}} | {name} ],\n"
							   "                      dim_overrides{symbol:dim}, run{repeat,warmup}, outputs{dir,save,dump_json,stats} }\n"
							   "  inputs: omit entirely to auto-generate random tensors for every graph input; {name} alone means random defaults for that input.");
	}

	auto render(const OnnxModel& model, const std::string& format, bool include_weights) -> std::string
	{
		if (format == "json")
		{
			return model.to_json(include_weights);
		}
		if (format == "dot")
		{
			return model.to_dot();
		}
		return model.to_summary();
	}

	auto run_inspect(const OnnxModel& model, const InspectSpec& spec) -> int
	{
		const std::string rendered = render(model, spec.format_, spec.include_weights_);
		if (spec.out_path_.empty())
		{
			Logger::handle().write(LogTypes::Information, rendered);
			return 0;
		}

		File out;
		if (auto opened = out.open(spec.out_path_, std::ios::out | std::ios::binary | std::ios::trunc); !opened)
		{
			Logger::handle().write(LogTypes::Error, std::format("cannot open output file '{}': {}", spec.out_path_, opened.error()));
			return 1;
		}
		if (auto written = out.write_bytes(reinterpret_cast<const uint8_t*>(rendered.data()), rendered.size()); !written)
		{
			Logger::handle().write(LogTypes::Error, std::format("cannot write output file '{}': {}", spec.out_path_, written.error()));
			return 1;
		}
		Logger::handle().write(LogTypes::Information, std::format("wrote {} to {}", spec.format_, spec.out_path_));
		return 0;
	}

	auto cli_input_specs(const Configurations& configurations) -> std::vector<InputSpec>
	{
		std::vector<InputSpec> specs;
		for (const auto& path : configurations.input_paths())
		{
			InputSpec spec;
			spec.path_ = path;
			specs.push_back(std::move(spec));
		}
		return specs;
	}

	auto resolve_script_path(const std::string& script_dir, const std::string& path) -> std::string
	{
		if (path.empty() || script_dir.empty())
		{
			return path;
		}
		std::filesystem::path candidate(path);
		if (candidate.is_absolute())
		{
			return path;
		}
		return (std::filesystem::path(script_dir) / candidate).string();
	}

	auto execute(const Configurations& configurations) -> int
	{
		std::optional<InputProject> project;
		std::string script_dir;
		if (!configurations.input_script().empty())
		{
			auto [loaded, error] = InputProject::load(configurations.input_script());
			if (!loaded.has_value())
			{
				Logger::handle().write(LogTypes::Error, error.value_or("cannot load input script"));
				return 2;
			}
			project = std::move(loaded.value());
			script_dir = std::filesystem::path(configurations.input_script()).parent_path().string();
		}

		const bool model_from_script = project.has_value() && !project->model_path().empty();
		const std::string model_path = model_from_script ? resolve_script_path(script_dir, project->model_path()) : configurations.model_path();
		if (model_path.empty())
		{
			Logger::handle().write(LogTypes::Error, std::format("input script '{}' does not set 'model' and no --model was given", configurations.input_script()));
			return 2;
		}

		auto [model, error] = OnnxModel::load(model_path);
		if (!model.has_value())
		{
			Logger::handle().write(LogTypes::Error, error.value_or("unknown error"));
			return 1;
		}

		if (!project.has_value())
		{
			if (!configurations.input_paths().empty())
			{
				InferenceJob job;
				job.inputs_ = cli_input_specs(configurations);
				job.outputs_.dir_ = configurations.output_dir();
				job.tuning_ = configurations.session_tuning();
				return run_inference(model.value(), model_path, job);
			}

			InspectSpec spec;
			spec.format_ = configurations.output_format();
			spec.out_path_ = configurations.output_path();
			spec.include_weights_ = configurations.include_weights();
			return run_inspect(model.value(), spec);
		}

		bool executed = false;
		if (auto inspect = project->inspect(); inspect.has_value())
		{
			InspectSpec spec = inspect.value();
			if (!spec.has_format_)
			{
				spec.format_ = configurations.output_format();
			}
			if (!spec.has_out_path_)
			{
				spec.out_path_ = configurations.output_path();
			}
			else
			{
				spec.out_path_ = resolve_script_path(script_dir, spec.out_path_);
			}
			if (!spec.has_include_weights_)
			{
				spec.include_weights_ = configurations.include_weights();
			}
			if (int code = run_inspect(model.value(), spec); code != 0)
			{
				return code;
			}
			executed = true;
		}

		if (project->infer_requested())
		{
			InferenceJob job;
			job.inputs_ = project->inputs();
			if (job.inputs_.empty() && !configurations.input_paths().empty())
			{
				job.inputs_ = cli_input_specs(configurations);
			}
			else
			{
				for (auto& spec : job.inputs_)
				{
					spec.path_ = resolve_script_path(script_dir, spec.path_);
				}
			}
			job.dim_overrides_ = project->dim_overrides();
			job.run_ = project->run();
			job.outputs_ = project->outputs();
			if (!job.outputs_.dir_.empty())
			{
				job.outputs_.dir_ = resolve_script_path(script_dir, job.outputs_.dir_);
			}
			job.tuning_ = configurations.session_tuning();
			if (job.outputs_.dir_.empty())
			{
				job.outputs_.dir_ = configurations.output_dir();
			}
			return run_inference(model.value(), model_path, job);
		}

		if (!executed)
		{
			InspectSpec spec;
			spec.format_ = configurations.output_format();
			spec.out_path_ = configurations.output_path();
			spec.include_weights_ = configurations.include_weights();
			return run_inspect(model.value(), spec);
		}
		return 0;
	}
} // namespace

auto main(int argc, char* argv[]) -> int
{
	const std::vector<std::string> cli_arguments(argv + 1, argv + argc);
	const auto unknown_flag = Configurations::find_unknown_flag(cli_arguments);

	auto configurations = std::make_shared<Configurations>(ArgumentParser(argc, argv));

	const bool direct_output = configurations->show_version() || configurations->show_help() || configurations->invalid_reason().has_value() || unknown_flag.has_value();

	LogTypes console_mode = configurations->write_console();
	if (direct_output && console_mode < LogTypes::Information)
	{
		console_mode = LogTypes::Information;
	}

	LogTypes file_mode = configurations->write_file();
	if (configurations->show_version() || configurations->show_help())
	{
		file_mode = LogTypes::None;
	}

	Logger::handle().file_mode(file_mode);
	Logger::handle().console_mode(console_mode);
	Logger::handle().write_interval(configurations->write_interval());
	Logger::handle().log_root(configurations->log_root_path());

	Logger::handle().start(configurations->app_title());

	if (auto warning = configurations->load_warning(); warning.has_value())
	{
		Logger::handle().write(LogTypes::Warning, warning.value());
	}

	int exit_code = 0;

	if (configurations->show_version())
	{
		Logger::handle().write(LogTypes::Information, std::format("yirang-onnx {}", YIRANG_ONNX_VERSION));
	}
	else if (configurations->show_help())
	{
		print_usage(LogTypes::Information);
	}
	else if (unknown_flag.has_value())
	{
		Logger::handle().write(LogTypes::Error, std::format("unknown argument '{}'", unknown_flag.value()));
		print_usage(LogTypes::Error);
		exit_code = 2;
	}
	else if (auto reason = configurations->invalid_reason(); reason.has_value())
	{
		Logger::handle().write(LogTypes::Error, std::format("invalid configuration: {}", reason.value()));
		print_usage(LogTypes::Error);
		exit_code = 2;
	}
	else
	{
		try
		{
			exit_code = execute(*configurations);
		}
		catch (const std::exception& e)
		{
			Logger::handle().write(LogTypes::Error, std::format("fatal error: {}", e.what()));
			exit_code = 1;
		}
	}

	configurations.reset();

	Logger::handle().stop();
	Logger::destroy();

	return exit_code;
}
