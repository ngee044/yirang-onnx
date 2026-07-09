#include "Configurations.h"
#include "InputProject.h"
#include "OnnxModel.h"
#include "RunCommand.h"

#include "ArgumentParser.h"
#include "File.h"
#include "Logger.h"

#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifndef YIRANG_ONNX_VERSION
#define YIRANG_ONNX_VERSION "0.0.0"
#endif

using namespace Utilities;
using namespace YirangOnnx;

namespace
{
	auto print_usage(LogTypes level) -> void
	{
		Logger::handle().write(level, "usage:\n"
									  "  script  : yirang-onnx --input-script <input_project.json> [--model <path.onnx>]\n"
									  "  inspect : yirang-onnx --model <path.onnx> [--format summary|json|dot] [--out <path>] [--weights true]\n"
									  "  infer   : yirang-onnx --model <path.onnx> --input <in.pb>[,<in2.pb>] [--out-dir <dir>]\n"
									  "  common  : [--title <name>] [--log_root_path <dir>] [--write_console_log <LogTypes>] [--write_file_log <LogTypes>] [--help] "
									  "[--version]\n"
									  "input_project.json: { model, inspect{format,out,weights}, inputs[ path | {name,path} | {name,random{data_type,shape,seed}} ],\n"
									  "                      dim_overrides{symbol:dim}, run{repeat,warmup}, outputs{dir,save,dump_json,stats} }");
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

	auto execute(const Configurations& configurations) -> int
	{
		std::optional<InputProject> project;
		if (!configurations.input_script().empty())
		{
			auto [loaded, error] = InputProject::load(configurations.input_script());
			if (!loaded.has_value())
			{
				Logger::handle().write(LogTypes::Error, error.value_or("cannot load input script"));
				return 2;
			}
			project = std::move(loaded.value());
		}

		const std::string model_path = (project.has_value() && !project->model_path().empty()) ? project->model_path() : configurations.model_path();
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
			if (int code = run_inspect(model.value(), inspect.value()); code != 0)
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
			job.dim_overrides_ = project->dim_overrides();
			job.run_ = project->run();
			job.outputs_ = project->outputs();
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
	auto configurations = std::make_shared<Configurations>(ArgumentParser(argc, argv));

	Logger::handle().file_mode(configurations->write_file());
	Logger::handle().console_mode(configurations->write_console());
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
