#include "Configurations.h"
#include "OnnxModel.h"

#include "ArgumentParser.h"
#include "File.h"
#include "Logger.h"
#include "RunCommand.h"

#include <format>
#include <memory>
#include <string>

using namespace Utilities;
using namespace YirangOnnx;

namespace
{
	auto print_usage(void) -> void
	{
		Logger::handle().write(LogTypes::Error, "usage:\n"
												"  inspect : yirang-onnx --model <path.onnx> [--format summary|json|dot] [--out <path>] [--weights true]\n"
												"  infer   : yirang-onnx --model <path.onnx> --input <in.pb>[,<in2.pb>] [--out-dir <dir>]\n"
												"  common  : [--title <name>] [--log_root_path <dir>] [--write_console_log <LogTypes>] [--write_file_log <LogTypes>]");
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

	if (auto reason = configurations->invalid_reason(); reason.has_value())
	{
		Logger::handle().write(LogTypes::Error, std::format("invalid configuration: {}", reason.value()));
		print_usage();
		exit_code = 2;
	}
	else if (!configurations->input_paths().empty())
	{
		exit_code = run_inference(configurations->model_path(), configurations->input_paths(), configurations->output_dir());
	}
	else if (auto [model, error] = OnnxModel::load(configurations->model_path()); !model.has_value())
	{
		Logger::handle().write(LogTypes::Error, error.value_or("unknown error"));
		exit_code = 1;
	}
	else
	{
		const std::string rendered = render(model.value(), configurations->output_format(), configurations->include_weights());
		const std::string out_path = configurations->output_path();

		if (out_path.empty())
		{
			Logger::handle().write(LogTypes::Information, rendered);
		}
		else
		{
			File out;
			if (auto opened = out.open(out_path, std::ios::out | std::ios::binary | std::ios::trunc); !opened)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot open output file '{}': {}", out_path, opened.error()));
				exit_code = 1;
			}
			else if (auto written = out.write_bytes(reinterpret_cast<const uint8_t*>(rendered.data()), rendered.size()); !written)
			{
				Logger::handle().write(LogTypes::Error, std::format("cannot write output file '{}': {}", out_path, written.error()));
				exit_code = 1;
			}
			else
			{
				Logger::handle().write(LogTypes::Information, std::format("wrote {} to {}", configurations->output_format(), out_path));
			}
		}
	}

	configurations.reset();

	Logger::handle().stop();
	Logger::destroy();

	return exit_code;
}
