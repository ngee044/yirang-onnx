#include "Configurations.h"
#include "OnnxModel.h"

#include "ArgumentParser.h"
#include "Logger.h"

#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace Utilities;
using namespace YirangOnnx;

namespace
{
	auto print_usage(void) -> void
	{
		std::cerr << "usage: yirang-onnx --model <path.onnx> [--format summary|json|dot] [--out <path>]\n"
					 "                   [--title <name>] [--log_root_path <dir>]\n"
					 "                   [--write_console_log <LogTypes>] [--write_file_log <LogTypes>]\n";
	}

	auto render(const OnnxModel& model, const std::string& format) -> std::string
	{
		if (format == "json")
		{
			return model.to_json();
		}
		if (format == "dot")
		{
			return model.to_dot();
		}
		return model.to_summary();
	}
}

auto main(int argc, char* argv[]) -> int
{
	auto configurations = std::make_shared<Configurations>(ArgumentParser(argc, argv));

	Logger::handle().file_mode(configurations->write_file());
	Logger::handle().console_mode(configurations->write_console());
	Logger::handle().write_interval(configurations->write_interval());
	Logger::handle().log_root(configurations->log_root_path());
	Logger::handle().start(configurations->app_title());

	int exit_code = 0;

	if (auto reason = configurations->invalid_reason(); reason.has_value())
	{
		Logger::handle().write(LogTypes::Error, std::format("invalid configuration: {}", reason.value()));
		print_usage();
		std::cerr << "error: " << reason.value() << '\n';
		exit_code = 2;
	}
	else if (auto [model, error] = OnnxModel::load(configurations->model_path()); !model.has_value())
	{
		Logger::handle().write(LogTypes::Error, error.value_or("unknown error"));
		std::cerr << "error: " << error.value_or("unknown error") << '\n';
		exit_code = 1;
	}
	else
	{
		const std::string rendered = render(model.value(), configurations->output_format());
		const std::string out_path = configurations->output_path();

		if (out_path.empty())
		{
			std::cout << rendered;
		}
		else
		{
			std::ofstream out(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
			if (out.is_open())
			{
				out << rendered;
				Logger::handle().write(LogTypes::Information, std::format("wrote {} to {}", configurations->output_format(), out_path));
			}
			else
			{
				std::cerr << "error: cannot open output file '" << out_path << "'\n";
				exit_code = 1;
			}
		}
	}

	configurations.reset();

	Logger::handle().stop();
	Logger::destroy();

	return exit_code;
}
