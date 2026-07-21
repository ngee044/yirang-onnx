#include "InputProject.h"

#include "Converter.h"
#include "File.h"
#include "InputBuilder.h"
#include "ModelTypes.h"

#include <cstddef>
#include <format>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <utility>

#include <boost/json.hpp>

using namespace Utilities;

namespace YirangOnnx
{
	namespace
	{
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

		auto find_unknown_key(const boost::json::object& object, std::initializer_list<std::string_view> allowed) -> std::optional<std::string>
		{
			for (const auto& entry : object)
			{
				const std::string key(entry.key());
				if (!key.empty() && key.front() == '_')
				{
					continue;
				}
				bool known = false;
				for (const auto& candidate : allowed)
				{
					if (key == candidate)
					{
						known = true;
						break;
					}
				}
				if (!known)
				{
					return key;
				}
			}
			return std::nullopt;
		}
	} // namespace

	auto InputProject::model_path(void) const -> std::string { return model_path_; }
	auto InputProject::inputs(void) const -> std::vector<InputSpec> { return inputs_; }
	auto InputProject::dim_overrides(void) const -> std::map<std::string, int64_t> { return dim_overrides_; }
	auto InputProject::run(void) const -> RunSpec { return run_; }
	auto InputProject::outputs(void) const -> OutputSpec { return outputs_; }
	auto InputProject::inspect(void) const -> std::optional<InspectSpec> { return inspect_; }
	auto InputProject::infer_requested(void) const -> bool { return infer_requested_; }

	auto InputProject::load(const std::string& path) -> std::tuple<std::optional<InputProject>, std::optional<std::string>>
	{
		try
		{
			File file;
			if (auto opened = file.open(path, std::ios::in | std::ios::binary); !opened)
			{
				return { std::nullopt, std::format("cannot open input script '{}': {}", path, opened.error()) };
			}

			auto read = file.read_bytes();
			if (!read)
			{
				return { std::nullopt, std::format("cannot read input script '{}': {}", path, read.error()) };
			}

			return parse(Converter::to_string(read.value()), path);
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("cannot load input script '{}': {}", path, e.what()) };
		}
	}

	auto InputProject::parse(const std::string& json_text, const std::string& origin) -> std::tuple<std::optional<InputProject>, std::optional<std::string>>
	{
		const auto fail = [&origin](const std::string& reason) -> std::tuple<std::optional<InputProject>, std::optional<std::string>>
		{ return { std::nullopt, std::format("input script '{}': {}", origin, reason) }; };

		boost::json::value parsed;
		try
		{
			parsed = boost::json::parse(json_text);
		}
		catch (const std::exception& e)
		{
			return fail(std::format("invalid JSON ({})", e.what()));
		}

		if (!parsed.is_object())
		{
			return fail("root must be a JSON object");
		}
		const auto& root = parsed.as_object();

		if (auto unknown = find_unknown_key(root, { "model", "inspect", "inputs", "dim_overrides", "run", "outputs" }); unknown.has_value())
		{
			return fail(std::format("unknown key '{}'", unknown.value()));
		}

		InputProject project;

		if (const auto* model = root.if_contains("model"))
		{
			if (!model->is_string())
			{
				return fail("'model' must be a string");
			}
			project.model_path_ = model->as_string().c_str();
		}

		if (const auto* inspect = root.if_contains("inspect"))
		{
			if (!inspect->is_object())
			{
				return fail("'inspect' must be an object");
			}
			InspectSpec spec;
			const auto& entry = inspect->as_object();
			if (auto unknown = find_unknown_key(entry, { "format", "out", "weights" }); unknown.has_value())
			{
				return fail(std::format("unknown key 'inspect.{}'", unknown.value()));
			}
			if (const auto* format = entry.if_contains("format"))
			{
				if (!format->is_string())
				{
					return fail("'inspect.format' must be a string");
				}
				spec.format_ = format->as_string().c_str();
				spec.has_format_ = true;
			}
			if (spec.format_ != "summary" && spec.format_ != "json" && spec.format_ != "dot")
			{
				return fail(std::format("'inspect.format' must be summary|json|dot (got '{}')", spec.format_));
			}
			if (const auto* out = entry.if_contains("out"))
			{
				if (!out->is_string())
				{
					return fail("'inspect.out' must be a string");
				}
				spec.out_path_ = out->as_string().c_str();
				spec.has_out_path_ = true;
			}
			if (const auto* weights = entry.if_contains("weights"))
			{
				if (!weights->is_bool())
				{
					return fail("'inspect.weights' must be a boolean");
				}
				spec.include_weights_ = weights->as_bool();
				spec.has_include_weights_ = true;
			}
			project.inspect_ = std::move(spec);
		}

		if (const auto* inputs = root.if_contains("inputs"))
		{
			project.infer_requested_ = true;
			if (!inputs->is_array())
			{
				return fail("'inputs' must be an array");
			}

			size_t index = 0;
			for (const auto& item : inputs->as_array())
			{
				InputSpec spec;
				if (item.is_string())
				{
					spec.path_ = item.as_string().c_str();
					if (spec.path_.empty())
					{
						return fail(std::format("'inputs[{}]' must not be an empty path", index));
					}
				}
				else if (item.is_object())
				{
					const auto& entry = item.as_object();
					if (auto unknown = find_unknown_key(entry, { "name", "path", "random" }); unknown.has_value())
					{
						return fail(std::format("unknown key 'inputs[{}].{}'", index, unknown.value()));
					}
					if (const auto* name = entry.if_contains("name"))
					{
						if (!name->is_string())
						{
							return fail(std::format("'inputs[{}].name' must be a string", index));
						}
						spec.name_ = name->as_string().c_str();
					}
					if (const auto* path = entry.if_contains("path"))
					{
						if (!path->is_string())
						{
							return fail(std::format("'inputs[{}].path' must be a string", index));
						}
						spec.path_ = path->as_string().c_str();
						if (spec.path_.empty())
						{
							return fail(std::format("'inputs[{}].path' must not be empty", index));
						}
					}
					if (const auto* random = entry.if_contains("random"))
					{
						if (!random->is_object())
						{
							return fail(std::format("'inputs[{}].random' must be an object", index));
						}
						RandomInputSpec random_spec;
						std::optional<int32_t> parsed_dtype;
						const auto& random_entry = random->as_object();
						if (auto unknown = find_unknown_key(random_entry, { "data_type", "shape", "seed" }); unknown.has_value())
						{
							return fail(std::format("unknown key 'inputs[{}].random.{}'", index, unknown.value()));
						}
						if (const auto* data_type = random_entry.if_contains("data_type"))
						{
							if (!data_type->is_string())
							{
								return fail(std::format("'inputs[{}].random.data_type' must be a string", index));
							}
							random_spec.data_type_ = data_type->as_string().c_str();

							auto dtype_id = data_type_id(random_spec.data_type_);
							if (!dtype_id.has_value())
							{
								return fail(std::format("'inputs[{}].random.data_type' unknown type '{}'", index, random_spec.data_type_));
							}
							if (!random_generation_supports(dtype_id.value()))
							{
								return fail(std::format("'inputs[{}].random.data_type' '{}' is not supported for random generation (use FLOAT|DOUBLE|INT32|INT64|BOOL)",
														index, random_spec.data_type_));
							}
							parsed_dtype = dtype_id.value();
						}
						if (const auto* shape = random_entry.if_contains("shape"))
						{
							if (!shape->is_array())
							{
								return fail(std::format("'inputs[{}].random.shape' must be an array", index));
							}
							size_t element_count = 1;
							for (const auto& dim : shape->as_array())
							{
								auto value = to_int64(dim);
								if (!value.has_value() || value.value() < 1)
								{
									return fail(std::format("'inputs[{}].random.shape' entries must be positive integers", index));
								}
								const size_t magnitude = static_cast<size_t>(value.value());
								if (element_count > kMaxTensorElements / magnitude)
								{
									return fail(std::format("'inputs[{}].random.shape' element count exceeds limit {}", index, kMaxTensorElements));
								}
								element_count *= magnitude;
								random_spec.shape_.push_back(value.value());
							}
							if (parsed_dtype.has_value())
							{
								if (const size_t width = random_element_byte_width(parsed_dtype.value()); width != 0 && element_count > kMaxTensorBytes / width)
								{
									return fail(std::format("'inputs[{}].random' tensor byte size exceeds limit {} bytes", index, kMaxTensorBytes));
								}
							}
						}
						if (const auto* seed = random_entry.if_contains("seed"))
						{
							auto value = to_int64(*seed);
							if (!value.has_value() || value.value() < 0)
							{
								return fail(std::format("'inputs[{}].random.seed' must be a non-negative integer", index));
							}
							random_spec.seed_ = static_cast<uint64_t>(value.value());
						}
						spec.random_ = std::move(random_spec);
					}

					if (spec.path_.empty() && !spec.random_.has_value())
					{
						if (spec.name_.empty())
						{
							return fail(std::format("'inputs[{}]' needs 'path' or 'random'", index));
						}
						spec.random_ = RandomInputSpec{};
					}
					if (!spec.path_.empty() && spec.random_.has_value())
					{
						return fail(std::format("'inputs[{}]': 'path' and 'random' are exclusive", index));
					}
					if (spec.random_.has_value() && spec.name_.empty())
					{
						return fail(std::format("'inputs[{}]': random input needs 'name' (a graph input name)", index));
					}
				}
				else
				{
					return fail(std::format("'inputs[{}]' must be a string path or an object", index));
				}
				project.inputs_.push_back(std::move(spec));
				++index;
			}
		}

		if (const auto* overrides = root.if_contains("dim_overrides"))
		{
			if (!overrides->is_object())
			{
				return fail("'dim_overrides' must be an object");
			}
			for (const auto& kv : overrides->as_object())
			{
				auto value = to_int64(kv.value());
				if (!value.has_value() || value.value() < 1)
				{
					return fail(std::format("'dim_overrides.{}' must be a positive integer", std::string(kv.key())));
				}
				if (static_cast<size_t>(value.value()) > kMaxTensorElements)
				{
					return fail(std::format("'dim_overrides.{}' exceeds limit {}", std::string(kv.key()), kMaxTensorElements));
				}
				project.dim_overrides_[std::string(kv.key())] = value.value();
			}
		}

		if (const auto* run = root.if_contains("run"))
		{
			project.infer_requested_ = true;
			if (!run->is_object())
			{
				return fail("'run' must be an object");
			}
			const auto& entry = run->as_object();
			if (auto unknown = find_unknown_key(entry, { "repeat", "warmup" }); unknown.has_value())
			{
				return fail(std::format("unknown key 'run.{}'", unknown.value()));
			}
			constexpr int64_t max_iterations = 1'000'000;
			if (const auto* repeat = entry.if_contains("repeat"))
			{
				auto value = to_int64(*repeat);
				if (!value.has_value() || value.value() < 1 || value.value() > max_iterations)
				{
					return fail(std::format("'run.repeat' must be an integer in [1, {}]", max_iterations));
				}
				project.run_.repeat_ = static_cast<uint32_t>(value.value());
			}
			if (const auto* warmup = entry.if_contains("warmup"))
			{
				auto value = to_int64(*warmup);
				if (!value.has_value() || value.value() < 0 || value.value() > max_iterations)
				{
					return fail(std::format("'run.warmup' must be an integer in [0, {}]", max_iterations));
				}
				project.run_.warmup_ = static_cast<uint32_t>(value.value());
			}
		}

		if (const auto* outputs = root.if_contains("outputs"))
		{
			project.infer_requested_ = true;
			if (!outputs->is_object())
			{
				return fail("'outputs' must be an object");
			}
			const auto& entry = outputs->as_object();
			if (auto unknown = find_unknown_key(entry, { "dir", "save", "dump_json", "stats" }); unknown.has_value())
			{
				return fail(std::format("unknown key 'outputs.{}'", unknown.value()));
			}
			if (const auto* dir = entry.if_contains("dir"))
			{
				if (!dir->is_string())
				{
					return fail("'outputs.dir' must be a string");
				}
				project.outputs_.dir_ = dir->as_string().c_str();
			}
			if (const auto* save = entry.if_contains("save"))
			{
				if (!save->is_bool())
				{
					return fail("'outputs.save' must be a boolean");
				}
				project.outputs_.save_ = save->as_bool();
			}
			if (const auto* dump = entry.if_contains("dump_json"))
			{
				if (!dump->is_bool())
				{
					return fail("'outputs.dump_json' must be a boolean");
				}
				project.outputs_.dump_json_ = dump->as_bool();
			}
			if (const auto* stats = entry.if_contains("stats"))
			{
				if (!stats->is_bool())
				{
					return fail("'outputs.stats' must be a boolean");
				}
				project.outputs_.stats_ = stats->as_bool();
			}
		}

		return { std::move(project), std::nullopt };
	}
} // namespace YirangOnnx
