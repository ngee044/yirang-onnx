#include "InferenceEngine.h"

#include <format>
#include <limits>
#include <memory>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace YirangOnnx
{
	namespace
	{
		auto shared_env(void) -> std::shared_ptr<Ort::Env>
		{
			// ORT recommends one Env per process; share it across engines and
			// defer creation until first load() so the constructor cannot throw.
			static std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yirang-onnx");
			return env;
		}

		auto element_size(int32_t elem_type) -> size_t
		{
			switch (elem_type)
			{
			case 1:		  // FLOAT
			case 6:		  // INT32
			case 12:	  // UINT32
				return 4;
			case 2:		  // UINT8
			case 3:		  // INT8
			case 9:		  // BOOL
				return 1;
			case 4:		  // UINT16
			case 5:		  // INT16
			case 10:	  // FLOAT16
			case 16:	  // BFLOAT16
				return 2;
			case 7:		  // INT64
			case 11:	  // DOUBLE
			case 13:	  // UINT64
				return 8;
			default:
				return 0; // unsupported (e.g. STRING)
			}
		}

		auto validate_input(const Tensor& tensor) -> std::expected<void, std::string>
		{
			size_t expected = element_size(tensor.elem_type_);
			if (expected == 0)
			{
				return std::unexpected(std::format("input '{}': unsupported data_type {}", tensor.name_, tensor.elem_type_));
			}
			for (int64_t dim : tensor.shape_)
			{
				if (dim < 0)
				{
					return std::unexpected(std::format("input '{}': negative dimension {}", tensor.name_, dim));
				}
				if (dim != 0 && expected > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))
				{
					return std::unexpected(std::format("input '{}': tensor byte size overflows", tensor.name_));
				}
				expected *= static_cast<size_t>(dim);
			}
			if (tensor.data_.size() != expected)
			{
				return std::unexpected(
					std::format("input '{}': data size {} bytes does not match shape ({} bytes expected)", tensor.name_, tensor.data_.size(), expected));
			}
			return {};
		}

		auto copy_outputs(const std::vector<std::string>& names, const std::vector<Ort::Value>& values)
			-> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>
		{
			std::vector<Tensor> outputs;
			outputs.reserve(values.size());
			for (size_t i = 0; i < values.size(); ++i)
			{
				const auto info = values[i].GetTensorTypeAndShapeInfo();
				const int32_t elem_type = static_cast<int32_t>(info.GetElementType());
				const size_t elem = element_size(elem_type);
				if (elem == 0)
				{
					return { std::nullopt, std::format("output '{}': unsupported data_type {}", names[i], elem_type) };
				}

				Tensor out;
				out.name_ = names[i];
				out.elem_type_ = elem_type;
				out.shape_ = info.GetShape();
				const size_t bytes = info.GetElementCount() * elem;
				if (bytes != 0)
				{
					const uint8_t* data = values[i].GetTensorData<uint8_t>();
					out.data_.assign(data, data + bytes);
				}
				outputs.push_back(std::move(out));
			}
			return { std::move(outputs), std::nullopt };
		}
	} // namespace

	struct InferenceEngine::Backend
	{
		struct BoundInput
		{
			std::string name_;
			int32_t elem_type_;
			std::vector<int64_t> shape_;
			const uint8_t* data_;
			size_t byte_size_;
		};

		struct PreparedRun
		{
			explicit PreparedRun(Ort::Session& session) : binding_(session) {}

			std::vector<BoundInput> inputs_;
			std::vector<std::string> dynamic_outputs_;
			Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
			Ort::IoBinding binding_;
			bool has_outputs_ = false;
		};

		std::shared_ptr<Ort::Env> env_;
		std::optional<Ort::Session> session_;
		std::vector<std::string> output_names_;
		std::unique_ptr<PreparedRun> prepared_;
	};

	InferenceEngine::InferenceEngine(void) : backend_(std::make_unique<Backend>()) {}

	InferenceEngine::~InferenceEngine(void) = default;

	InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;

	auto InferenceEngine::operator=(InferenceEngine&&) noexcept -> InferenceEngine& = default;

	auto InferenceEngine::load(const std::string& model_path, const SessionTuning& tuning) -> std::expected<void, std::string>
	{
		if (backend_ == nullptr)
		{
			backend_ = std::make_unique<Backend>();
		}

		try
		{
			backend_->prepared_.reset();
			backend_->env_ = shared_env();

			Ort::SessionOptions session_options;

			if (tuning.intra_op_threads_.has_value())
			{
				session_options.SetIntraOpNumThreads(tuning.intra_op_threads_.value());
			}
			if (tuning.inter_op_threads_.has_value())
			{
				session_options.SetInterOpNumThreads(tuning.inter_op_threads_.value());
			}

			if (tuning.enable_mem_pattern_)
			{
				session_options.EnableMemPattern();
			}
			else
			{
				session_options.DisableMemPattern();
			}

			if (tuning.enable_cpu_mem_arena_)
			{
				session_options.EnableCpuMemArena();
			}
			else
			{
				session_options.DisableCpuMemArena();
			}

			switch (tuning.execution_mode_)
			{
			case ExecutionMode::sequential:
				session_options.SetExecutionMode(ORT_SEQUENTIAL);
				break;
			case ExecutionMode::parallel:
				session_options.SetExecutionMode(ORT_PARALLEL);
				break;
			}

			switch (tuning.graph_optimization_)
			{
			case GraphOptimization::disabled:
				session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
				break;
			case GraphOptimization::basic:
				session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
				break;
			case GraphOptimization::extended:
				session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
				break;
			case GraphOptimization::all:
				session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
				break;
			}

			backend_->session_.emplace(*backend_->env_, model_path.c_str(), session_options);

			backend_->output_names_.clear();
			Ort::AllocatorWithDefaultOptions allocator;
			const size_t output_count = backend_->session_->GetOutputCount();
			backend_->output_names_.reserve(output_count);
			for (size_t i = 0; i < output_count; ++i)
			{
				backend_->output_names_.push_back(backend_->session_->GetOutputNameAllocated(i, allocator).get());
			}
			return {};
		}
		catch (const Ort::Exception& e)
		{
			backend_->session_.reset();
			return std::unexpected(std::format("cannot load model '{}': {}", model_path, e.what()));
		}
		catch (const std::exception& e)
		{
			backend_->session_.reset();
			return std::unexpected(std::format("cannot load model '{}': {}", model_path, e.what()));
		}
	}

	auto InferenceEngine::loaded(void) const -> bool { return backend_ != nullptr && backend_->session_.has_value(); }

	auto InferenceEngine::run(const std::vector<Tensor>& inputs) const -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>
	{
		if (!loaded())
		{
			return { std::nullopt, "no model loaded (call load() first)" };
		}

		// ORT's ONNXTensorElementDataType enum shares the numbering of ONNX
		// TensorProto.DataType, so elem_type_ casts across directly.
		try
		{
			Ort::Session& session = backend_->session_.value();
			Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			std::vector<Ort::Value> input_values;
			std::vector<const char*> input_names;
			input_values.reserve(inputs.size());
			input_names.reserve(inputs.size());
			for (const auto& tensor : inputs)
			{
				if (auto valid = validate_input(tensor); !valid)
				{
					return { std::nullopt, valid.error() };
				}
				input_values.push_back(Ort::Value::CreateTensor(memory_info, const_cast<uint8_t*>(tensor.data_.data()), tensor.data_.size(), tensor.shape_.data(),
																tensor.shape_.size(), static_cast<ONNXTensorElementDataType>(tensor.elem_type_)));
				input_names.push_back(tensor.name_.c_str());
			}

			const std::vector<std::string>& output_names_str = backend_->output_names_;
			std::vector<const char*> output_names;
			output_names.reserve(output_names_str.size());
			for (const auto& name : output_names_str)
			{
				output_names.push_back(name.c_str());
			}

			auto results
				= session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_values.data(), input_values.size(), output_names.data(), output_names.size());

			return copy_outputs(output_names_str, results);
		}
		catch (const Ort::Exception& e)
		{
			return { std::nullopt, std::format("onnxruntime error: {}", e.what()) };
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("inference failed: {}", e.what()) };
		}
	}

	auto InferenceEngine::prepare(const std::vector<Tensor>& inputs) -> std::expected<void, std::string>
	{
		if (!loaded())
		{
			return std::unexpected("no model loaded (call load() first)");
		}
		backend_->prepared_.reset();
		try
		{
			auto prepared = std::make_unique<Backend::PreparedRun>(backend_->session_.value());
			prepared->inputs_.reserve(inputs.size());
			for (const auto& tensor : inputs)
			{
				if (auto valid = validate_input(tensor); !valid)
				{
					return valid;
				}
				for (const auto& bound : prepared->inputs_)
				{
					if (bound.name_ == tensor.name_)
					{
						return std::unexpected(std::format("duplicate input name '{}'", tensor.name_));
					}
				}
				auto value = Ort::Value::CreateTensor(prepared->memory_info_, const_cast<uint8_t*>(tensor.data_.data()), tensor.data_.size(), tensor.shape_.data(),
													  tensor.shape_.size(), static_cast<ONNXTensorElementDataType>(tensor.elem_type_));
				prepared->binding_.BindInput(tensor.name_.c_str(), value);
				prepared->inputs_.push_back({ tensor.name_, tensor.elem_type_, tensor.shape_, tensor.data_.data(), tensor.data_.size() });
			}

			Ort::AllocatorWithDefaultOptions allocator;
			for (size_t i = 0; i < backend_->output_names_.size(); ++i)
			{
				const auto& name = backend_->output_names_[i];
				const auto type = backend_->session_->GetOutputTypeInfo(i);
				if (type.GetONNXType() != ONNX_TYPE_TENSOR)
				{
					return std::unexpected(std::format("output '{}': only tensor outputs can be prepared", name));
				}
				const auto info = type.GetTensorTypeAndShapeInfo();
				const auto shape = info.GetShape();
				if (element_size(static_cast<int32_t>(info.GetElementType())) == 0)
				{
					return std::unexpected(std::format("output '{}': unsupported data_type {}", name, static_cast<int32_t>(info.GetElementType())));
				}
				bool dynamic = shape.empty();
				for (int64_t dim : shape)
				{
					dynamic = dynamic || dim < 0;
				}
				if (dynamic)
				{
					prepared->dynamic_outputs_.push_back(name);
					prepared->binding_.BindOutput(name.c_str(), prepared->memory_info_);
				}
				else
				{
					auto value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType());
					prepared->binding_.BindOutput(name.c_str(), value);
				}
			}
			backend_->prepared_ = std::move(prepared);
			return {};
		}
		catch (const std::exception& e)
		{
			return std::unexpected(std::format("cannot prepare inference: {}", e.what()));
		}
	}

	auto InferenceEngine::run_prepared(const std::vector<Tensor>& inputs) -> std::expected<void, std::string>
	{
		if (!loaded() || backend_->prepared_ == nullptr)
		{
			return std::unexpected("no prepared inputs (call prepare() first)");
		}
		auto& prepared = *backend_->prepared_;
		prepared.has_outputs_ = false;
		try
		{
			if (inputs.size() != prepared.inputs_.size())
			{
				return std::unexpected("input set changed (call prepare() again)");
			}
			for (size_t i = 0; i < inputs.size(); ++i)
			{
				const auto& input = inputs[i];
				const auto& bound = prepared.inputs_[i];
				if (input.name_ != bound.name_ || input.elem_type_ != bound.elem_type_ || input.shape_ != bound.shape_ || input.data_.data() != bound.data_
					|| input.data_.size() != bound.byte_size_)
				{
					return std::unexpected(std::format("input '{}' changed its binding (call prepare() again)", input.name_));
				}
			}
			for (const auto& name : prepared.dynamic_outputs_)
			{
				prepared.binding_.BindOutput(name.c_str(), prepared.memory_info_);
			}
			backend_->session_->Run(Ort::RunOptions{ nullptr }, prepared.binding_);
			prepared.has_outputs_ = true;
			return {};
		}
		catch (const std::exception& e)
		{
			return std::unexpected(std::format("prepared inference failed: {}", e.what()));
		}
	}

	auto InferenceEngine::collect_outputs(void) const -> std::tuple<std::optional<std::vector<Tensor>>, std::optional<std::string>>
	{
		if (!loaded() || backend_->prepared_ == nullptr || !backend_->prepared_->has_outputs_)
		{
			return { std::nullopt, "no prepared outputs (call run_prepared() successfully first)" };
		}
		try
		{
			return copy_outputs(backend_->output_names_, backend_->prepared_->binding_.GetOutputValues());
		}
		catch (const std::exception& e)
		{
			return { std::nullopt, std::format("cannot collect outputs: {}", e.what()) };
		}
	}
} // namespace YirangOnnx
