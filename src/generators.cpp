// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "generators.h"
#include "models/streaming_processor.h"
#include "models/nemotron_speech.h"
#include "sequences.h"
#include "models/env_utils.h"
#include "models/model.h"
#include "models/model_type.h"
#include "models/decoder_only.h"
#include "models/decoder_only_pipeline.h"
#include "constrained_logits_processor.h"
#include "search.h"
#include "tracing.h"
#include "cpu/interface.h"
#include "cuda/interface.h"
#include "dml/interface.h"
#include "qnn/interface.h"
#include "webgpu/interface.h"
#include "openvino/interface.h"
#include "ryzenai/interface.h"
#include "morphizen_ep/interface.h"
#include "engine/engine.h"

#include <chrono>
#include <cstdio>

#if defined(_WIN32)
EXTERN_C IMAGE_DOS_HEADER __ImageBase;

std::string CurrentModulePath() {
  char path[MAX_PATH];
  GetModuleFileNameA((HINSTANCE)&__ImageBase, path, _countof(path));

  char absolute_path[MAX_PATH];
  char* name;
  GetFullPathNameA(path, _countof(path), absolute_path, &name);

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::string(absolute_path);
  out_path.resize(idx);

  return out_path;
}

#include "dll_load_error.h"
#endif

void ThrowErrorIfSessionTerminated(bool is_session_terminated) {
  if (is_session_terminated)
    throw std::runtime_error("Session in Terminated state, exiting!");
}

namespace Generators {

namespace {
// Monotonic clock used by the Generator-level overhead profiler. Mirrors the
// helper in decoder_only_pipeline.cpp.
using GenProfileClock = std::chrono::steady_clock;

inline uint64_t GenNsBetween(GenProfileClock::time_point a, GenProfileClock::time_point b) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}
}  // namespace

static bool _ = (Ort::InitApi(), false);

static OrtLoggingLevel GetDefaultOrtLoggingLevel() {
  bool ort_verbose_logging = false;
  GetEnv("ORTGENAI_ORT_VERBOSE_LOGGING", ort_verbose_logging);
  return ort_verbose_logging ? OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE : OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
}

OrtGlobals::OrtGlobals()
    : env_{OrtEnv::Create(GetDefaultOrtLoggingLevel())} {
  const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk"};
  const size_t values[] = {static_cast<size_t>(0), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  auto arena_config = OrtArenaCfg::Create(keys, values, 4);
  Ort::Allocator& allocator_cpu{Ort::Allocator::GetWithDefaultOptions()};
  env_->CreateAndRegisterAllocator(allocator_cpu.GetInfo(), *arena_config);

  // Init the CPU device (special case because it always exists, and its allocator is special
  GetDeviceInterface(DeviceType::CPU)->InitOrt(*Ort::api, allocator_cpu);
}

// Ensure Shutdown() has been called before process exit
struct EnsureShutdown {
  ~EnsureShutdown() {
    if (GetOrtGlobals()) {
      Shutdown();
    }
  }
};

std::unique_ptr<OrtGlobals>&
GetOrtGlobals() {
  static auto globals = std::make_unique<OrtGlobals>();
  static auto validate = std::make_unique<EnsureShutdown>();  // Must be after the above line so the destructor runs before the above destructor
  return globals;
}

// Used by Shutdown() to display the counts and types of any leaked objects
template <typename... Types>
bool LeakTypeList<Types...>::Dump() {
  ((LeakChecked<Types>::Count() != 0 ? std::cerr << "OGA Error: " << LeakChecked<Types>::Count() << " instances of " << typeid(Types).name() << " were leaked." << std::endl : std::cerr), ...);
  return ((LeakChecked<Types>::Count() != 0) || ...);
}

void Shutdown() {
  if (LeakTypes::Dump()) {
    std::cerr << "    Please see the documentation for the API being used to ensure proper cleanup." << std::endl;
  }

  GetOrtGlobals().reset();  // Delete now because on process exit is too late

  RyzenAIInterface::Shutdown();
  MorphiZenEPInterface::Shutdown();
}

OrtEnv& GetOrtEnv() {
  return *GetOrtGlobals()->env_;
}

// Fallback to copy between two separate device buffers by going through CPU memory (slow unless we're the CPU device)
void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) {
  source.CopyDeviceToCpu();
  auto source_span = std::span<const uint8_t>(source.p_cpu_ + begin_source, size_in_bytes);
  // If we're overwriting the entire destination
  if (dest.size_in_bytes_ == size_in_bytes)
    dest.AllocateCpu();
  else
    dest.CopyDeviceToCpu();  // Overwriting part of destination, so copy over initial contents first
  std::copy(source_span.begin(), source_span.end(), dest.p_cpu_ + begin_dest);
  dest.CopyCpuToDevice();
}

struct GenaiInterfaceImpl : GenaiInterface {
#if _WIN32
  void* HeapAllocate(size_t size) override { return std::malloc(size); }
  void HeapFree(void* p) override { std::free(p); }
#endif

  void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    return Generators::CopyThroughCpu(dest, begin_dest, source, begin_source, size_in_bytes);
  }

  Generators::LogItems& GetLogItems() override { return g_log; }
  std::ostream& operator_leftshift(std::ostream& stream, Generators::SGR sgr_code) override { return stream << sgr_code; }
  std::ostream& Log(std::string_view label, std::string_view text = {}) override { return Log(label, text); }

  void DumpSpan(std::ostream& stream, std::span<const float> values) override { return Generators::DumpSpan(stream, values); }
  void DumpSpan(std::ostream& stream, std::span<const int> values) override { return Generators::DumpSpan(stream, values); }

  void Sequences_AfterAppendNextTokens(Sequences* p_this, DeviceSpan<int32_t> next_tokens, size_t batch_beam_size) override { return p_this->AfterAppendNextTokens(next_tokens, batch_beam_size); }
  void Sequences_RewindTo(Sequences* p_this, size_t new_length) override { return p_this->RewindTo(new_length); }
} g_genai;

#if defined(_WIN32)
struct LibraryHandle {
  LibraryHandle(const char* filename) {
    auto path = CurrentModulePath() + filename;
    if (!fs::path(path).exists())
      path = filename;
    handle_ = LoadLibrary(path.c_str());
    if (!handle_)
      throw std::runtime_error(std::string("Failed to load library: ") + DetermineLoadLibraryError(filename));
  };

  ~LibraryHandle() { FreeLibrary(handle_); }

  FARPROC __stdcall GetSymbol(const char* name) { return ::GetProcAddress(handle_, name); }

  operator HANDLE() { return handle_; }

 private:
  HMODULE handle_{};
};
#elif defined(__linux__) && !defined(__ANDROID__)
struct LibraryHandle {
  LibraryHandle(const char* filename) {
    auto path = Ort::GetCurrentModuleDir() + "/" + filename;
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error(std::string("Failed to load library: ") + dlerror());  // dlerror() includes the path
  }
  ~LibraryHandle() {
    dlclose(handle_);
  }

  void* GetSymbol(const char* name) { return ::dlsym(handle_, name); }

  operator void*() { return handle_; }

 private:
  void* handle_{};
};
#else
struct LibraryHandle {
  LibraryHandle(const char* filename) {}
  ~LibraryHandle() {}

  void* GetSymbol(const char* name) { return nullptr; }

  operator bool() { return false; }
};
#endif

DeviceInterface* GetCudaInterface(DeviceType type) {
  assert(type == DeviceType::NvTensorRtRtx || type == DeviceType::CUDA);
  try {
#if defined(_WIN32)
    static LibraryHandle library{"onnxruntime-genai-cuda.dll"};
#elif defined(__linux__) && !defined(__ANDROID__)
    static LibraryHandle library{"libonnxruntime-genai-cuda.so"};
#else
    static LibraryHandle library{""};
#endif
    if (!library)
      throw std::runtime_error("Shared library load failure (see first error)");

    Generators::DeviceInterface* GetInterface(GenaiInterface * p_genai, const char* deviceType);
    static DeviceInterface* cuda_interface =
        reinterpret_cast<decltype(&GetInterface)>(
            library.GetSymbol("GetInterface"))(&g_genai, to_string(type).c_str());

    return cuda_interface;
  } catch (const std::exception& e) {
    throw std::runtime_error("Cuda interface not available: " + std::string(e.what()));
  }
}

std::string to_string(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::DML:
      return "DirectML";
    case DeviceType::WEBGPU:
      return "WebGPU";
    case DeviceType::QNN:
      return "QnnWithSharedMemory";
    case DeviceType::OpenVINO:
      return "OpenVINO";
    case DeviceType::NvTensorRtRtx:
      return "NvTensorRtRtx";
    case DeviceType::RyzenAI:
      return "RyzenAI";
    case DeviceType::MorphiZenEP:
      return "MorphiZenEP";
    default:
      throw std::runtime_error("Unknown device type");
  }
}

DeviceInterface* GetDeviceInterface(DeviceType type) {
  switch (type) {
    default:
    case DeviceType::CPU:
      return GetCpuInterface();
    case DeviceType::CUDA:
    case DeviceType::NvTensorRtRtx:
      return GetCudaInterface(type);
#if USE_DML
    case DeviceType::DML:
      return GetDmlInterface();
#endif
    case DeviceType::WEBGPU:
      return GetWebGPUInterface();
    case DeviceType::QNN:
      return GetQNNInterface();
    case DeviceType::OpenVINO:
      return GetOpenVINOInterface();
    case DeviceType::RyzenAI:
      return GetRyzenAIInterface();
    case DeviceType::MorphiZenEP:
      return GetMorphiZenEPInterface();
  }
}

GeneratorParams::GeneratorParams(const Config& config)
    : config{config},
      p_device{GetDeviceInterface(DeviceType::CPU)} {
}

GeneratorParams::GeneratorParams(const Model& model)
    : config{*model.config_.get()},
      use_graph_capture{IsGraphCaptureEnabled(model.config_->model.decoder.session_options)},
      use_multi_profile{IsMultiProfileEnabled(model.config_->model.decoder.session_options)},
      p_device{model.p_device_scoring_} {
  if (use_graph_capture) {
    max_batch_size = 1;  // set it to 1 by default
  }
}

void GeneratorParams::SetGuidance(std::string_view type, std::string_view data, bool enable_ff_tokens = false) {
  guidance_type = type;
  guidance_data = data;
  guidance_ff_tokens_enabled = enable_ff_tokens;
}

bool GeneratorParams::IsPastPresentShareBufferEnabled(const std::string& model_type) const {
  // past_present_share_buffer is only actually enabled when:
  // 1. The config option is set to true, AND
  // 2. Either num_beams == 1 OR the model is Whisper
  return search.past_present_share_buffer &&
         (search.num_beams == 1 || model_type == "whisper");
}

double GeneratorParams::GetSearchNumber(std::string_view name) const {
  if (name == "batch_size") {
    return static_cast<double>(search.batch_size);
  } else if (name == "chunk_size") {
    return static_cast<double>(search.chunk_size.value_or(0));
  } else if (name == "diversity_penalty") {
    return search.diversity_penalty;
  } else if (name == "length_penalty") {
    return search.length_penalty;
  } else if (name == "max_length") {
    return static_cast<double>(search.max_length);
  } else if (name == "min_length") {
    return static_cast<double>(search.min_length);
  } else if (name == "no_repeat_ngram_size") {
    return static_cast<double>(search.no_repeat_ngram_size);
  } else if (name == "num_beams") {
    return static_cast<double>(search.num_beams);
  } else if (name == "num_return_sequences") {
    return static_cast<double>(search.num_return_sequences);
  } else if (name == "random_seed") {
    return static_cast<double>(search.random_seed);
  } else if (name == "repetition_penalty") {
    return search.repetition_penalty;
  } else if (name == "temperature") {
    return search.temperature;
  } else if (name == "top_k") {
    return static_cast<double>(search.top_k);
  } else if (name == "top_p") {
    return search.top_p;
  } else {
    throw std::runtime_error(std::string(name) + " is an invalid name for GetSearchNumber.");
  }
}

bool GeneratorParams::GetSearchBool(std::string_view name) const {
  if (name == "do_sample") {
    return search.do_sample;
  } else if (name == "early_stopping") {
    return search.early_stopping;
  } else if (name == "past_present_share_buffer") {
    return search.past_present_share_buffer;
  } else {
    throw std::runtime_error(std::string(name) + " is an invalid name for GetSearchBool.");
  }
}

std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params) {
  return std::make_unique<Generator>(model, params);
}

std::unique_ptr<Search> CreateSearch(const GeneratorParams& params) {
  if (params.search.num_beams > 1)
    return params.p_device->CreateBeam(params);
  return params.p_device->CreateGreedy(params);
}

Generator::Generator(const Model& model, const GeneratorParams& params) : model_{model.shared_from_this()} {
  // Same gate as the State-level profiler so a single env var enables both
  // and the two reports compose into a full top-to-bottom view.
  GetEnv("ORTGENAI_PIPELINE_OVERHEAD_PROFILE", overhead_profile_enabled_);

  // RNNT models don't use the traditional search/logits pipeline,
  // so skip the standard validations and just create the state.
  if (ModelType::IsRNNT(model.config_->model.type)) {
    state_ = model.CreateState({}, params);
    return;
  }

  if (params.search.max_length == 0)
    throw std::runtime_error("search max_length is 0");
  if (params.search.max_length > model.config_->model.context_length)
    throw std::runtime_error("max_length (" + std::to_string(params.search.max_length) + ") cannot be greater than model context_length (" + std::to_string(model.config_->model.context_length) + ")");
  if (params.search.batch_size < 1)
    throw std::runtime_error("batch_size must be 1 or greater, is " + std::to_string(params.search.batch_size));
  if (params.config.model.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.config.model.vocab_size));

  search_ = CreateSearch(params);
  state_ = model.CreateState(search_->GetSequenceLengths(), params);    // Search sequence lengths set when creating state
  guidance_logits_processor_ = CreateGuidanceLogitsProcessor(*state_);  // Could be nullptr if use_guidance (constrained decoding) is not used
}

Generator::~Generator() {
  if (!overhead_profile_enabled_) {
    return;
  }
  if (append_tokens_stats_.calls == 0 &&
      generate_next_token_with_logits_stats_.calls == 0 &&
      generate_next_token_sample_only_stats_.calls == 0) {
    return;
  }

  // ---- Helpers -------------------------------------------------------------
  // Layout: 2-space outer indent + per-row indent + label (left-justified to
  // value column) + 14-char right-aligned value + " us  (" + 6-char % + " %)".
  // Total line width = kValueCol + 30 chars, regardless of indent.
  constexpr int kValueCol = 56;

  auto fmt_us = [](double us, int width) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f", us < 0.0 ? 0.0 : us);
    std::string s = buf;
    size_t dot = s.find('.');
    if (dot == std::string::npos) dot = s.size();
    for (int i = static_cast<int>(dot) - 3; i > 0; i -= 3) {
      s.insert(static_cast<size_t>(i), ",");
    }
    if (static_cast<int>(s.size()) < width) {
      s.insert(0, static_cast<size_t>(width) - s.size(), ' ');
    }
    return s;
  };
  auto fmt_pct = [](double num, double den) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%6.2f", den > 0.0 ? (num / den) * 100.0 : 0.0);
    return std::string(buf);
  };

  // Row with value + percent (used for breakdown children and totals).
  auto row = [&](int indent, const char* label, double us, double total_us) {
    const int label_w = std::max(1, kValueCol - indent);
    std::fprintf(stderr, "%*s%-*s %s us  (%s %%)\n",
                 indent, "",
                 label_w, label,
                 fmt_us(us, 14).c_str(),
                 fmt_pct(us, total_us).c_str());
  };
  // Row without percent (per-stage averages: setup / inner / teardown).
  auto row_no_pct = [&](int indent, const char* label, double us) {
    const int label_w = std::max(1, kValueCol - indent);
    std::fprintf(stderr, "%*s%-*s %s us\n",
                 indent, "",
                 label_w, label,
                 fmt_us(us, 14).c_str());
  };
  // Divider line spanning the same width as a data row at this indent.
  auto divider = [&](int indent) {
    const int len = std::max(8, (kValueCol + 30) - indent);
    const std::string dashes(static_cast<size_t>(len), '-');
    std::fprintf(stderr, "%*s%s\n", indent, "", dashes.c_str());
  };
  // Phase header: leading blank lines, thick rule, title, call-label,
  // then a thin rule before the body. The thick rules above and below each
  // phase make the boundaries between PREFILL / DECODE / FIRST-TOKEN
  // SAMPLING obvious at a glance.
  const std::string thick_rule(80, '=');
  const std::string thin_rule(80, '-');
  auto section_header = [&](const char* title, const char* call_label) {
    std::fprintf(stderr, "\n\n%s\n", thick_rule.c_str());
    std::fprintf(stderr, "  %s\n", title);
    if (call_label && call_label[0]) {
      std::fprintf(stderr, "  %s\n", call_label);
    }
    std::fprintf(stderr, "%s\n", thin_rule.c_str());
  };

  // ---- Pull State stats if available, then suppress its own destructor print
  DecoderOnlyPipelineState* pipe_state = dynamic_cast<DecoderOnlyPipelineState*>(state_.get());
  DecoderOnlyPipelineState::OverheadStats empty_state_stats{};
  const auto& state_prefill = pipe_state ? pipe_state->GetPrefillOverheadStats() : empty_state_stats;
  const auto& state_decode = pipe_state ? pipe_state->GetDecodeOverheadStats() : empty_state_stats;
  if (pipe_state) {
    pipe_state->SuppressOverheadDestructorReport();
  }

  // ---- Per-phase printer ---------------------------------------------------
  auto print_phase = [&](const char* phase_title,
                         const char* call_label,
                         const char* benchmark_ref,
                         const char* other_row_label,
                         const GenStats& gen_stats,
                         const DecoderOnlyPipelineState::OverheadStats* state_stats) {
    if (gen_stats.calls == 0) return;

    const double calls = static_cast<double>(gen_stats.calls);
    const double gen_total_us = (gen_stats.total_ns / 1000.0) / calls;
    const double gen_sampling_us = (gen_stats.sampling_ns / 1000.0) / calls;

    const bool have_state =
        state_stats != nullptr && state_stats->runs > 0 && state_stats->stages > 0;
    const double state_runs = have_state ? static_cast<double>(state_stats->runs) : 0.0;
    const double state_stages = have_state ? static_cast<double>(state_stats->stages) : 0.0;
    const double state_steps = have_state ? static_cast<double>(state_stats->steps) : 0.0;
    const double state_full_us =
        have_state
            ? (state_stats->setup_ns + state_stats->inner_run_ns +
               state_stats->teardown_ns + state_stats->outer_ns) / 1000.0 / state_runs
            : 0.0;
    const double state_inner_us =
        have_state ? (state_stats->inner_run_ns / 1000.0) / state_runs : 0.0;
    const double state_orch_us =
        have_state ? ((state_stats->setup_ns + state_stats->teardown_ns) / 1000.0) / state_runs : 0.0;
    const double state_outer_us =
        have_state ? (state_stats->outer_ns / 1000.0) / state_runs : 0.0;
    const double chunks_per_run = (have_state && state_runs > 0.0) ? state_steps / state_runs : 0.0;
    const double stages_per_run = (have_state && state_runs > 0.0) ? state_stages / state_runs : 0.0;
    const double stages_per_chunk =
        (have_state && state_steps > 0.0) ? state_stages / state_steps : 0.0;

    const double per_stage_setup_us =
        have_state ? (state_stats->setup_ns / 1000.0) / state_stages : 0.0;
    const double per_stage_inner_us =
        have_state ? (state_stats->inner_run_ns / 1000.0) / state_stages : 0.0;
    const double per_stage_teardown_us =
        have_state ? (state_stats->teardown_ns / 1000.0) / state_stages : 0.0;

    // Generator scaffolding = everything Generator does outside state_->Run.
    // Reference State.full_run rather than Generator.compute_logits so the
    // tiny ComputeLogits epilogue (SetLogits + guidance) lands in "other" too.
    const double gen_scaffolding_us =
        have_state ? std::max(0.0, gen_total_us - state_full_us) : 0.0;
    const double gen_other_us = std::max(0.0, gen_scaffolding_us - gen_sampling_us);

    // OGA-side overhead = everything except the actual session.Run kernel time.
    const double oga_overhead_us =
        have_state ? std::max(0.0, gen_total_us - state_inner_us) : 0.0;

    // ---- Print
    section_header(phase_title, call_label);
    std::fprintf(stderr, "  Calls       : %llu\n",
                 static_cast<unsigned long long>(gen_stats.calls));
    if (have_state) {
      std::fprintf(stderr,
                   "  Topology    : %.2f chunks/Run   %.2f stages/Run   %.2f stages/chunk\n",
                   chunks_per_run, stages_per_run, stages_per_chunk);
    }
    std::fprintf(stderr,
                 "  Wall clock  : %s us  (%.2f ms)\n"
                 "                benchmark \"%s\"\n",
                 fmt_us(gen_total_us, 0).c_str(),
                 gen_total_us / 1000.0,
                 benchmark_ref);

    if (have_state) {
      // Per-call inner / orchestration rows decompose into per-stage averages
      // multiplied by stages/Run. Both equalities are surfaced inline below
      // so the reader can follow the data flow without cross-referencing.
      char inner_eq[80];
      char orch_eq[80];
      std::snprintf(inner_eq, sizeof(inner_eq),
                    "= inner %s * %.2f stages/Run",
                    fmt_us(per_stage_inner_us, 0).c_str(), stages_per_run);
      std::snprintf(orch_eq, sizeof(orch_eq),
                    "= (setup %s + teardown %s) * %.2f stages/Run",
                    fmt_us(per_stage_setup_us, 0).c_str(),
                    fmt_us(per_stage_teardown_us, 0).c_str(),
                    stages_per_run);

      std::fprintf(stderr, "\n  Decomposition (children sum to wall clock):\n");
      row(4, "inner session.Run", state_inner_us, gen_total_us);
      std::fprintf(stderr, "        %s\n", inner_eq);
      row(4, "Generator scaffolding (outside State::Run)", gen_scaffolding_us, gen_total_us);
      row(6, "sampling (SelectTop / Sample*)", gen_sampling_us, gen_total_us);
      row(6, other_row_label, gen_other_us, gen_total_us);
      row(4, "in-RunPipeline orchestration", state_orch_us, gen_total_us);
      std::fprintf(stderr, "        %s\n", orch_eq);
      row(4, "outside RunPipeline (UpdateIO + slide + cleanup)", state_outer_us, gen_total_us);
      divider(4);
      row(4, "= Wall clock per call", gen_total_us, gen_total_us);
      row(4, "    of which CPU outside inner session.Run", oga_overhead_us, gen_total_us);

      // Per-chunk breakdown -- only printed when the chunk loop runs multiple
      // times (sliding-window prefill). For decode (1 chunk/Run) the per-chunk
      // number equals the per-call wall clock; the per-stage averages already
      // appear inline in the Decomposition above, so we skip this subsection.
      if (chunks_per_run > 1.0 && state_stats->steps > 0) {
        const double per_chunk_total_us =
            (state_stats->chunk_loop_body_ns / 1000.0) / state_steps;
        const double per_chunk_runpipeline_us =
            ((state_stats->setup_ns + state_stats->inner_run_ns +
              state_stats->teardown_ns) / 1000.0) / state_steps;
        // Slide block runs (steps - runs) times per phase: once per chunk
        // EXCEPT the last chunk of each Run. To keep the per-chunk
        // decomposition additive (children sum to wall clock), the row value
        // is amortized over ALL chunks rather than over slide invocations.
        // The per-invocation cost is still surfaced inline beneath.
        const double slide_total_us = std::max(
            0.0,
            (per_chunk_total_us - per_chunk_runpipeline_us) * state_steps);
        const double per_chunk_slide_us = slide_total_us / state_steps;
        const uint64_t total_slides =
            state_stats->steps > state_stats->runs ? state_stats->steps - state_stats->runs : 0;
        const double per_slide_invocation_us =
            total_slides > 0 ? slide_total_us / static_cast<double>(total_slides) : 0.0;
        const unsigned long long slides_per_run = static_cast<unsigned long long>(
            total_slides / std::max<uint64_t>(1, state_stats->runs));

        std::fprintf(stderr,
                     "\n  Per-chunk wall clock (%.0f chunks/Run, %llu Run total):\n",
                     chunks_per_run,
                     static_cast<unsigned long long>(state_stats->runs));
        row(4, "RunPipeline (1/chunk)", per_chunk_runpipeline_us, per_chunk_total_us);
        std::fprintf(stderr,
                     "        = (setup %s + inner %s + teardown %s) * %.2f stages/chunk\n",
                     fmt_us(per_stage_setup_us, 0).c_str(),
                     fmt_us(per_stage_inner_us, 0).c_str(),
                     fmt_us(per_stage_teardown_us, 0).c_str(),
                     stages_per_chunk);
        row(4, "between-chunk slide (KV / positions / mask)",
            per_chunk_slide_us, per_chunk_total_us);
        std::fprintf(stderr,
                     "        = %s us/invocation * %llu invocations / %.0f chunks\n",
                     fmt_us(per_slide_invocation_us, 0).c_str(),
                     slides_per_run,
                     chunks_per_run);
        divider(4);
        row(4, "= per-chunk wall clock", per_chunk_total_us, per_chunk_total_us);
      }
    } else {
      // No State data: degraded view from Generator stats only.
      const double gen_cl_us = (gen_stats.compute_logits_ns / 1000.0) / calls;
      const double gen_other_only_us = std::max(0.0, gen_total_us - gen_cl_us - gen_sampling_us);
      std::fprintf(stderr, "\n  Decomposition (children sum to wall clock):\n");
      row(4, "ComputeLogits (state_->Run, lumped)", gen_cl_us, gen_total_us);
      row(4, "sampling (SelectTop / Sample*)", gen_sampling_us, gen_total_us);
      row(4, "other (search ops, validation, guidance)", gen_other_only_us, gen_total_us);
      divider(4);
      row(4, "= Wall clock per call", gen_total_us, gen_total_us);
      std::fprintf(stderr, "  (detailed State::Run breakdown unavailable for this State type)\n");
    }
  };

  std::fprintf(stderr,
               "\n[OGA profile] Per-call CPU breakdown"
               "   (ORTGENAI_PIPELINE_OVERHEAD_PROFILE=1)\n");

  print_phase("PREFILL  (prompt processing)",
              "AppendTokens(), one call per prompt batch",
              "Prompt processing (time to first token): avg (us)",
              "other (alloc + search.Append + ComputeLogits epilogue)",
              append_tokens_stats_,
              pipe_state ? &state_prefill : nullptr);

  print_phase("DECODE  (token generation)",
              "GenerateNextToken(), once per generated token",
              "Token generation: avg (us)",
              "other (per-token: MinLen + RepPen + guidance + epilogue)",
              generate_next_token_with_logits_stats_,
              pipe_state ? &state_decode : nullptr);

  // Sampling-only is special: this is the very first GenerateNextToken call
  // after AppendTokens. Logits were already computed during prefill, so this
  // call only samples -- no ComputeLogits / state_->Run happens here, hence
  // there is never State data to attach.
  if (generate_next_token_sample_only_stats_.calls > 0) {
    const double calls = static_cast<double>(generate_next_token_sample_only_stats_.calls);
    const double total_us = (generate_next_token_sample_only_stats_.total_ns / 1000.0) / calls;
    const double samp_us = (generate_next_token_sample_only_stats_.sampling_ns / 1000.0) / calls;
    const double other_us = std::max(0.0, total_us - samp_us);
    section_header("FIRST-TOKEN SAMPLING (no ComputeLogits)",
                   "GenerateNextToken(), 1st call (logits already computed)");
    std::fprintf(stderr, "  Calls       : %llu\n",
                 static_cast<unsigned long long>(generate_next_token_sample_only_stats_.calls));
    std::fprintf(stderr,
                 "  Wall clock  : %s us\n"
                 "                benchmark \"Token sampling: avg (us)\"\n",
                 fmt_us(total_us, 0).c_str());
    std::fprintf(stderr, "\n  Decomposition (children sum to wall clock):\n");
    row(4, "sampling (SelectTop / Sample*)", samp_us, total_us);
    row(4, "other", other_us, total_us);
    divider(4);
    row(4, "= Wall clock per call", total_us, total_us);
  }

  // Single, compact Notes block. Anything previously inlined per-row now
  // lives here so the report stays scannable and the explanation appears
  // exactly once.
  std::fprintf(stderr,
               "\n"
               "Notes\n"
               "--------------------------------------------------------------------------------\n"
               "  Layout       Indented rows are children of the row above. Totals are\n"
               "               marked '=' and preceded by a divider line. \"of which\"\n"
               "               rows re-slice the total above (not added to it).\n"
               "\n"
               "  Required vs avoidable (rows in the per-call decomposition):\n"
               "    REQUIRED   inner session.Run, sampling, other, outside RunPipeline\n"
               "               (cannot be removed without breaking output)\n"
               "    AVOIDABLE  in-RunPipeline orchestration\n"
               "               (pipeline-mode-only: per-stage rebind, ortvalue_store_\n"
               "               lookups, HasInput/HasOutput scans, cross-stage forwarding)\n"
               "\n"
               "  \"other\" content per phase (wall clock minus state_->Run minus sampling):\n"
               "    PREFILL    AllocateInputIdsOnDevice, search_->AppendTokens,\n"
               "               ComputeLogits epilogue (SetLogits, optional\n"
               "               guidance.CommitTokens / GetFFTokens fast-forward).\n"
               "    DECODE     ComputeLogits epilogue (SetLogits, optional\n"
               "               guidance fast-forward), optional guidance.ProcessLogits,\n"
               "               ApplyMinLength, ApplyRepetitionPenalty, validation.\n"
               "               These per-token search ops are why DECODE \"other\" is\n"
               "               larger per call than PREFILL \"other\".\n"
               "\n"
               "  Per-stage <-> per-call <-> per-chunk:\n"
               "    inner session.Run        = inner    * stages/Run     [per call]\n"
               "    in-RunPipeline orch.     = (setup + teardown) * stages/Run\n"
               "    per-chunk RunPipeline    = (setup + inner + teardown) * stages/chunk\n"
               "    The inline equations on each row above use these identities to tie\n"
               "    per-call totals back to the per-stage averages.\n"
               "\n"
               "  Caveats:\n"
               "    * chunks/Run and stages/Run are per State::Run() invocation. They equal\n"
               "      per-Generator-call values when each AppendTokens / GenerateNextToken\n"
               "      triggers exactly one State::Run (the rare exception is the guidance\n"
               "      fast-forward path that fires a 2nd state_->Run inside one\n"
               "      AppendTokens call).\n"
               "    * PREFILL Wall clock measures Generator::AppendTokens. The benchmark\n"
               "      wraps Generator::AppendTokenSequences which additionally calls\n"
               "      Generators::PadInputs, so this number can be a few microseconds below\n"
               "      the benchmark Prompt processing total.\n"
               "================================================================================\n");
  std::fflush(stderr);
}

DeviceSpan<int32_t> Generator::AllocateInputIdsOnDevice(cpu_span<const int32_t> input_ids) {
  size_t padded_input_ids_size = input_ids.size();
  if (model_->config_->model.decoder.sliding_window.has_value()) {
    // If the model has a sliding window, pad the input_ids to the next multiple of the window size
    // so that the input_ids can be divided into window size chunks.
    const auto window_size = model_->config_->model.decoder.sliding_window->window_size;

    if (model_->config_->model.decoder.sliding_window->slide_inputs) {
      padded_input_ids_size = ((input_ids.size() + window_size - 1) / window_size) * window_size;
    }
  }

  auto input_ids_device = state_->params_->p_device->Allocate<int32_t>(padded_input_ids_size);
  auto cpu_span = input_ids_device.CpuSpan();
  auto padding_begin = cpu_span.begin();
  auto data_end = cpu_span.end();
  if (model_->config_->model.decoder.sliding_window.has_value() && model_->config_->model.decoder.sliding_window->alignment == "left") {
    padding_begin = cpu_span.begin() + input_ids.size();
    data_end = padding_begin;
  }
  std::fill_n(padding_begin, padded_input_ids_size - input_ids.size(), model_->config_->model.pad_token_id);
  std::copy_backward(input_ids.begin(), input_ids.end(), data_end);
  input_ids_device.CopyCpuToDevice();
  return input_ids_device;
}

void Generator::AppendTokens(cpu_span<const int32_t> input_ids) {
  DurationTrace trace{"Generator::AppendTokens"};

  // Bracket the whole AppendTokens call; the inner ComputeLogits call below
  // is bracketed separately so the destructor report can show
  // total = compute_logits + other.
  const auto t_call_start = overhead_profile_enabled_ ? GenProfileClock::now() : GenProfileClock::time_point{};
  uint64_t call_compute_logits_ns = 0;

  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (input_ids.size() == 0)
    throw std::runtime_error("input_ids is empty");
  if ((input_ids.size() / state_->params_->search.batch_size) + search_->GetSequenceLength() > state_->params_->search.max_length)
    throw std::runtime_error("input_ids size (" + std::to_string(input_ids.size()) + ") + current sequence length (" + std::to_string(search_->GetSequenceLength()) + ") exceeds max length (" + std::to_string(state_->params_->search.max_length) + ")");
  if (search_->GetSequenceLength() != 0 && state_->params_->search.batch_size > 1)
    throw std::runtime_error("AppendTokens can only be called once for batch_size > 1. To call AppendTokens again, use RewindToLength(0)");

  // Some models fallback to CPU for the attention operator (for example, some decoder-pipeline NPU models).
  // Continuous decoding is supported for this case as the kv cache for such models is always on CPU.
  constexpr std::array<DeviceType, 6> devices_supporting_continuous_decoding{
      DeviceType::CPU,
      DeviceType::CUDA,
      DeviceType::WEBGPU,
      DeviceType::OpenVINO,
      DeviceType::NvTensorRtRtx,
      DeviceType::RyzenAI};

  if (search_->GetSequenceLength() != 0 &&
      std::none_of(devices_supporting_continuous_decoding.begin(), devices_supporting_continuous_decoding.end(),
                   [this](DeviceType device_type) { return device_type == state_->model_.p_device_kvcache_->GetType(); }))
    // Support for continuous decoding should be based on the type of device used for KV cache
    throw std::runtime_error("Continuous decoding is not supported on the selected device type (" + to_string(state_->model_.p_device_kvcache_->GetType()) +
                             "). Please recreate the generator instance to avoid using continuous decoding.");

  // [NO_CHUNK_EXPERIMENTAL] The legacy fixed_prompt_length path was never
  // designed for continuous decoding (chat mode's multiple AppendTokens
  // calls): the single-Run static-shape flow has no notion of a running
  // KV history across separate prompt batches. Reject explicitly rather
  // than silently corrupting outputs. Use the sliding_window path with
  // alignment="left" for chat-style scenarios.
  if (search_->GetSequenceLength() != 0 &&
      model_->config_->model.decoder.fixed_prompt_length > 0) {
    throw std::runtime_error(
        "Continuous decoding (multiple AppendTokens calls) is not supported "
        "on the experimental fixed_prompt_length path. Use sliding_window "
        "with alignment=\"left\" instead, or call RewindToLength(0) before "
        "the next AppendTokens call.");
  }

  // Set any extra inputs (those defined in extra_inputs and those defined in the PresetExtraInputs registry)
  if (set_extra_inputs_) {
    state_->SetExtraInputs(extra_inputs_);
    set_extra_inputs_ = false;
  }

  auto input_ids_device = AllocateInputIdsOnDevice(input_ids);

  // For sliding-window models with slide_inputs, AllocateInputIdsOnDevice pads
  // input_ids up to a multiple of window_size with pad_token_id. The pad tokens
  // are needed by the model (so each chunk has window_size entries), but they
  // must NOT be appended to the search sequence -- otherwise the user-visible
  // generated text contains spurious pad/EOS tokens between the prompt and the
  // model's first real generated token (the pad count = window_size - real
  // prompt length, e.g. 107 for a 21-token prompt with window_size=128).
  const bool slide_inputs_padded =
      model_->config_->model.decoder.sliding_window.has_value() &&
      model_->config_->model.decoder.sliding_window->slide_inputs &&
      input_ids_device.size() != input_ids.size();

  if (slide_inputs_padded) {
    auto unpadded_for_search = state_->params_->p_device->Allocate<int32_t>(input_ids.size());
    std::copy(input_ids.begin(), input_ids.end(), unpadded_for_search.CpuSpan().begin());
    unpadded_for_search.CopyCpuToDevice();
    search_->AppendTokens(unpadded_for_search);
  } else {
    search_->AppendTokens(input_ids_device);
  }

  computed_logits_ = false;
  {
    const auto t_cl_pre = overhead_profile_enabled_ ? GenProfileClock::now() : GenProfileClock::time_point{};
    ComputeLogits(input_ids_device);
    if (overhead_profile_enabled_) {
      call_compute_logits_ns = GenNsBetween(t_cl_pre, GenProfileClock::now());
    }
  }

  if (overhead_profile_enabled_) {
    const uint64_t total = GenNsBetween(t_call_start, GenProfileClock::now());
    ++append_tokens_stats_.calls;
    append_tokens_stats_.total_ns += total;
    append_tokens_stats_.compute_logits_ns += call_compute_logits_ns;
    // sampling_ns intentionally not bumped: AppendTokens never samples.
  }
}

void Generator::SetInputs(const NamedTensors& named_tensors) {
  if (ModelType::IsLLM(model_->config_->model.type) || ModelType::IsPipe(model_->config_->model.type)) {
    throw std::runtime_error("Please use generator.AppendTokens for " + model_->config_->model.type + ". SetInputs is not supported for this model type.");
  }

  cpu_span<int32_t> input_ids;
  for (const auto& [name, tensor] : named_tensors) {
    if (name == Config::Defaults::InputIdsName) {
      input_ids = cpu_span<int32_t>(tensor->ort_tensor_->GetTensorMutableData<int32_t>(),
                                    tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
    } else {
      // If the nominal name is found in the map, use the graph name.
      // Else, use the nominal name as the graph name.
      [[maybe_unused]] const auto [graph_name, found] = model_->config_->GetGraphName(name);
      extra_inputs_.push_back({graph_name, tensor});
    }
  }

  // Set any extra inputs (those defined in extra_inputs and those defined in the PresetExtraInputs registry)
  if (set_extra_inputs_) {
    state_->SetExtraInputs(extra_inputs_);
    set_extra_inputs_ = false;
  }

  // Append tokens and run ComputeLogits after setting all other possible inputs
  if (input_ids.size() > 0) {
    AppendTokens(input_ids);
  }
}

void Generator::ComputeLogits(DeviceSpan<int32_t> next_tokens) {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling AppendTokens or GenerateNextToken first");

  // search_->GetSequenceLength() != next_tokens.size() implies that this is not the first time ComputeLogits
  // is being called (i.e. we're not computing logits for the initial input tokens), so we need to commit
  // tokens to the guidance logits processor before running the model.
  if (guidance_logits_processor_ && search_->GetSequenceLength() != next_tokens.size()) {
    auto next_tokens_span = next_tokens.CopyDeviceToCpu();
    guidance_logits_processor_->CommitTokens(next_tokens_span);
  }

  auto logits = state_->Run(search_->GetSequenceLength(), next_tokens, search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpValues(stream, Ort::TypeToTensorType<float>, logits.CopyDeviceToCpu().data(), logits.size());
    stream << std::endl;
  }
  SetLogits(logits);

  if (guidance_logits_processor_ && search_->GetSequenceLength() != next_tokens.size()) {
    auto ff_tokens = guidance_logits_processor_->GetFFTokens(0);
    if (!ff_tokens.empty()) {
      // process fast-forward tokens
      std::span<int32_t> forced_tokens_span{ff_tokens};
      auto forced_tokens = AllocateInputIdsOnDevice(forced_tokens_span);
      search_->AppendTokens(forced_tokens);

      std::span<int32_t> new_next_token_span{ff_tokens};
      auto new_next_token = AllocateInputIdsOnDevice(new_next_token_span);
      logits = state_->Run(search_->GetSequenceLength(), new_next_token, search_->GetNextIndices());
      if (g_log.enabled && g_log.model_logits) {
        auto& stream_ = Log("model_logits");
        DumpValues(stream_, Ort::TypeToTensorType<float>, logits.CopyDeviceToCpu().data(), logits.size());
        stream_ << std::endl;
      }
      SetLogits(logits);
    }
  }

  last_action_ = Action::standard;
  computed_logits_ = true;
}

void Generator::SetRuntimeOption(const char* key, const char* value) {
  state_->SetRunOption(key, value);
}

size_t Generator::TokenCount() const {
  if (auto* speech_state = dynamic_cast<NemotronSpeechState*>(state_.get()))
    return speech_state->TokenCount();
  return static_cast<size_t>(search_->GetSequenceLength());
}

bool Generator::IsDone() {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);

  if (auto* speech_state = dynamic_cast<NemotronSpeechState*>(state_.get())) {
    // Pending mel input means we haven't started processing this chunk yet
    if (!extra_inputs_.empty()) return false;
    return speech_state->IsChunkDone();
  }

  if (computed_logits_) {
    return false;
  }

  bool is_done = search_->IsDone();
  if (is_done) {
    state_->Finalize(search_->GetSequenceLength());
    if (guidance_logits_processor_) {
      guidance_logits_processor_->ResetWithoutCompute();
      last_action_ = Action::standard;
    }
  }

  return is_done;
}

bool Generator::IsSessionTerminated() const {
  return state_->session_terminated_;
}

void Generator::SetLogits(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  computed_logits_ = true;
}

void Generator::GenerateNextToken() {
  DurationTrace trace{"Generator::GenerateNextToken"};

  // Profile bracketing. We need an RAII commit because the function has
  // early returns (RNNT; SelectTop fast path used to return). The bucket
  // (with-logits vs sampling-only) is decided by the value of
  // computed_logits_ on ENTRY -- false means this call also runs
  // ComputeLogits/state_->Run, true means this call only samples.
  const bool profile = overhead_profile_enabled_;
  const auto t_call_start = profile ? GenProfileClock::now() : GenProfileClock::time_point{};
  uint64_t call_compute_logits_ns = 0;
  uint64_t call_sampling_ns = 0;
  const bool had_logits_on_entry = computed_logits_;

  struct StatsCommitter {
    bool enabled;
    GenStats* with_logits;
    GenStats* sample_only;
    bool had_logits_on_entry;
    const uint64_t* compute_logits_ns;
    const uint64_t* sampling_ns;
    GenProfileClock::time_point start;
    bool cancelled{false};
    ~StatsCommitter() {
      if (!enabled || cancelled) return;
      GenStats& target = had_logits_on_entry ? *sample_only : *with_logits;
      ++target.calls;
      target.total_ns += GenNsBetween(start, GenProfileClock::now());
      target.compute_logits_ns += *compute_logits_ns;
      target.sampling_ns += *sampling_ns;
    }
  };
  StatsCommitter committer{profile,
                           &generate_next_token_with_logits_stats_,
                           &generate_next_token_sample_only_stats_,
                           had_logits_on_entry,
                           &call_compute_logits_ns,
                           &call_sampling_ns,
                           t_call_start,
                           false};

  ThrowErrorIfSessionTerminated(state_->session_terminated_);

  // RNNT models: yield one token per call from the decoder state machine
  if (auto* speech_state = dynamic_cast<NemotronSpeechState*>(state_.get())) {
    committer.cancelled = true;  // RNNT path is unrelated to TTFT/decode bucketing.
    state_->SetExtraInputs(extra_inputs_);
    extra_inputs_.clear();
    speech_state->StepToken();
    return;
  }

  if (search_->GetSequenceLength() == 0 && !computed_logits_)
    throw std::runtime_error("GenerateNextToken called with no prior state. Please call AppendTokens, SetLogits, or SetInputs before calling GenerateNextToken.");

  // TRT-RTX and DML EPs use a single rope factor for all tokens: https://github.com/microsoft/onnxruntime-genai/blob/d5dc8cb02fd02b0dce99c6938449566371da0d28/src/python/py/models/builder.py#L1464-L1473
  // TODO: change this when these EPs support multi rope factors
  const bool epUsesSingleRopeFactor = model_->p_device_->GetType() == DeviceType::NvTensorRtRtx || model_->p_device_->GetType() == DeviceType::DML;

  // TODO: Extend the solution to make it work for batch size > 1, num beams > 1, multimodal and DML
  // Phi3 model switches from short factor to long factor at 4097 (original_max_position_embeddings+1) token, needs Recomputation of Position IDs and KV Cache
  // at this stage which is achieved by rewinding to zero and appending the current sequence
  // Scenarios where this solution works: Batch size = 1, Num beams = 1, decoder model, EP is either CPU or CUDA
  // Scenarios where it doesn't work: Batch size > 1 OR Num beams > 1 OR Multimodal model (like phi3 vision) OR EP is DML
  if (search_->params_->BatchBeamSize() == 1 && !epUsesSingleRopeFactor) {
    if (((search_->GetSequenceLength() == 4097) && (model_->config_->model.type == "phi3" || model_->config_->model.type == "phimoe")) || ((search_->GetSequenceLength() == 8193) && (model_->config_->model.type == "phi3small"))) {
      auto current_seq = cpu_span<int32_t>(GetSequence(0).CopyDeviceToCpu());
      RewindToLength(0);
      AppendTokens(current_seq);
    }
  }

  if (!computed_logits_) {
    auto next_tokens = search_->GetNextTokens();
    if (last_action_ == Action::rewound)
      search_->AppendTokens(next_tokens);
    const auto t_cl_pre = profile ? GenProfileClock::now() : GenProfileClock::time_point{};
    ComputeLogits(next_tokens);
    if (profile) {
      call_compute_logits_ns = GenNsBetween(t_cl_pre, GenProfileClock::now());
    }
  }
  if (guidance_logits_processor_) {
    auto logits = GetLogits();
    guidance_logits_processor_->ProcessLogits(logits);
  }
  computed_logits_ = false;
  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);

  if (g_log.enabled && g_log.generate_next_token) {
    auto& stream = Log("generate_next_token");
    stream << SGR::Fg_Green << "do_sample: " << SGR::Reset << search.do_sample << ' '
           << SGR::Fg_Green << "top_k: " << SGR::Reset << search.top_k << ' '
           << SGR::Fg_Green << "top_p: " << SGR::Reset << search.top_p << ' '
           << SGR::Fg_Green << "temperature: " << SGR::Reset << search.temperature << ' '
           << SGR::Fg_Cyan << "sequence length: " << SGR::Reset << search_->GetSequenceLength()
           << std::endl;
  }

  last_action_ = Action::generated;

  // Bracket only the actual sampling op so call_sampling_ns reflects the
  // time spent in SelectTop / SampleTopK / SampleTopKTopP / SampleTopP.
  // The branching+validation above is bookkeeping and stays in "other".
  // Note: the original code had an early "return" on the SelectTop path;
  // converting that branch to an else preserves identical behaviour
  // (validation and Sample* calls were already gated on do_sample) while
  // letting the RAII committer fire on a single exit path.
  const auto t_sample_pre = profile ? GenProfileClock::now() : GenProfileClock::time_point{};
  if (!search.do_sample || search.top_k == 1 || search.temperature == 0) {
    search_->SelectTop();
  } else {
    // The user explicitly called TopK_TopP on a beam search
    if (search.num_beams != 1)
      throw std::runtime_error("TopK and TopP cannot be used with a beam search");

    // Sanity checks
    if (search.top_p < 0.0f || search.top_p > 1.0f)
      throw std::runtime_error("top_p must be between 0.0 and 1.0");
    if (search.top_k < 0)
      throw std::runtime_error("top_k must be 0 or greater");

    if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1) {
      search_->SampleTopKTopP(search.top_k, search.top_p, search.temperature);
    } else if (search.top_k > 1) {
      search_->SampleTopK(search.top_k, search.temperature);
    } else {
      assert(search.top_k == 0);
      search_->SampleTopP(search.top_p, search.temperature);
    }
  }
  if (profile) {
    call_sampling_ns = GenNsBetween(t_sample_pre, GenProfileClock::now());
  }
}

void Generator::RewindToLength(size_t new_length) {
  if (model_->config_->model.type == "whisper" || model_->config_->model.type == "phi3v" || model_->config_->model.type == "decoder-pipeline")
    throw std::runtime_error("RewindTo is currently not supported for " + model_->config_->model.type + ".");
  if (new_length > search_->GetSequenceLength())
    throw std::runtime_error("Cannot rewind to a length greater than the current sequence length");
  if (new_length == search_->GetSequenceLength())
    return;
  size_t batch_size = search_->params_->search.batch_size;
  if (batch_size > 1 && new_length != 0)
    throw std::runtime_error("RewindToLength must be called with new_length=0 when batch_size > 1");
  search_->RewindTo(new_length);
  state_->RewindTo(new_length);
  if (guidance_logits_processor_) {
    guidance_logits_processor_->ResetWithoutCompute();
  }
  computed_logits_ = false;
  last_action_ = Action::rewound;
}

DeviceSpan<float> Generator::GetLogits() {
  if (!computed_logits_) {
    ComputeLogits(search_->GetNextTokens());
  }
  return search_->GetLogits();
}

DeviceSpan<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

}  // namespace Generators
