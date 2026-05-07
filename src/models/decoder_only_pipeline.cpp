// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../logging.h"
#include "../tracing.h"
#include "decoder_only_pipeline.h"
#include "env_utils.h"
#include "windowed_kv_cache.h"

#include <chrono>
#include <cstdio>

namespace Generators {

namespace {
// Monotonic clock used by the RunPipeline overhead profiler. We deliberately
// pick steady_clock (not high_resolution_clock, which is not guaranteed
// monotonic on Windows) so accumulated deltas can never go negative.
using ProfileClock = std::chrono::steady_clock;

inline uint64_t NsBetween(ProfileClock::time_point a, ProfileClock::time_point b) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}
}  // namespace

DecoderOnlyPipelineModel::DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, ort_env_{ort_env} {
  for (const auto& model : config_->model.decoder.pipeline) {
    sessions_.emplace_back(CreateSession(ort_env, model.filename, GetSessionOptions(model.model_id)));
  }

  for (auto& session : sessions_) {
    session_info_.Add(*session);
  }
}

std::unique_ptr<State> DecoderOnlyPipelineModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                             const GeneratorParams& params) const {
  return std::make_unique<DecoderOnlyPipelineState>(*this, sequence_lengths, params);
}

IntermediatePipelineState::IntermediatePipelineState(const DecoderOnlyPipelineModel& model, const GeneratorParams& params,
                                                     size_t pipeline_state_index)
    : State{params, model},
      id_{pipeline_state_index},
      model_{model} {}

bool IntermediatePipelineState::HasInput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].inputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].inputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediatePipelineState::HasOutput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].outputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].outputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediatePipelineState::SupportsPrimaryDevice() const {
  if (model_.p_device_->GetType() == DeviceType::CPU || model_.p_device_->GetType() == DeviceType::QNN) {
    return true;
  } else if (model_.p_device_->GetType() == DeviceType::CUDA) {
    if (!model_.config_->model.decoder.pipeline[id_].session_options.has_value()) {
      // No session options, so this session uses the default session options.
      // Default session options supports the cuda device type.
      return true;
    } else if (auto& provider_options = (*model_.config_->model.decoder.pipeline[id_].session_options).provider_options;
               std::any_of(provider_options.begin(), provider_options.end(),
                           [](const Config::ProviderOptions& elem) { return elem.name == "cuda"; })) {
      // cuda is listed as one of the providers. This session supports the cuda device type.
      return true;
    } else {
      // cuda is not listed as one of the providers. This session does not support the cuda device type.
      return false;
    }
  } else if (model_.p_device_->GetType() == DeviceType::DML) {
    if (!model_.config_->model.decoder.pipeline[id_].session_options.has_value()) {
      return true;
    } else if (auto& provider_options = (*model_.config_->model.decoder.pipeline[id_].session_options).provider_options;
               std::any_of(provider_options.begin(), provider_options.end(),
                           [](const Config::ProviderOptions& elem) { return elem.name == "DML"; })) {
      return true;
    } else {
      return false;
    }
  } else if (model_.p_device_->GetType() == DeviceType::MorphiZenEP) {
    if (!model_.config_->model.decoder.pipeline[id_].session_options.has_value()) {
      return true;
    } else if (auto& provider_options = (*model_.config_->model.decoder.pipeline[id_].session_options).provider_options;
               std::any_of(provider_options.begin(), provider_options.end(),
                           [](const Config::ProviderOptions& elem) { return elem.name == "MorphiZenEP"; })) {
      return true;
    } else {
      // Scenario: VLM pipeline where the embedding sub-model runs on
      // CPU EP (session_options: {}).
      // For MorphiZenEP, p_device_inputs_ defaults to CPU, so managed
      // inputs (input_ids, etc.) reside in CPU memory. MorphiZenEP can
      // also access CPU memory, so the sub-model is compatible.
      return model_.p_device_inputs_->GetType() == DeviceType::CPU;
    }
  }

  return false;
}

DeviceSpan<float> IntermediatePipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                 DeviceSpan<int32_t> next_indices) {
  if (!model_.sessions_[id_]) {
    const_cast<DecoderOnlyPipelineModel*>(&model_)->sessions_[id_] =
        OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(model_.config_->model.decoder.pipeline[id_].filename)).c_str(),
                           model_.GetSessionOptions(model_.config_->model.decoder.pipeline[id_].model_id));
  }

  if (model_.config_->model.decoder.pipeline[id_].run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.pipeline[id_].run_options.value());
  }
  State::Run(*model_.sessions_[id_]);
  return {};
}

using NameToLayerIdxMap = std::unordered_map<std::string, size_t>;

static NameToLayerIdxMap GeneratePastKeyNameToLayerIdxMap(const Config& config) {
  const size_t num_layers = config.model.decoder.num_hidden_layers;
  const std::string& past_key_name_template = config.model.decoder.inputs.past_key_names;
  NameToLayerIdxMap m{};
  for (size_t i = 0; i < num_layers; ++i) {
    m.emplace(ComposeKeyValueName(past_key_name_template, static_cast<int>(i)), i);
  }
  return m;
}

static std::vector<size_t> GetLayerIndicesSetFromPastKeyNameInputs(
    const NameToLayerIdxMap& past_key_name_to_layer_idx, std::span<const std::string> inputs) {
  std::vector<size_t> layer_indices{};
  for (const auto& input_name : inputs) {
    const auto it = past_key_name_to_layer_idx.find(input_name);
    if (it != past_key_name_to_layer_idx.end()) {
      layer_indices.push_back(it->second);
    }
  }
  // sort and remove duplicates
  std::sort(layer_indices.begin(), layer_indices.end());
  layer_indices.erase(std::unique(layer_indices.begin(), layer_indices.end()),
                      layer_indices.end());
  return layer_indices;
}

DecoderOnlyPipelineState::DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      input_ids_{CreateInputIDs(*this)},
      model_{model},
      key_value_cache_{CreateKeyValueCache(*this)},
      do_key_value_cache_partial_update_{key_value_cache_ && key_value_cache_->IsPartialUpdateSupported()},
      recurrent_state_{CreateRecurrentState(*this)},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)} {
  // Opt-in CPU-overhead profiler. See decoder_only_pipeline.h for what each
  // bucket measures. Cost when disabled is a single bool check per stage.
  GetEnv("ORTGENAI_PIPELINE_OVERHEAD_PROFILE", overhead_profile_enabled_);

  input_ids_->Add();
  position_inputs_->Add();
  logits_.Add();
  if (key_value_cache_) {
    key_value_cache_->Add();
  }
  if (recurrent_state_) {
    recurrent_state_->Add();
  }

  const auto& config_pipeline = model_.config_->model.decoder.pipeline;

  for (size_t i = 0; i < config_pipeline.size(); ++i) {
    auto pipeline_model_state = std::make_unique<IntermediatePipelineState>(model_, params, pipeline_states_.size());
    pipeline_states_.emplace_back(std::move(pipeline_model_state));
  }

  if (do_key_value_cache_partial_update_) {
    const auto past_key_name_to_layer_idx = GeneratePastKeyNameToLayerIdxMap(*model_.config_);

    std::map<std::vector<size_t>, size_t> layer_indices_to_update_record_idx{};
    std::unordered_set<size_t> layer_indices_encountered{};

    for (size_t i = 0; i < config_pipeline.size(); ++i) {
      const auto& pipeline_model = config_pipeline[i];

      const auto layer_indices = GetLayerIndicesSetFromPastKeyNameInputs(past_key_name_to_layer_idx,
                                                                         pipeline_model.inputs);

      if (layer_indices.empty()) {
        continue;
      }

      size_t record_idx{};

      if (auto layer_indices_to_update_record_it = layer_indices_to_update_record_idx.find(layer_indices);
          layer_indices_to_update_record_it != layer_indices_to_update_record_idx.end()) {
        // we have seen this exact set of layer indices before. reuse the existing record.
        record_idx = layer_indices_to_update_record_it->second;
      } else {
        // verify that the new set of layer indices is valid.
        // i.e., it is disjoint with the set of all layer indices we've seen so far.
        const bool layer_indices_valid =
            std::all_of(layer_indices.begin(), layer_indices.end(),
                        [&layer_indices_encountered](size_t layer_idx) {
                          return layer_indices_encountered.find(layer_idx) == layer_indices_encountered.end();
                        });

        if (!layer_indices_valid) {
          throw std::runtime_error(
              "Invalid layer indices. Layer index sets for partial key value cache update must be either an exact "
              "match with another set or disjoint with all other sets.");
        }

        // add a new record
        auto record = PartialKeyValueCacheUpdateRecord{};
        record.layer_indices = layer_indices;

        partial_kv_cache_update_records_.emplace_back(std::move(record));
        record_idx = partial_kv_cache_update_records_.size() - 1;

        // add layer_indices to what we've seen so far
        layer_indices_encountered.insert(layer_indices.begin(), layer_indices.end());
        layer_indices_to_update_record_idx.emplace(layer_indices, record_idx);
      }

      pipeline_state_id_to_partial_kv_cache_update_record_idx_.emplace(i, record_idx);
    }

    if (!partial_kv_cache_update_records_.empty()) {
      key_value_cache_update_worker_thread_.emplace();
    }
  }
}

DecoderOnlyPipelineState::~DecoderOnlyPipelineState() {
  if (!overhead_profile_enabled_) {
    return;
  }
  if (suppress_destructor_print_) {
    // Generator owns the unified report; nothing to print here.
    return;
  }
  if (prefill_stats_.stages == 0 && decode_stats_.stages == 0) {
    return;
  }

  // Print one line per phase. Times are reported per stage iteration so the
  // ratio is comparable across configs with different num_chunks/num_stages.
  // overhead = setup + teardown (the per-call rebind/forwarding work);
  // total    = setup + inner + teardown (everything inside the stage body).
  // Use stderr + fprintf to avoid interleaving with std::cout token streams
  // and to be safe in a destructor.
  auto print_phase = [](const char* label, const OverheadStats& s) {
    if (s.stages == 0 && s.runs == 0) {
      std::fprintf(stderr, "[OGA pipeline overhead] %s: 0 runs, 0 steps, 0 stages\n", label);
      return;
    }
    const double stages = static_cast<double>(s.stages);
    // Per-stage averages (inside RunPipeline only).
    const double setup_us = stages > 0.0 ? (s.setup_ns / 1000.0) / stages : 0.0;
    const double inner_us = stages > 0.0 ? (s.inner_run_ns / 1000.0) / stages : 0.0;
    const double teardown_us = stages > 0.0 ? (s.teardown_ns / 1000.0) / stages : 0.0;
    const double stage_total_us = setup_us + inner_us + teardown_us;
    const double stage_overhead_us = setup_us + teardown_us;
    const double stage_ratio_pct = stage_total_us > 0.0 ? (stage_overhead_us / stage_total_us) * 100.0 : 0.0;

    const double runs = static_cast<double>(s.runs);
    const double steps = static_cast<double>(s.steps);
    const double chunks_per_run = runs > 0.0 ? steps / runs : 0.0;  // == num_chunks for sliding-window prefill
    const double stages_per_run = runs > 0.0 ? stages / runs : 0.0;

    // Per-Run() rollup: sum the stage measurements across chunks, then add
    // the outer_ns bucket (UpdateInputsOutputs + chunk slide + post-cleanup).
    // Together these cover the WHOLE Run() wall clock, so the totals here
    // should match the benchmark's per-token / per-prefill numbers.
    const double rp_inner_per_run_us = inner_us * stages_per_run;
    const double rp_orchestration_per_run_us = stage_overhead_us * stages_per_run;
    const double outer_per_run_us = runs > 0.0 ? (s.outer_ns / 1000.0) / runs : 0.0;
    const double full_run_us = rp_inner_per_run_us + rp_orchestration_per_run_us + outer_per_run_us;
    const double oga_overhead_per_run_us = rp_orchestration_per_run_us + outer_per_run_us;
    const double full_overhead_ratio_pct = full_run_us > 0.0 ? (oga_overhead_per_run_us / full_run_us) * 100.0 : 0.0;

    std::fprintf(stderr,
                 "[OGA pipeline overhead] %s: %llu runs, %llu steps (%.2f chunks/run), %llu stages (%.2f stages/run)\n"
                 "    [in RunPipeline] per stage:  setup %8.2f us + inner %8.2f us + teardown %8.2f us = %8.2f us\n"
                 "                     in-RunPipeline overhead ratio (setup+teardown)/stage_total: %6.2f%%\n"
                 "    per Run() WHOLE-STAGE breakdown (covers full Run() wall clock):\n"
                 "        RunPipeline x %.2f chunks: inner %10.2f us  + orchestration %8.2f us\n"
                 "        outside RunPipeline      :                   %10.2f us  (UpdateIO + chunk slide + post-cleanup)\n"
                 "        ---------------------------------------------------------------------------\n"
                 "        full Run() total         :                   %10.2f us\n"
                 "        OGA-side overhead (orchestration + outside)  %10.2f us  = %6.2f%% of full Run()\n",
                 label,
                 static_cast<unsigned long long>(s.runs),
                 static_cast<unsigned long long>(s.steps), chunks_per_run,
                 static_cast<unsigned long long>(s.stages), stages_per_run,
                 setup_us, inner_us, teardown_us, stage_total_us,
                 stage_ratio_pct,
                 chunks_per_run, rp_inner_per_run_us, rp_orchestration_per_run_us,
                 outer_per_run_us,
                 full_run_us,
                 oga_overhead_per_run_us, full_overhead_ratio_pct);
  };

  std::fprintf(stderr, "\n[OGA pipeline overhead] DecoderOnlyPipelineState summary\n");
  print_phase("prefill", prefill_stats_);
  print_phase("decode ", decode_stats_);
  std::fflush(stderr);
}

void DecoderOnlyPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (auto& session : model_.sessions_) {
    extra_inputs_.Add(extra_inputs, session->GetInputNames());
  }
}

void DecoderOnlyPipelineState::RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, bool is_last_chunk) {
  // Bump per-step counters once per call, regardless of how many stages survive
  // the run_on_prompt / is_lm_head / run_on_token_gen filters below.
  if (overhead_profile_enabled_) {
    if (first_run_) {
      ++prefill_stats_.steps;
    } else {
      ++decode_stats_.steps;
    }
  }

  for (auto& pipeline_state : pipeline_states_) {
    if (first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_prompt) {
      continue;
    } else if (first_run_ && model_.config_->model.decoder.pipeline[pipeline_state->id_].is_lm_head && !is_last_chunk) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
      continue;
    }

    // t0 is taken AFTER the filter so the unconditional MakeString allocation
    // for DurationTrace and the rest of the per-stage rebind work below are
    // included in the "setup" bucket, matching the analysis. Stages that
    // continue out above are not charged any cost.
    const auto t0 = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};

    DurationTrace trace{MakeString("DecoderOnlyPipelineState::RunPipeline[", pipeline_state->id_, "]")};

    if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx > -1) {
      if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx >=
          static_cast<int>(model_.sessions_.size())) {
        throw std::runtime_error(
            MakeString("Invalid reset_session_idx ", model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx,
                       " for pipeline model ", model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id));
      }
      (const_cast<DecoderOnlyPipelineModel*>(&model_))->sessions_[model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx].reset();
    }

    auto* const partial_kv_cache_update_record = [&]() -> PartialKeyValueCacheUpdateRecord* {
      auto it = pipeline_state_id_to_partial_kv_cache_update_record_idx_.find(pipeline_state->id_);
      if (it != pipeline_state_id_to_partial_kv_cache_update_record_idx_.end()) {
        return &partial_kv_cache_update_records_[it->second];
      }
      return nullptr;
    }();

    // If there is any outstanding partial KV cache update, wait for it to finish.
    // It is important to synchronize at this point, before setting input/output tensors for this pipeline state run,
    // because a KV cache update may replace the KV cache input/output tensors.
    if (partial_kv_cache_update_record) {
      if (partial_kv_cache_update_record->outstanding_update.valid()) {
        partial_kv_cache_update_record->outstanding_update.get();
      }
    }

    // Clear the intermediate pipeline state outputs from the previous runs.
    // These outputs will be replaced by the outputs from the current run.
    for (const auto& output_name : pipeline_state->output_names_) {
      if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
        ortvalue_store_.erase(iter);
      }
    }
    pipeline_state->ClearIO();

    // Managed inputs and outputs are those inputs and outputs that the
    // Model knows how to create and update from one run to the next.

    // Add all the managed inputs to the intermediate pipeline state
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasInput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->input_names_.push_back(input_name);
        pipeline_state->inputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add outputs from the previous pipeline states to the current pipeline state
    for (auto& [name, ortvalue] : ortvalue_store_) {
      if (pipeline_state->HasInput(name)) {
        pipeline_state->input_names_.push_back(name.c_str());
        pipeline_state->inputs_.push_back(ortvalue.get());
      }
    }

    // Add all the managed outputs to the intermediate pipeline state
    for (const auto& output_name : output_names_) {
      if (pipeline_state->HasOutput(output_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed output ", output_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->output_names_.push_back(output_name);
        pipeline_state->outputs_.push_back(State::GetOutput(output_name));
      }
    }

    // Output of pipeline models could also be managed inputs.
    // For example, the output of a pipeline model could be the key-value cache.
    // In such cases, use the managed output buffers and register them with the pipeline model as outputs.
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasOutput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->output_names_.push_back(input_name);
        pipeline_state->outputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add all the remaining outputs for the intermediate pipeline state
    for (const auto& output_name : model_.config_->model.decoder.pipeline[pipeline_state->id_].outputs) {
      if (std::none_of(pipeline_state->output_names_.begin(), pipeline_state->output_names_.end(),
                       [&](const std::string& elem) { return elem == output_name; })) {
        pipeline_state->output_names_.push_back(output_name.c_str());
        pipeline_state->outputs_.push_back(nullptr);
      }
    }

    // Run the intermediate pipeline state
    const auto t1 = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    pipeline_state->Run(total_length, next_tokens, next_indices);
    const auto t2 = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};

    // If there is any partial KV cache update to start, enqueue it.
    if (partial_kv_cache_update_record) {
      assert(key_value_cache_update_worker_thread_.has_value());
      auto update_fn = [&key_value_cache = *key_value_cache_.get(),
                        layer_indices = partial_kv_cache_update_record->layer_indices,
                        next_indices, total_length]() {
        key_value_cache.PartialUpdate(next_indices, total_length, layer_indices);
      };
      partial_kv_cache_update_record->outstanding_update = key_value_cache_update_worker_thread_->Enqueue(update_fn);
    }

    // Transfer ownership of all the non-managed outputs from the current pipeline state to the ortvalue store.
    // All non managed outputs are assumed to be on CPU
    for (size_t i = 0; i < pipeline_state->output_names_.size(); ++i) {
      if (std::none_of(output_names_.begin(), output_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; }) &&
          std::none_of(input_names_.begin(), input_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; })) {
        auto forwarded_output = model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.find(pipeline_state->output_names_[i]);
        if (forwarded_output != model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.end()) {
          ortvalue_store_[forwarded_output->second] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        } else {
          ortvalue_store_[pipeline_state->output_names_[i]] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        }
      }
    }

    if (overhead_profile_enabled_) {
      const auto t3 = ProfileClock::now();
      auto& stats = first_run_ ? prefill_stats_ : decode_stats_;
      ++stats.stages;
      stats.setup_ns += NsBetween(t0, t1);
      stats.inner_run_ns += NsBetween(t1, t2);
      stats.teardown_ns += NsBetween(t2, t3);
    }
  }
}

DeviceSpan<float> DecoderOnlyPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  DurationTrace trace{"DecoderOnlyPipelineState::Run"};

  // Bracket the WHOLE Run() so we can attribute everything not inside
  // RunPipeline() (UpdateInputsOutputs, between-chunk slide, post-loop
  // cleanup) to an "outer" bucket. Without this, the per-Run total would
  // miss work that the OGA-side State path does on every step.
  const auto t_run_start = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
  uint64_t run_pipeline_total_ns_local = 0;
  uint64_t chunk_loop_body_ns_local = 0;

  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  // first_run_ should be thought of as prompt_processing_run_.
  // It is true only for the prompt processing part when the provided tokens are more than 1.
  // Use padded shape (from input_ids_) so the chunked sliding-window path sees
  // window_size > 1 even on the last chunk.
  first_run_ = static_cast<size_t>(input_ids_->GetShape()[1]) > 1;
  size_t num_chunks{1};
  if (first_run_ && model_.config_->model.decoder.sliding_window.has_value()) {
    const int window_size = model_.config_->model.decoder.sliding_window->window_size;
    num_chunks = (next_tokens.size() + window_size - 1) / window_size;
  }

  // Capture the phase BEFORE the chunk loop runs (and before first_run_ is
  // flipped to false at the bottom of Run()) so the post-loop accounting
  // attributes the outer time to the correct bucket.
  const bool profile_phase_is_prefill = first_run_;

  // Count outer Run() invocations per phase so the report can show
  // chunks/run = steps/runs (1.0 for non-chunked prefill or decode,
  // > 1.0 when sliding-window chunking multiplies the rebind tax).
  if (overhead_profile_enabled_) {
    if (profile_phase_is_prefill) {
      ++prefill_stats_.runs;
    } else {
      ++decode_stats_.runs;
    }
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    // t_chunk_pre brackets the WHOLE per-chunk iteration body (RunPipeline +
    // between-chunk slide). Lets the report attribute per-chunk wall clock
    // independently of UpdateInputsOutputs/post-loop work which happen
    // once per Run() and live in outer_ns instead.
    const auto t_chunk_pre = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};

    const auto t_rp_pre = overhead_profile_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    RunPipeline(total_length, next_tokens, next_indices, (i == num_chunks - 1));
    if (overhead_profile_enabled_) {
      run_pipeline_total_ns_local += NsBetween(t_rp_pre, ProfileClock::now());
    }

    if (model_.config_->model.decoder.sliding_window.has_value() && i < num_chunks - 1) {
      // Sliding the window over the input_ids, key_cache, and value_cache, position_ids, and attention_mask
      input_ids_->Update(next_tokens);
      UpdateKeyValueCache(next_indices, total_length);
      position_inputs_->Update(next_tokens, total_length, static_cast<int>(input_ids_->GetShape()[1]));
      logits_.Update(WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get()),
                     static_cast<int>(input_ids_->GetShape()[1]));
    }

    if (overhead_profile_enabled_) {
      chunk_loop_body_ns_local += NsBetween(t_chunk_pre, ProfileClock::now());
    }
  }

  // Clear the outputs of the pipeline models that are only run on prompt since this cannot happen earlier.
  if (!first_run_) {
    for (auto& pipeline_state : pipeline_states_) {
      if (!model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
        for (const auto& output_name : pipeline_state->output_names_) {
          if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
            ortvalue_store_.erase(iter);
          }
        }
      }
    }
  }

  // Last-prefill-chunk pad cleanup (alignment="left") is handled by
  // WindowedPositionInputs::Update itself via a two-phase defer/consume
  // pair: DeferLastChunkPadClearLeft records the pad count at the end of
  // the last prefill Update(), and ConsumeDeferredPadClearLeft zeros the
  // mask cells at the TOP of the NEXT Update() call -- which may be a
  // decode step OR the chunk-0 init of a subsequent AppendTokens() batch
  // (chat mode's system-then-user flow). Either way the consume runs AFTER
  // the last prefill Run() has completed, so total_sequence_length on that
  // Run() still equals window_size*num_windows (GQA places K/V at the
  // correct past_seq = forward_offset - window_size slot). No explicit
  // post-prefill hook is needed here.

  // [NO_CHUNK_EXPERIMENTAL] Legacy no-chunk static-shape path. When
  // fixed_prompt_length is set the prompt ran through DefaultInputIDs /
  // DefaultPositionInputs in a single session.Run with pad tokens at the
  // tail of input_ids; clear the matching trailing mask cells here so
  // decode sees mask sum == real_length. Mutually exclusive with the
  // sliding_window block above by config-load validation.
  const int fixed_len = model_.config_->model.decoder.fixed_prompt_length;
  if (first_run_ && fixed_len > 0 && total_length < padded_total_) {
    position_inputs_->RewindStaticMaskAfterPadding(total_length, padded_total_);
  }

  // Charge everything in Run() that wasn't inside RunPipeline() to outer_ns.
  // This is what makes the report cover the WHOLE prefill / decode stage:
  // setup + inner + teardown + outer == full Run() wall clock.
  if (overhead_profile_enabled_) {
    const uint64_t total_run_ns = NsBetween(t_run_start, ProfileClock::now());
    auto& stats = profile_phase_is_prefill ? prefill_stats_ : decode_stats_;
    // Guard against measurement drift in case clock skew causes the
    // bracketed inner sum to slightly exceed the outer measurement.
    stats.outer_ns += (total_run_ns > run_pipeline_total_ns_local)
                          ? (total_run_ns - run_pipeline_total_ns_local)
                          : 0;
    stats.chunk_loop_body_ns += chunk_loop_body_ns_local;
  }

  first_run_ = false;

  return logits_.Get();
}

void DecoderOnlyPipelineState::UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length) {
  if (key_value_cache_) {
    const bool outstanding_key_value_cache_partial_update =
        do_key_value_cache_partial_update_ &&
        std::any_of(partial_kv_cache_update_records_.rbegin(),
                    partial_kv_cache_update_records_.rend(),
                    [](const PartialKeyValueCacheUpdateRecord& record) {
                      return record.outstanding_update.valid();
                    });

    if (outstanding_key_value_cache_partial_update) {
      // If there is any outstanding partial KV cache update, don't update the KV cache here.
    } else {
      key_value_cache_->Update(beam_indices, total_length);
    }
  }
}

void DecoderOnlyPipelineState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                                                   DeviceSpan<int32_t> beam_indices, int total_length) {
  const int actual_new = static_cast<int>(next_tokens.size());
  input_ids_->Update(next_tokens);
  size_t new_length = input_ids_->GetShape()[1];

  // For the chunked sliding-window path, WindowedInputIDs may emit a
  // window_size-shaped tensor that is larger than the actual_new tokens we are
  // appending (the last chunk includes pad tokens). Translate total_length into
  // that padded coordinate system so position_inputs_->Update sees a total
  // consistent with the padded next_tokens span it receives. For decode
  // (actual_new == new_length == 1) this reduces to total_length unchanged.
  padded_total_ = (total_length - actual_new) + static_cast<int>(new_length);

  auto padded_tokens = WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get());

  // WindowedPositionInputs needs the original (un-windowed) token span to calculate
  // the correct number of windows. padded_tokens is already windowed by WindowedInputIDs
  // (e.g. 128 tokens), so num_windows_ would be 1 instead of the actual chunk count.
  const bool slide_inputs = model_.config_->model.decoder.sliding_window.has_value() &&
                            model_.config_->model.decoder.sliding_window->slide_inputs;
  if (slide_inputs && actual_new > 1) {
    position_inputs_->Update(next_tokens, padded_total_, static_cast<int>(new_length));
  } else {
    position_inputs_->Update(padded_tokens, padded_total_, static_cast<int>(new_length));
  }

  UpdateKeyValueCache(beam_indices, total_length);
  if (recurrent_state_) {
    recurrent_state_->Update();
  }

  logits_.Update(padded_tokens, new_length);
}

OrtValue* DecoderOnlyPipelineState::GetOutput(const char* name) {
  // Check the ortvalue store to search if name is one of the non-managed output.
  auto it = ortvalue_store_.find(name);
  if (it != ortvalue_store_.end()) {
    return it->second.get();
  }

  // Search managed outputs saved in this State.
  return State::GetOutput(name);
}

}  // namespace Generators
