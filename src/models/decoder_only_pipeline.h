// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <future>
#include <optional>

#include "../worker_thread.h"
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"
#include "recurrent_state.h"

namespace Generators {

struct DecoderOnlyPipelineModel : Model {
  DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  DecoderOnlyPipelineModel(const DecoderOnlyPipelineModel&) = delete;
  DecoderOnlyPipelineModel& operator=(const DecoderOnlyPipelineModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::vector<std::unique_ptr<OrtSession>> sessions_;
  OrtEnv& ort_env_;
};

struct IntermediatePipelineState : State {
  IntermediatePipelineState(const DecoderOnlyPipelineModel& model, const GeneratorParams& params,
                            size_t pipeline_state_index);

  IntermediatePipelineState(const IntermediatePipelineState&) = delete;
  IntermediatePipelineState& operator=(const IntermediatePipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  bool HasInput(std::string_view name) const;

  bool HasOutput(std::string_view name) const;

  bool SupportsPrimaryDevice() const;

  size_t id_;

 private:
  const DecoderOnlyPipelineModel& model_;
};

struct DecoderOnlyPipelineState : State {
  DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model, DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  DecoderOnlyPipelineState(const DecoderOnlyPipelineState&) = delete;
  DecoderOnlyPipelineState& operator=(const DecoderOnlyPipelineState&) = delete;

  ~DecoderOnlyPipelineState() override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetOutput(const char* name) override;

  void RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices, bool is_last_chunk);

 protected:
  // Virtual hook called after each pipeline stage completes, before next stage starts.
  // Allows derived classes to modify stage outputs (e.g., inject vision embeddings).
  // stage_id: ID of the stage that just completed
  // next_tokens: current input tokens for pipeline
  virtual void OnStageComplete(size_t stage_id, DeviceSpan<int32_t>& next_tokens) {}

  // Stores all the outputs from the previous pipeline state(s)
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> ortvalue_store_;
  std::unique_ptr<InputIDs> input_ids_;  // Made protected for derived class access

 private:
  void UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int total_length);

  const DecoderOnlyPipelineModel& model_;
  std::vector<std::unique_ptr<IntermediatePipelineState>> pipeline_states_;

  struct PartialKeyValueCacheUpdateRecord {
    std::vector<size_t> layer_indices{};     // indicates which layers of the KV cache are to be updated
    std::future<void> outstanding_update{};  // future for an outstanding update task
  };

  std::map<size_t, size_t> pipeline_state_id_to_partial_kv_cache_update_record_idx_;
  std::vector<PartialKeyValueCacheUpdateRecord> partial_kv_cache_update_records_;

  Logits logits_{*this};

  std::unique_ptr<KeyValueCache> key_value_cache_;
  const bool do_key_value_cache_partial_update_;
  std::optional<WorkerThread> key_value_cache_update_worker_thread_{};

  std::unique_ptr<RecurrentState> recurrent_state_;
  std::unique_ptr<PositionInputs> position_inputs_;
  int padded_total_{};  // total_length in padded coordinate system for static-shape models
  ExtraInputs extra_inputs_{*this};

  // Per-stage CPU-overhead profiling for RunPipeline(). See cpp file for the
  // exact bracketing. Enable by setting ORTGENAI_PIPELINE_OVERHEAD_PROFILE=1.
  // The Generator destructor reads these via the public accessors below to
  // produce a single unified per-call breakdown report.
  bool overhead_profile_enabled_{false};
  bool suppress_destructor_print_{false};  // Set by Generator when it owns the report.
 public:
  struct OverheadStats {
    uint64_t runs{};          // number of outer Run() invocations in this phase
    uint64_t steps{};         // number of RunPipeline() invocations (== number of chunks for prefill)
    uint64_t stages{};        // number of stage iterations that actually executed
    uint64_t setup_ns{};      // [in RunPipeline] pre-session.Run rebind work
    uint64_t inner_run_ns{};  // [in RunPipeline] pipeline_state->Run() (the only call OGA must make)
    uint64_t teardown_ns{};   // [in RunPipeline] post-session.Run forwarding
    uint64_t outer_ns{};      // [in Run() but OUTSIDE RunPipeline] UpdateInputsOutputs +
                              // between-chunk slide (input_ids/KV/positions/logits Update) +
                              // post-loop ortvalue_store cleanup + static-shape mask cleanup.
                              // Together with setup+inner+teardown this sums to the full Run()
                              // wall-clock time, so the report covers the whole stage.
    uint64_t chunk_loop_body_ns{};  // Cumulative time across all iterations of the chunk loop
                                    // in Run() (one iteration per chunk). Each iteration
                                    // bracketed contains the RunPipeline call plus the
                                    // between-chunk slide block (slide runs num_chunks-1
                                    // times per Run, so per_chunk_slide = (chunk_loop_body
                                    // - sum_RunPipeline) / (steps - runs) on average).
                                    // Lets the report show "time cost per chunk" directly.
  };
  bool IsOverheadProfileEnabled() const { return overhead_profile_enabled_; }
  const OverheadStats& GetPrefillOverheadStats() const { return prefill_stats_; }
  const OverheadStats& GetDecodeOverheadStats() const { return decode_stats_; }
  void SuppressOverheadDestructorReport() { suppress_destructor_print_ = true; }

 private:
  OverheadStats prefill_stats_{};  // first_run_ == true
  OverheadStats decode_stats_{};   // first_run_ == false
};

}  // namespace Generators
