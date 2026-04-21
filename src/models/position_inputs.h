#pragma once

namespace Generators {

struct PositionInputs {
  virtual ~PositionInputs() = default;
  virtual void Add() = 0;
  virtual void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) = 0;
  virtual void RewindTo(size_t index) = 0;

  // [NO_CHUNK_EXPERIMENTAL] Called once after the single prefill Run on the
  // legacy fixed_prompt_length path. Default no-op; overridden only by
  // DefaultPositionInputs to zero the trailing pad cells of the static
  // attention_mask so subsequent decode steps see mask sum == real_length.
  // WindowedPositionInputs does NOT override this; it handles its own
  // last-chunk pad cleanup via the DeferLastChunkPadClearLeft /
  // ConsumeDeferredPadClearLeft pair inside Update().
  virtual void RewindStaticMaskAfterPadding(int real_length, int padded_length) {
    (void)real_length;
    (void)padded_length;
  }
};

struct DefaultPositionInputs : PositionInputs {
  DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk, const std::string& attention_mask_name);

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;

  void RewindTo(size_t index) override;

  // [NO_CHUNK_EXPERIMENTAL] See base class banner.
  void RewindStaticMaskAfterPadding(int real_length, int padded_length) override;

 private:
  void AddAttentionMask();
  void AddPositionIDs();

  void CreateNextPositionIDsTensor();
  void CreateNextAttentionMaskTensor(int total_length);

  void UpdatePositionIDs(int total_length, int new_length);
  void UpdateAttentionMask(int total_length, int new_length);

  template <typename T>
  void InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk);
  template <typename T>
  void CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void InitializeStaticMask(OrtValue& cpu_attention_mask);

  void RewindMask(size_t index);

  // This returns true when either:
  // 1. Graph capture is enabled, OR
  // 2. Past-present buffer sharing is enabled AND the device is NvTensorRtRtx
  // Both scenarios require static mask allocation and special shape handling for
  // optimization.
  bool ShouldUseStaticMaskHandling() const;

  const Model& model_;
  State& state_;
  std::string attention_mask_name_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask

  bool has_mask_input_{};
  bool has_posid_input_{};

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<Tensor> position_ids_;
  std::unique_ptr<Tensor> position_ids_next_;      // Replaces position_ids_ after the first Run() call
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<Tensor> attention_mask_;
  std::unique_ptr<Tensor> attention_mask_next_;  // Replaces attention_mask_ after each run

  bool is_first_update_{true};
};

// Certain models can only process a fixed number of tokens at a time.
// For example, given a prompt with 120 tokens, and a model that can only process 20 tokens at a time,
// this class will split the position ids into 6 windows of 20 tokens each.
// At each update step, the next window of position ids is prepared.
// This is done until all windows have been processed before switching to the model-generation phase
// where position ids are prepared one id at a time.
// This class will also prepare the attention mask for each iteration. The attention mask buffer is allocated just
// once and reused for each iteration by setting the mask to 1 for current window tokens and previously active window tokens
// In contrast, DefaultPositionInputs processes all position ids at once.
struct WindowedPositionInputs : PositionInputs {
  WindowedPositionInputs(State& state);
  WindowedPositionInputs(const WindowedPositionInputs&) = delete;
  WindowedPositionInputs& operator=(const WindowedPositionInputs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;
  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedPositionInputs does not support RewindTo.");
  };

 private:
  // Alignment-specific chunk-0 prefill initializers.
  //
  // "left"  -> exercised end-to-end on this PR's stack (chat mode, chunked
  //            prefill, DefaultKeyValueCache). This is the tested path.
  // "right" -> reference implementation for Phi-3.5-MoE-style models paired
  //            with WindowedKeyValueCache. RISK: NOT YET VALIDATED on our
  //            EP stack in this PR; pairing "right" with DefaultKeyValueCache
  //            will silently corrupt attention outputs. See the per-helper
  //            banners in the .cpp for the full risk description.
  //
  // TODO(sliding_window): add an end-to-end test for alignment="right" +
  //   WindowedKeyValueCache before treating that path as supported here.
  void InitChunk0Left(DeviceSpan<int32_t> next_tokens, bool has_prior_tokens);
  void InitChunk0Right(DeviceSpan<int32_t> next_tokens);

  // Intermediate prefill chunk (chunks 1..num_windows_-1). Split per
  // alignment because the mask-walk direction is opposite and only ~6 lines
  // of posid iota are genuinely common; a single "shared" function would
  // branch on is_left_alignment_ throughout its body and blur the
  // frozen-vs-maintained boundary between the two paths.
  //
  // *Right helpers are kept identical to the pre-split upstream behaviour
  // and are NOT YET VALIDATED end-to-end on this PR's stack. Treat them as
  // frozen reference code -- see the per-helper banners in the .cpp and
  // the class-level banner above InitChunk0Right.
  void UpdateIntermediateChunkLeft();
  void UpdateIntermediateChunkRight();

  // Per-decode-step mask extension. Split per alignment for the same reason
  // as UpdateIntermediateChunk*: the write direction is opposite and there
  // is nothing meaningful to share beyond a one-line has_mask_input_ guard.
  void UpdateDecodeMaskExtensionLeft();
  void UpdateDecodeMaskExtensionRight();

  // Two-phase last-prefill-chunk pad cleanup for alignment="left":
  //   1. DeferLastChunkPadClearLeft is called from the prefill branches
  //      once window_index_ reaches num_windows_. It records the pad count
  //      in pending_last_chunk_pad_clear_ but does NOT touch the mask,
  //      because the last prefill chunk's Run() has not yet executed and
  //      its GQA expects total_sequence_length == forward_offset_
  //      (including pads) to place K/V at past_seq = forward_offset_ -
  //      window_size_. Clearing too early shrinks total_sequence_length,
  //      which makes past_seq collide with the prior chunk's K/V slots.
  //   2. ConsumeDeferredPadClearLeft is called at the TOP of every
  //      subsequent Update() (before branch dispatch), AFTER the last
  //      prefill chunk's Run() has completed. The next Update may be a
  //      decode step OR a fresh prefill-init from another AppendTokens()
  //      batch (chat mode's system-then-user flow); both need the mask
  //      clean before they touch forward_offset_. It zeros the pad cells
  //      and rewinds forward_offset_ so the next branch picks up at
  //      exactly num_reals.
  // Both are no-ops for alignment="right" (upstream paints pad cells as 0
  // inline during chunk-0 init, no post-chunk cleanup needed).
  void DeferLastChunkPadClearLeft(DeviceSpan<int32_t> next_tokens);
  void ConsumeDeferredPadClearLeft();

  // Counts pad tokens in next_tokens[chunk_index*window_size_ ..
  // chunk_index*window_size_+window_size_). Any tokens missing past the end
  // of next_tokens are counted as pads as well (the last chunk of a prompt
  // whose length is not a multiple of window_size_).
  size_t CountPadsInChunk(DeviceSpan<int32_t> next_tokens, size_t chunk_index) const;

  State& state_;
  const Model& model_{state_.model_};

  bool has_mask_input_{};
  bool has_posid_input_{};

  // True iff model.decoder.sliding_window.alignment == "left". Cached at ctor
  // so the hot Update path doesn't do a string compare every call.
  bool is_left_alignment_{};

  std::array<int64_t, 2> position_ids_shape_{};
  ONNXTensorElementDataType position_ids_type_{};
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};
  ONNXTensorElementDataType attention_mask_type_{};
  std::unique_ptr<OrtValue> attention_mask_;

  // Mask-layout cursors. Each alignment uses ONE of these; the other is
  // unused (a small memory overhead in exchange for a clean separation).
  //
  // alignment="left"  (rightward-growing layout):
  //   Mask is packed at the HEAD of the buffer: slots [0, forward_offset_)
  //   are the valid K/V (real tokens + pads-in-progress). Every prefill
  //   chunk / decode step extends forward_offset_ to the RIGHT. Finalize
  //   rewinds forward_offset_ by the last chunk's pad count so decode picks
  //   up exactly at num_reals. This matches DefaultKeyValueCache's packed-
  //   at-slot-0 K/V layout 1:1 and makes mask[i] <-> K/V slot i.
  //
  // alignment="right" (leftward-growing layout, reference implementation):
  //   Mask grows from the TAIL of the buffer toward the HEAD. Chunk 0 lives
  //   at [context_len-window_size, context_len). backward_offset_ starts at
  //   the first real slot - 1 and walks LEFTWARD as decode overwrites the
  //   leading pad slots. This matches WindowedKeyValueCache's sliding
  //   layout. RISK: Not validated on this stack; see class-level banner
  //   above.
  size_t attention_mask_forward_offset_{0};
  size_t attention_mask_backward_offset_{~0U};

  size_t attention_mask_index_{~0U};
  size_t position_ids_index_{~0U};

  size_t window_size_{};
  size_t num_windows_{};
  size_t window_index_{};

  // Total number of real (non-pad) prompt+generated tokens accumulated across
  // all prefill chunks and decode steps. Under alignment="left" this is used
  // to compute the next decode position_id and to detect "not the first
  // prefill" (multi-AppendTokenSequences / chat mode). It is also maintained
  // under alignment="right" but only as a diagnostic; the right decode path
  // uses the pre-split data[last]+1 formula.
  size_t historical_num_tokens_{0};

  // Deferred pad-clear count for alignment="left". Set by the prefill
  // branch of Update() when it dispatches the last prefill chunk; consumed
  // and cleared at the TOP of every subsequent Update() (whether decode or
  // another prefill batch via AppendTokens in chat mode). See
  // DeferLastChunkPadClearLeft / ConsumeDeferredPadClearLeft in the .cpp
  // for the full timing rationale.
  size_t pending_last_chunk_pad_clear_{0};
};
// Qwen2-VL uses 3D rotary position embeddings (mrope) for multimodal (vision + text) content.
// Position IDs have shape [3, batch_size, seq_len] where the 3 dimensions represent:
//   - Dimensions 0: Temporal position
//   - Dimensions 1: Height position
//   - Dimensions 2: Width position
// For text, all 3 dimensions are identical. For vision, they are distinct.
// This class implements the logic from `get_rope_index` to build these 3D IDs.
struct Qwen2VLPositionInputs : PositionInputs {
  Qwen2VLPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk);
  Qwen2VLPositionInputs(const Qwen2VLPositionInputs&) = delete;
  Qwen2VLPositionInputs& operator=(const Qwen2VLPositionInputs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;
  void RewindTo(size_t index) override;

  void SetGridTensors(const std::shared_ptr<Tensor>& image_grid_thw,
                      const std::shared_ptr<Tensor>& video_grid_thw,
                      const std::shared_ptr<Tensor>& second_per_grid_ts);

  // Friend declarations for functors that need access to private methods
  friend struct InitPositionIdsFunctor;
  friend struct InitAttentionMaskFunctor;

 private:
  void AddPositionIDs();
  void AddAttentionMask();

  template <typename T>
  void CreateAndInitialize3DPositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 3> shape);
  void Update3DPositionIDs(int base_pos);

  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  void UpdateAttentionMask();

  const Model& model_;
  State& state_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;

  bool has_mask_input_{false};
  bool has_posid_input_{false};

  std::array<int64_t, 3> position_ids_shape_{};  // {3, batch_size, sequence_length} for 3D positions
  std::unique_ptr<Tensor> position_ids_;

  std::array<int64_t, 2> attention_mask_shape_{};  // {batch_size, sequence_length}
  std::unique_ptr<Tensor> attention_mask_;

  bool is_first_update_{true};

  // Cached data from processor
  std::shared_ptr<Tensor> image_grid_thw_;
  std::shared_ptr<Tensor> video_grid_thw_;
  std::shared_ptr<Tensor> second_per_grid_ts_;
  std::vector<int64_t> rope_deltas_;

  // Config values initialized from model.config_ in constructor
  const int32_t image_token_id_;
  const int32_t video_token_id_;
  const int32_t vision_start_token_id_;
  const float tokens_per_second_;
  const int32_t spatial_merge_size_;
};

std::unique_ptr<PositionInputs> CreatePositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name);

}  // namespace Generators
