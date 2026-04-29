// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace Generators {

// Note: memory allocated through MorphiZenEP interface is host/cpu accessible
// because the underlying AMD APU iGPU shares physical memory with the host.
// Mirrors the RyzenAIInterface pattern; the difference is that this provider
// targets OrtHardwareDeviceType_GPU + AMD GPU vendor id (0x1002) instead of
// the NPU.
struct MorphiZenEPInterface : DeviceInterface {
  using ProviderOptions = std::vector<std::pair<std::string, std::string>>;

  virtual void SetupProvider(OrtSessionOptions&, const ProviderOptions&) = 0;

  static void Shutdown();
};

MorphiZenEPInterface* GetMorphiZenEPInterface();

}  // namespace Generators
