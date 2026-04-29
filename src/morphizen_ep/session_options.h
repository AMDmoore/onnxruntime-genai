// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../generators.h"

namespace Generators::MorphiZenEPExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

}  // namespace Generators::MorphiZenEPExecutionProvider
