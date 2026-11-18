// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_tokenizer.h"

// The extensions package used by this build does not export a word-piece query.
// These stand-ins keep timestamp grouping compiling. Every field stays zero, so
// that path does not invent word boundaries.
enum OrtxWordBoundaryStyle {
  OrtxWordBoundaryStyle_None = 0,
  OrtxWordBoundaryStyle_SuffixBpe = 1,
};

struct OrtxWordPieceInfo {
  int is_special;
  int starts_word;
  int ends_word;
  OrtxWordBoundaryStyle boundary_style;
};
