#!/usr/bin/env bash
# Model-free CPU vision test; keeps build output and temporary files in the worktree.
# No downloaded models, binary fixtures, C++ registration globals, or system changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-$ROOT/build-native-vision-regression}"
case "$BUILD" in
    "$ROOT"/*) ;;
    *) printf 'Build directory must be inside %s\n' "$ROOT" >&2; exit 2 ;;
esac
mkdir -p "$BUILD/tmp"
export TMPDIR="$BUILD/tmp"
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_BUILD_LLM=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON \
    -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_SEP_BUILD=OFF -DMNN_AAPL_FMWK=OFF -DMNN_METAL=OFF -DMNN_OPENCL=OFF \
    -DMNN_LLM_BUILD_DEMO=OFF -DLLM_SUPPORT_HTTP_RESOURCE=OFF
cmake --build "$BUILD" --target MNN --parallel "${JOBS:-4}"
if [[ "$(uname -s)" == Darwin ]]; then
    LINK=(-Wl,-force_load,"$BUILD/libMNN.a")
else
    LINK=(-Wl,--whole-archive "$BUILD/libMNN.a" -Wl,--no-whole-archive -pthread -ldl)
fi
"${CXX:-c++}" -std=c++11 -fno-rtti -fno-exceptions \
    -DMNN_BUILD_LLM -DLLM_SUPPORT_VISION -DMNN_OMNI_VISION_TEST_STANDALONE \
    -I"$ROOT/include" -I"$ROOT/test" -I"$ROOT/3rd_party" \
    -I"$ROOT/transformers/llm/engine/include" -I"$ROOT/transformers/llm/engine/src" \
    "$ROOT/test/llm/OmniVisionTest.cpp" "${LINK[@]}" -o "$BUILD/omni_vision_test"
"$BUILD/omni_vision_test"
