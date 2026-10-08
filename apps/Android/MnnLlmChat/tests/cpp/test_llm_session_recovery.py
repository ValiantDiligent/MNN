#!/usr/bin/env python3
"""Host-only regression for Android response recovery; no model, NDK or JVM needed.

Run: python3 apps/Android/MnnLlmChat/tests/cpp/test_llm_session_recovery.py
Requires Python 3 and a C++17 compiler (CXX, default c++). Production function
bodies and LlmStatus are extracted verbatim, not mirrored in the test. Use
--source with a pre-fix llm_session.cpp to verify the regression goes red.
Build outputs go to a unique directory under --build-dir (default: TMPDIR).
"""
import argparse
import ctypes
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CPP = HERE.parents[1] / "app/src/main/cpp"


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def extract_session(text):
    return (
        between(text, "namespace mls {", "// Process multimodal prompt")
        + between(text, "const MNN::Transformer::LlmContext * LlmSession::Response(",
                  "std::string LlmSession::getDebugInfo")
        + between(text, "const MNN::Transformer::LlmContext * LlmSession::ResponseWithHistory(",
                  "void LlmSession::clearHistory")
        + "\n} // namespace mls\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CPP / "llm_session.cpp")
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()
    build = Path(tempfile.mkdtemp(prefix="llm-session-recovery-", dir=args.build_dir))
    print("Build directory:", build, flush=True)
    header = (ROOT / "transformers/llm/engine/include/llm/llm.hpp").read_text()
    status = re.search(r"enum class LlmStatus \{.*?\};", header, re.S)
    if status is None:
        raise ValueError("LlmStatus declaration not found")
    (build / "llm_status.inc").write_text(status.group() + "\n")
    (build / "llm_session_under_test.inc").write_text(extract_session(args.source.read_text()))
    binary = build / "llm_session_recovery_test.so"
    command = shlex.split(os.environ.get("CXX", "c++"))
    env = dict(os.environ, TMPDIR=str(build), TMP=str(build), TEMP=str(build))
    shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(command + ["-std=c++17", "-fno-rtti", "-fno-exceptions", shared, "-fPIC",
                              "-I" + str(build), "-I" + str(CPP),
                              str(HERE / "llm_session_recovery_test.cpp"), "-o", str(binary)],
                   env=env, check=True)
    library = ctypes.CDLL(str(binary))
    entry = library.runLlmSessionRecoveryTests
    entry.argtypes = []
    entry.restype = ctypes.c_int
    if entry() != 0:
        raise RuntimeError("LlmSession recovery regression failed")


if __name__ == "__main__":
    main()
