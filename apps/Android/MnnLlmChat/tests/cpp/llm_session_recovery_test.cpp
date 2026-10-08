// Run with test_llm_session_recovery.py. Only engine/processor dependencies are fake;
// the response helpers, both entrypoints, stream code and status enum are production code.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#define MNN_DEBUG(...) ((void)0)
#define MNN_ERROR(...) ((void)0)
#define CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " << #c << '\n'; std::exit(1); } } while (0)

namespace MNN { namespace Transformer {
#include "llm_status.inc"
struct LlmContext {
    LlmStatus status = LlmStatus::RUNNING;
    int prompt_len = 11, gen_seq_len = 2;
    int64_t vision_us = 31, audio_us = 41, prefill_us = 51, decode_us = 61;
};
struct MultimodalPrompt { std::string prompt_template; std::map<std::string, int> images; };
struct Step {
    LlmStatus status;
    std::string text;
    bool eop = false;
    bool null_context = false;
};
class Llm {
public:
    LlmContext context;
    Step prefill{LlmStatus::RUNNING, ""}, audio{LlmStatus::NORMAL_FINISHED, ""};
    std::vector<Step> steps;
    bool missing_context = false, reset_changes_status = false;
    bool stale_kv = false, stale_prompt = false;
    std::vector<int> history_tokens;
    std::string last_prompt;
    int decode_calls = 0, response_calls = 0, waveform_calls = 0, sync_calls = 0, erase_calls = 0;
    int reset_calls = 0, dirty_resets = 0, blocked_responses = 0;
    size_t erased_begin = 0;
    std::ostream* output = nullptr;
    const LlmContext* getContext() { return missing_context ? nullptr : &context; }
    bool failed() const {
        return missing_context || context.status == LlmStatus::INTERNAL_ERROR ||
               context.status == LlmStatus::TIMEOUT || context.status == LlmStatus::NOT_LOADED;
    }
    void pollute() {
        stale_kv = stale_prompt = true;
        history_tokens.push_back(42);
    }
    void reset() {
        reset_calls++;
        if (stale_kv && stale_prompt && !history_tokens.empty()) dirty_resets++;
        stale_kv = stale_prompt = false;
        history_tokens.clear();
        // Also test a runtime that changes status: the wrapper must preserve the error for JNI.
        if (reset_changes_status) context.status = LlmStatus::RUNNING;
    }
    void apply(const Step& step) {
        if (!step.text.empty()) *output << step.text;
        if (step.eop) *output << "<eop>";
        context.status = step.status;
        missing_context = step.null_context;
        if (failed()) pollute();
    }
    void respond(const std::string& prompt, std::ostream* os, int n) {
        CHECK(n == 0);
        response_calls++;
        // Unlike the old fake, never silently turn a sticky failure into success.
        if (failed()) { blocked_responses++; return; }
        CHECK(!stale_kv && !stale_prompt && history_tokens.empty());
        output = os;
        last_prompt = prompt;
        apply(prefill);
    }
    void response(const std::vector<std::pair<std::string, std::string>>& history,
                  std::ostream* os, const char*, int n) {
        std::string prompt;
        for (const auto& item : history) prompt += item.second;
        respond(prompt, os, n);
    }
    void response(const MultimodalPrompt& prompt, std::ostream* os, const char*, int n) {
        respond(prompt.prompt_template, os, n);
    }
    void generate(int n) {
        CHECK(n == 1);
        int index = decode_calls++;
        if (index < static_cast<int>(steps.size())) apply(steps[index]);
    }
    void generateWavform() { waveform_calls++; apply(audio); }
    void syncPromptCache(const std::vector<std::pair<std::string, std::string>>&) { sync_calls++; }
    size_t getCurrentHistory() { return 17; }
    void eraseHistory(size_t begin, size_t end) { CHECK(end == 0); erased_begin = begin; erase_calls++; }
};
}}
using MNN::Transformer::Llm;
using MNN::Transformer::LlmContext;
using MNN::Transformer::LlmStatus;
using MNN::Transformer::Step;
namespace mls {
using PromptItem = std::pair<std::string, std::string>;
struct PromptProcessingResult {
    bool has_multimodal = false;
    MNN::Transformer::MultimodalPrompt multimodal_prompt;
    std::string error_message;
};
PromptProcessingResult processMultimodalPrompt(const std::string& text) {
    PromptProcessingResult result;
    result.has_multimodal = text.find("<img>") != std::string::npos;
    result.multimodal_prompt.prompt_template = text;
    return result;
}
class LlmSession {
public:
    Llm* llm_ = nullptr;
    bool model_loaded_ = true, keep_history_ = true, stop_requested_ = false, generate_text_end_ = false;
    bool is_r1_ = false, enable_audio_output_ = true;
    int max_new_tokens_ = 4;
    std::string response_string_for_debug, prompt_string_for_debug;
    std::vector<PromptItem> history_{{"system", "system-original"}, {"user", "old-user"}, {"assistant", "old-answer"}};
    const LlmContext* Response(const std::string&, const std::function<bool(const std::string&, bool)>&);
    const LlmContext* ResponseWithHistory(const std::vector<PromptItem>&,
                                        const std::function<bool(const std::string&, bool)>&);
};
}

// Production code (generated by the runner, never a copy of the implementation).
#include "utf8_stream_processor.hpp"
#include "llm_stream_buffer.hpp"
#include "llm_session_under_test.inc"

// Recovery cases (also reused by the optional real-JVM validation).
const LlmContext* run(mls::LlmSession& session, bool full, const std::string& text,
                     const std::function<bool(const std::string&, bool)>& progress) {
    if (!full) return session.Response(text, progress);
    auto history = session.history_;
    history.emplace_back("user", text);
    auto* result = session.ResponseWithHistory(history, progress);
    CHECK(history.back() == mls::PromptItem("user", text));
    return result;
}
int checkRecovery() {
    int count = 0;
    for (bool full : {false, true}) for (bool keep : {false, true})
    for (bool r1 : {false, true}) for (bool image : {false, true})
    for (int phase = 0; phase < 3; ++phase) for (bool eop : {false, true})
    for (bool same : {false, true}) for (bool reset_status : {false, true})
    for (int state = 0; state < 4; ++state) {
        Llm engine;
        mls::LlmSession session;
        session.llm_ = &engine;
        session.keep_history_ = keep;
        session.is_r1_ = r1;
        engine.reset_changes_status = reset_status;
        const auto original = session.history_;
        const LlmStatus errors[] = {LlmStatus::INTERNAL_ERROR, LlmStatus::TIMEOUT, LlmStatus::NOT_LOADED};
        Step failure{state == 3 ? LlmStatus::INTERNAL_ERROR : errors[state], "", eop, state == 3};
        if (phase == 0) engine.prefill = failure;
        if (phase == 1) engine.steps = {{LlmStatus::MAX_TOKENS_FINISHED, "A", true},
                                      {LlmStatus::MAX_TOKENS_FINISHED, "B", true}, failure};
        if (phase == 2) {
            engine.steps = {{LlmStatus::NORMAL_FINISHED, "AB", eop}};
            engine.audio = failure;
        }
        int endings = 0;
        std::string output;
        const std::string prompt = image ? "<img>failed.png</img>" : "failed-user";
        auto* result = run(session, full, prompt, [&](const std::string& text, bool end) {
            if (end) endings++; else output += text;
            return false;
        });
        CHECK(engine.reset_calls == 1 && engine.dirty_resets == 1);
        CHECK(!engine.stale_kv && !engine.stale_prompt && engine.history_tokens.empty());
        CHECK(state == 3 ? result == nullptr : result && result->status == failure.status);
        CHECK(session.history_ == original);
        CHECK(engine.sync_calls == 0 && engine.erase_calls == 0);
        CHECK(engine.decode_calls == (phase == 0 ? 0 : phase == 1 ? 3 : eop ? 1 : 4));
        CHECK(engine.waveform_calls == (phase == 2 ? 1 : 0));
        CHECK(output == (phase == 0 ? "" : "AB"));
        // Audio happens after text EOS; it may already have delivered the text completion callback.
        CHECK(endings == (phase == 2 && eop ? 1 : 0));

        engine.prefill = {LlmStatus::RUNNING, ""};
        engine.steps = {{LlmStatus::NORMAL_FINISHED, "retry-ok", true}};
        engine.audio = {LlmStatus::NORMAL_FINISHED, ""};
        engine.decode_calls = 0;
        output.clear();
        endings = 0;
        // Do not manually clear status, null context or poisoned cache to make retries succeed.
        const std::string retry = same ? prompt : "different-user";
        result = run(session, full, retry, [&](const std::string& text, bool end) {
            if (end) endings++; else output += text;
            return false;
        });
        if (state >= 2) {
            CHECK(state == 3 ? result == nullptr : result && result->status == LlmStatus::NOT_LOADED);
            CHECK(output.empty() && engine.decode_calls == 0 && engine.sync_calls == 0);
            CHECK(session.history_ == original);
        } else {
            CHECK(result && result->status == LlmStatus::NORMAL_FINISHED);
            CHECK(output == "retry-ok" && endings == 1 && engine.decode_calls == 1);
            CHECK(engine.sync_calls == 1 && engine.blocked_responses == 0);
            CHECK(engine.last_prompt.find(retry) != std::string::npos);
            if (!same) CHECK(engine.last_prompt.find(prompt) == std::string::npos);
            CHECK(engine.last_prompt.find("system-original") != std::string::npos);
            CHECK((engine.last_prompt.find("old-answer") != std::string::npos) == (full || keep));
            const std::string answer = r1 ? "retry-ok<|end_of_sentence|>" : "retry-ok";
            CHECK(full ? session.history_ == original : session.history_.back().second == answer);
        }
        count++;
    }
    return count;
}
int checkPromptBoundary() {
    int count = 0;
    for (bool full : {false, true}) for (bool loaded : {false, true})
    for (auto status : {LlmStatus::INTERNAL_ERROR, LlmStatus::TIMEOUT, LlmStatus::NOT_LOADED}) {
        Llm engine;
        mls::LlmSession session;
        session.llm_ = &engine;
        session.model_loaded_ = loaded;
        const auto original = session.history_;
        engine.context.status = status;
        engine.pollute();  // A failure left by another caller, not this wrapper.
        engine.steps = {{LlmStatus::NORMAL_FINISHED, "ok", true}};
        auto* result = run(session, full, "fresh", {});
        const bool recover = loaded && status != LlmStatus::NOT_LOADED;
        CHECK(result && result->status == (recover ? LlmStatus::NORMAL_FINISHED : status));
        CHECK(engine.blocked_responses == (recover ? 0 : 1));
        CHECK(engine.reset_calls == 1 && engine.dirty_resets == 1);
        CHECK(!engine.stale_kv && !engine.stale_prompt && engine.history_tokens.empty());
        CHECK(engine.decode_calls == (recover ? 1 : 0));
        if (!recover) CHECK(session.history_ == original);
        count++;
    }
    for (bool full : {false, true}) {
        mls::LlmSession session;
        const auto original = session.history_;
        CHECK(run(session, full, "hello", {}) == nullptr);
        Llm engine;
        engine.missing_context = true;
        session.llm_ = &engine;
        CHECK(run(session, full, "hello", {}) == nullptr);
        CHECK(session.history_ == original && engine.response_calls == 0 && engine.reset_calls == 0);
        count += 2;
    }
    return count;
}

int checkStepping() {
    int count = 0;
    for (bool full : {false, true}) for (int kind = 0; kind < 5; ++kind) {
        Llm engine;
        mls::LlmSession session;
        session.llm_ = &engine;
        int endings = 0;
        std::string output;
        if (kind == 0) engine.steps = {{LlmStatus::MAX_TOKENS_FINISHED, "A", true},
                                      {LlmStatus::NORMAL_FINISHED, "B", true}};
        if (kind == 1) engine.steps = std::vector<Step>(4, {LlmStatus::MAX_TOKENS_FINISHED, "A", true});
        if (kind == 2) engine.steps = std::vector<Step>(4, {LlmStatus::RUNNING, "A", false});
        if (kind == 3) engine.steps = {{LlmStatus::MAX_TOKENS_FINISHED, "A", true}};
        if (kind == 4) engine.steps = {{LlmStatus::NORMAL_FINISHED, "", false},
                                      {LlmStatus::NORMAL_FINISHED, "B", true}};
        auto* result = run(session, full, "normal-user", [&](const std::string& text, bool end) {
            if (end) endings++; else output += text;
            return kind == 3;
        });
        CHECK(result != nullptr && engine.reset_calls == 0);
        CHECK(engine.decode_calls == ((kind == 0 || kind == 4) ? 2 : kind == 3 ? 1 : 4));
        CHECK(engine.waveform_calls == (kind == 3 ? 0 : 1));
        CHECK(engine.sync_calls == (full && kind == 3 ? 0 : 1));
        CHECK(engine.erase_calls == (full && kind == 3 ? 1 : 0));
        CHECK(engine.erase_calls == 0 || engine.erased_begin == 17);
        CHECK(endings == ((kind == 2 || kind == 3) ? 0 : 1));
        CHECK(output == (kind == 0 ? "AB" : kind == 4 ? "B" : kind == 3 ? "A" : "AAAA"));
        count++;
    }
    return count;
}

// Host entry point.
extern "C" int runLlmSessionRecoveryTests() {
    int recovery = checkRecovery();
    int boundary = checkPromptBoundary();
    int stepping = checkStepping();
    std::cout << "PASS: " << recovery << " failure/retry cases, " << boundary << " prompt-boundary cases, "
              << stepping << " unchanged stepping cases\n";
    return 0;
}
