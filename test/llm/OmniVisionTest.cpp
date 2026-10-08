// Model-free native vision regression. Build this file with
// MNN_OMNI_VISION_TEST_STANDALONE and link the vision-enabled llm/MNN libraries.
// Explicit main avoids adding a global dynamic test-registration object.
#if defined(MNN_BUILD_LLM) && defined(LLM_SUPPORT_VISION)
#include "MNNTestSuite.h"
#include "omni.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"
#include "../../source/core/KVMeta.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <cmath>
#include <limits>

namespace MNN {
namespace Transformer {
namespace {
class VisionModuleStub : public Express::Module {
public:
    std::vector<VARP> result;
    std::vector<VARP> inputs;
    int calls = 0;
    std::vector<VARP> onForward(const std::vector<VARP>& values) override {
        ++calls;
        inputs = values;
        return result;
    }
};

class VisionTokenizerStub : public Tokenizer {
public:
    std::vector<std::string> chunks;
    VisionTokenizerStub() {
        stop_tokens_.push_back(0);
        set_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}");
    }
    std::string decode(int id) override { return std::string(1, static_cast<char>(id)); }
protected:
    bool load_vocab(std::ifstream&) override { return true; }
    void encode(const std::string& text, std::vector<int>& ids) override {
        chunks.push_back(text);
        for (unsigned char c : text) {
            ids.push_back(c);
        }
    }
};

static VARP image() {
    const uint8_t pixels[12] = {0, 127, 255, 0, 127, 255, 0, 127, 255, 0, 127, 255};
    return _Const(pixels, {2, 2, 3}, NHWC, halide_type_of<uint8_t>());
}

static VARP visionEmbedding(bool gemma, bool initialized = true) {
    auto value = _Input(gemma ? std::vector<int>{1, 280, 4} : std::vector<int>{2, 1, 4}, NCHW);
    if (initialized) {
        auto ptr = value->writeMap<float>();
        std::fill(ptr, ptr + value->getInfo()->size, 39.25f);
        value.fix(VARP::CONSTANT);
    }
    return value;
}

// Real metadata, tiny in-memory graph, no binary fixture and no model files.
static std::shared_ptr<Module> metadataModule(bool vision) {
    auto x = _Input({1, 3, 2, 2}, NCHW);
    x->setName("image");
    auto result = _ReduceSum(x, {}, false) * _Scalar<float>(0) + visionEmbedding(false);
    std::vector<std::string> names = {"image"};
    if (!vision) {
        auto mask = _Input({1}, NCHW);
        auto pos = _Input({1, 1}, NCHW, halide_type_of<int>());
        mask->setName("mask");
        pos->setName("position_ids");
        result = result + _ReduceSum(mask, {}, false) + _Cast<float>(_ReduceSum(pos, {}, false));
        names = {"image", "mask", "position_ids"};
    }
    result->setName("embedding");
    auto buffer = Variable::save({result});
    return std::shared_ptr<Module>(Module::load(names, {"embedding"},
                                               reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()));
}
} // namespace

// Friend of the internal Omni class only; no exported test hooks or public API.
class OmniVisionTest : public Omni {
public:
    int initCalls = 0;
    int embeddingCalls = 0;
    std::shared_ptr<VisionTokenizerStub> tokenizer;
    bool runPrefill = false;
    std::vector<int> prefillIds, prefillHistory;
    std::string cacheBeforePrefill;
    int seqBeforePrefill = -1;
    size_t previousBeforePrefill = 0, removeBeforePrefill = 0;

    OmniVisionTest() : Omni(config()) {
        mContext->status = LlmStatus::RUNNING;
        tokenizer = std::make_shared<VisionTokenizerStub>();
        mTokenizer = tokenizer;
        mModule = metadataModule(false);
    }

    void generate_init(std::ostream* os, const char* end) override {
        ++initCalls;
        Llm::generate_init(os, end);
    }

    VARP embedding(const std::vector<int>& ids) override {
        ++embeddingCalls;
        if (runPrefill) {
            prefillIds = ids;
            prefillHistory = mContext->history_tokens;
            cacheBeforePrefill = mCachedPromptText;
            seqBeforePrefill = mContext->all_seq_len;
            previousBeforePrefill = mMeta->previous;
            removeBeforePrefill = mMeta->remove;
            return _Const(0.0f, {static_cast<int>(ids.size()), 1, 4}, NCHW);
        }
        // Stop at the first text-inference boundary; no weights/decoder required.
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
        return nullptr;
    }

    static std::shared_ptr<LlmConfig> config() {
        auto c = std::make_shared<LlmConfig>();
        c->config_ = ujson::json::parse(R"({"is_visual":true,"hidden_size":4,"image_size":48,
            "use_template":false,"vision_start":7,"vision_end":8,"image_pad":9})");
        return c;
    }

    bool cleanError() const {
        return mContext->status == LlmStatus::INTERNAL_ERROR && mVisionEmbeddings.empty() &&
               mDeepStackEmbeddings.empty() && mAudioEmbeddings.empty() && mPositionIds.mT.empty() && mVisionNum == 0;
    }

    bool outputs() {
        auto stub = std::make_shared<VisionModuleStub>();
        mVisionModule = stub;
        for (bool gemma : {false, true}) {
            // Empty/null, rank, sequence/batch/hidden mismatch, dtype, zero length,
            // unavailable shape and a valid shape whose readMap deliberately fails.
            auto noInfo = _Reshape(_Scalar<float>(1), _Input({3}, NCHW, halide_type_of<int>()));
            MNNTEST_ASSERT(noInfo->getInfo() == nullptr);
            std::vector<std::vector<VARP>> invalid = {
                {}, {nullptr}, {_Scalar<float>(1)}, {_Input({2, 4}, NCHW)},
                {_Input({1, 2, 4}, NCHW)}, {_Input({1, 280, 3}, NCHW)},
                {_Input({2, 1, 4}, NCHW, halide_type_of<int>())},
                {_Input({0, 1, 4}, NCHW)}, {noInfo}, {visionEmbedding(gemma, false)}
            };
            for (const auto& result : invalid) {
                mContext->status = LlmStatus::RUNNING;
                mVisionEmbeddings.push_back(visionEmbedding(false));
                mVisionNum = 1;
                stub->result = result;
                auto ids = gemma ? gemma4VisionProcess(image()) : defaultVisionProcess(image());
                MNNTEST_ASSERT(ids.empty() && cleanError());
            }
            stub->result = {visionEmbedding(gemma)};
            mContext->status = LlmStatus::RUNNING;
            auto ids = gemma ? gemma4VisionProcess(image()) : defaultVisionProcess(image());
            MNNTEST_ASSERT(ids.size() == (gemma ? 3 : 4) && ids.front() == 7 && ids.back() == 8);
            MNNTEST_ASSERT(mVisionEmbeddings.size() == 1);
            auto output = mVisionEmbeddings[0];
            MNNTEST_ASSERT(output->getInfo()->dim == std::vector<int>({gemma ? 1 : 2, 1, 4}));
            MNNTEST_ASSERT(std::fabs(output->readMap<float>()[0] - (gemma ? 1.f : 39.25f)) < 1e-5f);
            mVisionEmbeddings.clear();
        }
        return true;
    }

    bool sizes() {
        auto stub = std::make_shared<VisionModuleStub>();
        mVisionModule = stub;
        stub->result = {visionEmbedding(false)};
        for (const auto& dims : std::vector<std::vector<int>>{{0, 48, 1, 2048}, {-1, 48, 1, 2048},
                {48, 0, 1, 2048}, {48, 48, 0, 2048}, {48, 48, -1, 2048}, {48, 48, 49, 48}}) {
            mVisionHeight = dims[0]; mVisionWidth = dims[1]; mVisionSizeUnit = dims[2]; mVisionMaxSize = dims[3];
            int before = stub->calls;
            MNNTEST_ASSERT(defaultVisionProcess(image()).empty() && cleanError() && stub->calls == before);
        }
        mVisionSizeUnit = 28;
        mVisionMaxSize = 101;
        mVisionHeight = mVisionWidth = std::numeric_limits<int>::max();
        mContext->status = LlmStatus::RUNNING;
        MNNTEST_ASSERT(!defaultVisionProcess(image()).empty());
        MNNTEST_ASSERT(mVisionHeight == 84 && mVisionWidth == 84);
        mVisionHeight = 1; mVisionWidth = 29;
        MNNTEST_ASSERT(!defaultVisionProcess(image()).empty());
        MNNTEST_ASSERT(mVisionHeight == 28 && mVisionWidth == 56);
        mVisionSizeUnit = mVisionMaxSize = std::numeric_limits<int>::max();
        mVisionHeight = mVisionWidth = std::numeric_limits<int>::max();
        MNNTEST_ASSERT(defaultVisionProcess(image()).empty() && cleanError());

        stub->result = {visionEmbedding(true)};
        for (int bad : {0, -1, std::numeric_limits<int>::min()}) {
            mVisionHeight = bad; mVisionWidth = 800;
            MNNTEST_ASSERT(gemma4VisionProcess(image()).empty() && cleanError());
            mVisionHeight = 800; mVisionWidth = bad;
            MNNTEST_ASSERT(gemma4VisionProcess(image()).empty() && cleanError());
        }
        for (int size : {800, std::numeric_limits<int>::max()}) {
            mVisionHeight = mVisionWidth = size;
            mContext->status = LlmStatus::RUNNING;
            MNNTEST_ASSERT(gemma4VisionProcess(image()).size() == 274);
            MNNTEST_ASSERT(mVisionHeight == 768 && mVisionWidth == 816);
            MNNTEST_ASSERT(stub->inputs[0]->getInfo()->dim == std::vector<int>({1, 2520, 768}));
            MNNTEST_ASSERT(stub->inputs[1]->getInfo()->dim == std::vector<int>({1, 2520, 2}));
            auto pos = stub->inputs[1]->readMap<int>();
            MNNTEST_ASSERT(pos && pos[0] == 0 && pos[1] == 0 && pos[2] == 1 && pos[3] == 0);
            MNNTEST_ASSERT(pos[2447 * 2] == 50 && pos[2447 * 2 + 1] == 47 && pos[2448 * 2] == -1);
            auto patch = stub->inputs[0]->readMap<float>();
            MNNTEST_ASSERT(patch && patch[0] == 1.f && patch[2] == 0.f && patch[2448 * 768] == 0.f);
        }
        // Exactly 280 soft tokens: exercise the no-slice path, and a bounded extreme aspect ratio.
        for (int width : {13440, std::numeric_limits<int>::max()}) {
            mVisionHeight = 48; mVisionWidth = width;
            MNNTEST_ASSERT(gemma4VisionProcess(image()).size() == 282);
            MNNTEST_ASSERT(mVisionHeight == 48 && mVisionWidth == 13440);
        }
        return true;
    }

    bool dispatchAndRequests() {
        mVisionModule.reset();
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError());
        mVisionModule = std::make_shared<VisionModuleStub>();
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError()); // missing metadata
        mVisionModule = metadataModule(true);
        MNNTEST_ASSERT(mVisionModule && mModule);
        auto info = const_cast<Module::Info*>(mVisionModule->getInfo());
        auto names = info->inputNames;
        info->inputNames.clear();
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError());
        info->inputNames = names;
        info->inputNames[0] = "input_patches";
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError()); // Gemma4 needs two inputs
        info->inputNames = names;
        auto inputs = info->inputs;
        info->inputNames.push_back("unexpected");
        info->inputs.push_back(info->inputs.front());
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError()); // default needs one input
        info->inputNames = names;
        info->inputs.clear();
        MNNTEST_ASSERT(visionProcess(image()).empty() && cleanError());
        info->inputs = inputs;
        for (VARP bad : std::vector<VARP>{nullptr, _Input({2, 2, 3}, NHWC, halide_type_of<uint8_t>()),
                _Input({0, 2, 3}, NHWC, halide_type_of<uint8_t>()), _Input({2, 2, 3}, NHWC),
                _Input({2, 2}, NHWC, halide_type_of<uint8_t>())}) {
            MNNTEST_ASSERT(visionProcess(bad).empty() && cleanError());
        }
        // An invalid byte extent is checked before attempting to map/read pixels.
        auto oversized = image();
        auto imageInfo = const_cast<Variable::Info*>(oversized->getInfo());
        imageInfo->dim[0] = std::numeric_limits<int>::max();
        MNNTEST_ASSERT(visionProcess(oversized).empty() && cleanError());

        MultimodalPrompt good;
        good.prompt_template = "before<img>one</img>after";
        good.images["one"] = {image(), 48, 48};
        MNNTEST_ASSERT(!tokenizer_encode(good).empty() && mVisionNum == 1);
        for (auto size : std::vector<std::pair<int, int>>{{-1, 48}, {48, -1}, {0, 48}, {48, 0}}) {
            auto invalidPart = good;
            invalidPart.images["one"].width = size.first;
            invalidPart.images["one"].height = size.second;
            MNNTEST_ASSERT(tokenizer_encode(invalidPart).empty() && cleanError());
        }
        MultimodalPrompt bad = good;
        bad.prompt_template = "before<img>one</img>between<img>two</img>never-encode-this";
        bad.images["two"] = {nullptr, 48, 48};
        tokenizer->chunks.clear();
        MNNTEST_ASSERT(tokenizer_encode(bad).empty() && cleanError());
        MNNTEST_ASSERT(std::find(tokenizer->chunks.begin(), tokenizer->chunks.end(), "never-encode-this") == tokenizer->chunks.end());
        int before = initCalls;
        Omni::response(std::vector<int>{}, nullptr, nullptr, 0);
        Omni::response(std::vector<int>{1}, nullptr, nullptr, 0);
        Llm::generate(1); // stepping cannot clear an error either
        MNNTEST_ASSERT(cleanError() && initCalls == before && embeddingCalls == 0);
        MNNTEST_ASSERT(!tokenizer_encode(good).empty() && mVisionNum == 1); // next request recovers

        // Both top-level input APIs must stop before generate_init and recover on the next request.
        Llm::response(bad, nullptr, nullptr, 0);
        MNNTEST_ASSERT(cleanError() && initCalls == before && embeddingCalls == 0);
        Llm::response(good, nullptr, nullptr, 0);
        MNNTEST_ASSERT(initCalls == before + 1 && embeddingCalls == 1);
        MNNTEST_ASSERT(mContext->status == LlmStatus::MAX_TOKENS_FINISHED);
        Llm::response(good, nullptr, nullptr, 0);
        MNNTEST_ASSERT(embeddingCalls == 2); // MAX_TOKENS_FINISHED -> next turn
        mContext->current_token = 0; // stop token: no decoder needed for the continuation gate
        Llm::generate(1);
        MNNTEST_ASSERT(mContext->status == LlmStatus::NORMAL_FINISHED && initCalls == before + 2);

        MNNTEST_ASSERT(tokenizer_encode(std::string("prefix<img>omni-vision-missing-fixture-4641.png</img>suffix")).empty());
        MNNTEST_ASSERT(cleanError());
        const std::string invalid = "prefix<img><hw>0,48</hw>unused</img>never-encode-this";
        tokenizer->chunks.clear();
        MNNTEST_ASSERT(tokenizer_encode(invalid).empty() && cleanError());
        MNNTEST_ASSERT(tokenizer->chunks.size() == 1 && tokenizer->chunks[0] == "prefix");
        int calls = embeddingCalls;
        Llm::response(invalid, nullptr, nullptr, 0);
        MNNTEST_ASSERT(cleanError() && embeddingCalls == calls);
        Llm::response(std::string("text retry"), nullptr, nullptr, 0);
        MNNTEST_ASSERT(embeddingCalls == calls + 1);
        ChatMessages messages = {{"user", invalid}};
        Llm::response(messages, nullptr, nullptr, 0);
        MNNTEST_ASSERT(cleanError() && embeddingCalls == calls + 1);
        Llm::response(ChatMessages{{"user", "chat retry"}}, nullptr, nullptr, 0);
        MNNTEST_ASSERT(embeddingCalls == calls + 2);
        // Cached-chat path must check error before tokenizer_encode("") resets it.
        mConfig->config_["prompt_cache"] = true;
        syncPromptCache(ChatMessages{{"user", "cached"}});
        Llm::response(ChatMessages{{"user", "cached"}, {"user", invalid}}, nullptr, nullptr, 0);
        MNNTEST_ASSERT(cleanError() && embeddingCalls == calls + 2);

        for (LlmStatus status : {LlmStatus::NOT_LOADED, LlmStatus::USER_CANCEL, LlmStatus::TIMEOUT}) {
            mContext->status = status;
            Llm::response(good, nullptr, nullptr, 0);
            MNNTEST_ASSERT(mContext->status == status && embeddingCalls == calls + 2);
        }
        // Run both new regressions even if one fails, just like the top-level groups.
        const bool recovery = cacheRecovery();
        const bool guards = protectedRequestStates();
        return recovery && guards;
    }

    bool cacheRecovery() {
        // Exercise the real chat template/cache/generate/forwardRaw/KVMeta::sync path.
        // Only token vocabulary, embeddings and model outputs are synthetic; no Gemma weights.
        for (bool reuse : {false, true}) {
            for (bool decodeFailure : {false, true}) {
                // Text, multimodal, identical chat, extended chat, raw multimodal tokenizer.
                for (int boundary = 0; boundary < 5; ++boundary) {
                    OmniVisionTest test;
                    test.runPrefill = true;
                    test.mConfig->config_["prompt_cache"] = true;
                    test.mConfig->config_["reuse_kv"] = reuse;
                    auto model = std::make_shared<VisionModuleStub>();
                    model->result = {_Scalar<float>(1)};
                    test.mModulePool[{test.mPrefillKey, false}] = model;
                    test.mModulePool[{1, false}] = model;
                    test.mPositionIdsVarVec.push_back(_Input({1, 1}, NCHW, halide_type_of<int>()));
                    const ChatMessages a = {{"user", "cached-A"}};
                    const ChatMessages ab = {{"user", "cached-A"}, {"user", "+B"}};
                    const std::string textAB = "cached-A+B";
                    const std::vector<int> idsA(a[0].second.begin(), a[0].second.end());
                    const std::vector<int> idsAB(textAB.begin(), textAB.end());
                    test.Llm::response(a, nullptr, nullptr, 0);
                    MNNTEST_ASSERT(test.mCachedPromptText == "cached-A" && test.prefillIds == idsA);
                    MNNTEST_ASSERT(test.mContext->history_tokens == idsA && test.mContext->all_seq_len == idsA.size());
                    MNNTEST_ASSERT(test.mMeta->previous == idsA.size() && test.mMeta->remove == 0);

                    if (!decodeFailure) {
                        model->result.clear(); // Real prefill error before KV metadata sync.
                    }
                    test.Llm::response(ab, nullptr, nullptr, 0);
                    // A normal cached request must still prefill only B, not reset unconditionally.
                    MNNTEST_ASSERT(test.prefillIds == std::vector<int>({'+', 'B'}));
                    MNNTEST_ASSERT(test.prefillHistory == idsAB && test.seqBeforePrefill == idsA.size());
                    MNNTEST_ASSERT(test.removeBeforePrefill == 0 && test.mCachedPromptText == textAB);
                    if (decodeFailure) {
                        MNNTEST_ASSERT(test.mMeta->previous == idsAB.size() &&
                                       test.mContext->all_seq_len == idsAB.size());
                        // A sampled token enters history/output before a failing decode forward.
                        test.mContext->history_tokens.push_back('z');
                        test.mContext->output_tokens.push_back('z');
                        test.mContext->current_token = 'z';
                        test.mContext->gen_seq_len = 1;
                        model->result.clear();
                        MNNTEST_ASSERT(test.Llm::forward(std::vector<int>{'z'}) == nullptr);
                    }
                    MNNTEST_ASSERT(test.mContext->status == LlmStatus::INTERNAL_ERROR);
                    MNNTEST_ASSERT(test.mMeta->previous == (decodeFailure ? idsAB.size() : idsA.size()));
                    MNNTEST_ASSERT(test.mMeta->add == (decodeFailure ? 1 : 2) && test.mMeta->remove == 0);
                    const auto brokenHistory = test.mContext->history_tokens;
                    const auto brokenOutput = test.mContext->output_tokens;
                    const int brokenSeq = test.mContext->all_seq_len;
                    const size_t brokenPrevious = test.mMeta->previous;
                    const size_t brokenAdd = test.mMeta->add;
                    const int initBefore = test.initCalls;
                    const int forwardBefore = model->calls;
                    test.Omni::response(std::vector<int>{}, nullptr, nullptr, 0);
                    test.Omni::response(std::vector<int>{'z'}, nullptr, nullptr, 0);
                    test.Llm::generate(1);
                    MNNTEST_ASSERT(test.mContext->status == LlmStatus::INTERNAL_ERROR);
                    MNNTEST_ASSERT(test.initCalls == initBefore && model->calls == forwardBefore);
                    MNNTEST_ASSERT(test.mContext->history_tokens == brokenHistory &&
                                   test.mContext->output_tokens == brokenOutput);
                    MNNTEST_ASSERT(test.mContext->all_seq_len == brokenSeq && test.mCachedPromptText == textAB);
                    MNNTEST_ASSERT(test.mMeta->previous == brokenPrevious && test.mMeta->add == brokenAdd &&
                                   test.mMeta->remove == 0);

                    model->result = {_Scalar<float>(1)};
                    auto retry = ab;
                    if (boundary == 3) {
                        retry.emplace_back("user", "+C");
                    }
                    const std::string expectedText = boundary == 3 ? textAB + "+C" : textAB;
                    const std::vector<int> expectedIds(expectedText.begin(), expectedText.end());
                    MultimodalPrompt prompt;
                    prompt.prompt_template = expectedText;
                    switch (boundary) {
                        case 0: test.Llm::response(expectedText, nullptr, nullptr, 0); break;
                        case 1: test.Llm::response(prompt, nullptr, nullptr, 0); break;
                        case 2:
                        case 3: test.Llm::response(retry, nullptr, nullptr, 0); break;
                        case 4: {
                            auto ids = test.tokenizer_encode(prompt);
                            MNNTEST_ASSERT(ids == expectedIds && test.mContext->status == LlmStatus::RUNNING);
                            MNNTEST_ASSERT(test.mContext->history_tokens.empty() &&
                                           test.mContext->output_tokens.empty());
                            MNNTEST_ASSERT(test.mContext->all_seq_len == 0 && test.mContext->gen_seq_len == 0);
                            MNNTEST_ASSERT(test.mCachedPromptText.empty() && test.mMeta->previous == brokenPrevious);
                            MNNTEST_ASSERT(test.mMeta->remove == brokenPrevious);
                            test.Omni::response(ids, nullptr, nullptr, 0);
                            break;
                        }
                    }
                    MNNTEST_ASSERT(test.initCalls == initBefore + 1 && model->calls == forwardBefore + 1);
                    MNNTEST_ASSERT(test.prefillIds == expectedIds && test.prefillHistory == expectedIds);
                    MNNTEST_ASSERT(test.cacheBeforePrefill.empty() && test.seqBeforePrefill == 0);
                    MNNTEST_ASSERT(test.previousBeforePrefill == brokenPrevious &&
                                   test.removeBeforePrefill == brokenPrevious);
                    MNNTEST_ASSERT(test.mContext->status == LlmStatus::RUNNING);
                    MNNTEST_ASSERT(test.mContext->history_tokens == expectedIds &&
                                   test.mContext->output_tokens.empty());
                    MNNTEST_ASSERT(test.mContext->all_seq_len == expectedIds.size() && test.mContext->gen_seq_len == 0);
                    MNNTEST_ASSERT(test.mMeta->previous == expectedIds.size() && test.mMeta->remove == 0 &&
                                   test.mMeta->add == 0);
                    MNNTEST_ASSERT(test.mCachedPromptText == ((boundary == 2 || boundary == 3) ? expectedText : ""));
                }
            }
        }
        MNN_PRINT("Cache recovery: passed:20 failed:0 (synthetic model outputs, real request/KV metadata path)\n");
        return true;
    }

    bool protectedRequestStates() {
        for (LlmStatus status : {LlmStatus::NOT_LOADED, LlmStatus::USER_CANCEL, LlmStatus::TIMEOUT}) {
            OmniVisionTest test;
            test.mContext->status = status;
            if (status == LlmStatus::NOT_LOADED) {
                test.mTokenizer.reset();
                test.mModule.reset();
            }
            MultimodalPrompt prompt;
            prompt.prompt_template = "must not tokenize";
            for (int attempt = 0; attempt < 2; ++attempt) {
                test.Omni::response(std::vector<int>{}, nullptr, nullptr, 0);
                MNNTEST_ASSERT(test.mContext->status == status);
                test.Llm::response(prompt.prompt_template, nullptr, nullptr, 0);
                MNNTEST_ASSERT(test.mContext->status == status);
                test.Llm::response(prompt, nullptr, nullptr, 0);
                test.Llm::response(ChatMessages{{"user", prompt.prompt_template}}, nullptr, nullptr, 0);
                MNNTEST_ASSERT(test.tokenizer_encode(prompt).empty());
                test.Omni::response(std::vector<int>{1}, nullptr, nullptr, 0);
                test.Llm::generate(1);
                MNNTEST_ASSERT(test.mContext->status == status && test.tokenizer->chunks.empty());
                MNNTEST_ASSERT(test.initCalls == 0 && test.embeddingCalls == 0);
            }
        }
        OmniVisionTest running;
        running.Omni::response(std::vector<int>{}, nullptr, nullptr, 0);
        MNNTEST_ASSERT(running.cleanError() && running.initCalls == 0);
        return true;
    }
};
} // namespace Transformer
} // namespace MNN

#ifdef MNN_OMNI_VISION_TEST_STANDALONE
int main() {
    using MNN::Transformer::OmniVisionTest;
    int passed = 0;
    int failed = 0;
    auto record = [&](const char* name, bool ok) {
        MNN_PRINT("Omni vision %s: %s\n", name, ok ? "PASS" : "FAIL");
        ok ? ++passed : ++failed;
    };
    { OmniVisionTest test; record("outputs", test.outputs()); }
    { OmniVisionTest test; record("sizes", test.sizes()); }
    { OmniVisionTest test; record("dispatchAndRequests", test.dispatchAndRequests()); }
    MNN_PRINT("Omni vision regression: passed:%d failed:%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
#endif
#endif
