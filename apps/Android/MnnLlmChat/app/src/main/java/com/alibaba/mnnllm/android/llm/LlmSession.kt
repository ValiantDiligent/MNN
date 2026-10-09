// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm;

import android.util.Log
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.getExtraConfigFile
import com.google.gson.Gson
import timber.log.Timber
import java.io.File
import android.util.Pair
import com.alibaba.mnnllm.android.utils.MmapUtils
import android.content.Context
import android.app.ActivityManager
import com.alibaba.mnnllm.android.modelsettings.Jinja
import com.alibaba.mnnllm.android.modelsettings.JinjaContext
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.loadConfig
import com.alibaba.mnnllm.android.utils.FileSplitter
import com.alibaba.mnnllm.android.qnn.QnnModule
class LlmSession (
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    var savedHistory: List<ChatDataItem>?,
    var backendType: String? = null,
    private val useCustomConfig: Boolean = true
): ChatSession{
    override var supportOmni: Boolean = false
    private val lifecycle = NativeSessionLifecycle(::releaseNative) { provide().removeSession(this) }

    @Volatile private var keepHistory = false

    private var isQnn = false

    override fun getHistory(): List<ChatDataItem>?{
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
    }

    override fun load() {
        lifecycle.load {
            Log.d(TAG, "MNN_DEBUG load begin modelId: $modelId backend: $backendType")
            isQnn = ModelTypeUtils.isQnnModel(modelId)
            checkAndMergeSplitFiles()
            val historyStringList = savedHistory?.takeIf { it.isNotEmpty() }?.mapNotNull { it.text }
            val llmConfig = if (useCustomConfig) {
                ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
            } else {
                ModelConfig.loadDefaultConfig(configPath)!!
            }
            var rootCacheDir: String? = ""
            if (llmConfig.useMmap == true) {
                rootCacheDir = MmapUtils.getMmapDir(modelId)
                File(rootCacheDir).mkdirs()
            }
            val configMap = HashMap<String, Any>().apply {
                put("is_r1", ModelTypeUtils.isR1Model(modelId))
                put("mmap_dir", rootCacheDir ?: "")
                put("keep_history", keepHistory)
            }
            if (backendType != null) {
                llmConfig.backendType = backendType
            }
            if (isQnn) {
                llmConfig.visualModel = "visual_qnn_${QnnModule.modelMiddleName()}.mnn"
            }
            initNative(configPath, historyStringList, Gson().toJson(llmConfig), Gson().toJson(configMap))
        }
    }

    /**
     * Check if the model is successfully loaded and ready for inference
     */
    fun isModelLoaded(): Boolean = lifecycle.isLoaded()
    
    /**
     * Check and merge split files for the current model
     */
    private fun checkAndMergeSplitFiles() {
        try {
            val configFile = File(configPath)
            val modelDir = configFile.parentFile
            
            if (modelDir != null && modelDir.exists()) {
                Log.d(TAG, "Checking for split files in model directory: ${modelDir.absolutePath}")
                
                if (FileSplitter.needsMerging(modelDir)) {
                    Log.d(TAG, "Found split files that need merging in ${modelDir.absolutePath}")
                    val success = FileSplitter.mergeAllSplitFiles(modelDir)
                    if (success) {
                        Log.d(TAG, "Successfully merged split files for model: $modelId")
                    } else {
                        Log.w(TAG, "Failed to merge some split files for model: $modelId")
                    }
                } else {
                    Log.d(TAG, "No split files found for model: $modelId")
                }
            } else {
                Log.w(TAG, "Model directory not found: ${modelDir?.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking/merging split files for model: $modelId", e)
        }
    }

    fun getConfig(): ModelConfig? {
        return ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))
    }

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun generate(prompt: String,
                          params: Map<String, Any>,
                          progressListener: GenerateProgressListener): HashMap<String, Any> {
        Log.d(TAG, "start generate prompt: $prompt")
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG generate intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            Log.d(TAG, "MNN_DEBUG submit$prompt")
            return lifecycle.generate { ptr ->
                submitNative(ptr, prompt, keepHistory, progressListener)
            }
        }
    }

    override fun reset(): String {
        lifecycle.mutate { ptr -> resetNative(ptr) }
        return generateNewSessionId()
    }

    override fun release() {
        lifecycle.release(waitForIdle = true)
    }

    private external fun initNative(
            configPath: String?,
            history: List<String>?,
            mergedConfigStr: String?,
            configJsonStr: String?
    ): Long

    private external fun submitNative(
            instanceId: Long,
            input: String,
            keepHistory: Boolean,
            listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long)

    private external fun setWavformCallbackNative(
            instanceId: Long,
            listener: AudioDataListener?
    ): Boolean

    override fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        lifecycle.mutate { ptr -> updateEnableAudioOutputNative(ptr, enable) }
    }

    // Busy native reads use the existing empty fallback instead of racing native string writes.
    override val debugInfo
        get() = lifecycle.withIdleHandle { ptr -> if (ptr != 0L) getDebugInfoNative(ptr) + "\n" else "" }

    fun setAudioDataListener(listener: AudioDataListener?) {
        lifecycle.mutate { ptr -> setWavformCallbackNative(ptr, listener) }
    }

    fun updateMaxNewTokens(maxNewTokens: Int) {
        lifecycle.mutate { ptr -> updateMaxNewTokensNative(ptr, maxNewTokens) }
    }

    fun updateSystemPrompt(systemPrompt: String) {
        lifecycle.mutate { ptr -> updateSystemPromptNative(ptr, systemPrompt) }
    }

    override fun updateThinking(thinking: Boolean) {
        val loadedConfig = loadConfig(modelId)
        loadedConfig?.let {
            loadedConfig.jinja = Jinja(context = JinjaContext(enableThinking = thinking))
            ModelConfig.saveConfig(getExtraConfigFile(modelId), loadedConfig)
            updateConfig(Gson().toJson(loadedConfig))
        }
    }

    fun updateConfig(configJson: String) {
        Log.d(TAG, "updateConfig: $configJson")
        lifecycle.mutate { ptr -> updateConfigNative(ptr, configJson) }
    }

    private external fun updateEnableAudioOutputNative(llmPtr: Long, enable: Boolean)


    private external fun updateMaxNewTokensNative(llmPtr: Long, maxNewTokens: Int)

    private external fun updateSystemPromptNative(llmPtr: Long, systemPrompt: String)

    private external fun updateAssistantPromptNative(llmPtr: Long, assistantPrompt: String)

    private external fun updateConfigNative(llmPtr: Long, configJson: String)


    companion object {
        const val TAG: String = "LlmSession"
        var mockLatex: Boolean = false
        var mockLatexContent: String? = null

        init {
            System.loadLibrary("mnnllmapp")
        }
    }



    //New: public method supporting complete history messages
    fun submitFullHistory(
        history: List<Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG submitFullHistory intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            return lifecycle.generate { ptr ->
                Timber.d("MNN_DEBUG submitFullHistory with ${history.size} messages")
                val androidHistory = history.map { android.util.Pair(it.first, it.second) }
                submitFullHistoryNative(ptr, androidHistory, progressListener)
            }
        }
    }

    private fun submitMockLatexHistory(progressListener: GenerateProgressListener): HashMap<String, Any> {
        val mockText = mockLatexContent ?: "Here is a math formula:\n\n\$E=mc^2$\n\nAnd a block formula:\n\n\$\$a^2 + b^2 = c^2\$\$\n\nEnd of mock."
        Thread {
            try {
                // Simulate streaming delay
                var index = 0
                val chunkSize = 3
                while (index < mockText.length) {
                    Thread.sleep(50)
                    val endIndex = Math.min(index + chunkSize, mockText.length)
                    val chunk = mockText.substring(index, endIndex)
                    if (progressListener.onProgress(chunk)) {
                        break
                    }
                    index = endIndex
                }
                progressListener.onProgress(null) // notify completion
            } catch (e: Exception) {
                Timber.e(e, "Mock generation failed")
            }
        }.start()
        val map = HashMap<String, Any>()
        map["success"] = true
        map["prompt_len"] = 10L
        map["decode_len"] = mockText.length.toLong()
        map["prefill_time"] = 100000L
        map["decode_time"] = 2000000L
        return map
    }
    private external fun submitFullHistoryNative(
        nativePtr: Long,
        history: List<android.util.Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>

    fun modelId(): String {
        //Create temporary variable to avoid modifying original modelId
        return modelId

    }

    fun getSystemPrompt(): String? {
        return lifecycle.withIdleHandle { ptr -> if (ptr != 0L) getSystemPromptNative(ptr) else null }
    }

    private external fun getSystemPromptNative(llmPtr: Long): String?

    private external fun dumpConfigNative(llmPtr: Long): String

    fun dumpConfig(): String {
        return lifecycle.withIdleHandle { ptr -> if (ptr != 0L) dumpConfigNative(ptr) else "{}" }
    }

    // Helper function to get current memory usage in MB
    private fun getCurrentMemoryUsageMB(context: Context): Long {
        val runtime = Runtime.getRuntime()
        val usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedMemoryBytes / (1024 * 1024) // Convert to MB
    }
    
    // Helper function to get total memory info
    private fun getMemoryInfo(context: Context): Pair<Long, Long> {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val runtime = Runtime.getRuntime()
        val usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val availMemoryMB = memoryInfo.availMem / (1024 * 1024)
        
        return Pair(usedMemoryMB, availMemoryMB)
    }

    // Official benchmark functionality following llm_bench.cpp approach
    fun runBenchmark(
        context: Context,
        commandParams: com.alibaba.mnnllm.android.benchmark.CommandParameters,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult {
        // Use coroutine instead of Thread for better lifecycle management
        return try {
            // Run the actual benchmark in C++ following llm_bench.cpp structure
            lifecycle.generate { ptr ->
                runBenchmarkNative(
                    ptr,
                    commandParams.backend,
                    commandParams.threads,
                    commandParams.useMmap,
                    commandParams.power,
                    commandParams.precision,
                    commandParams.memory,
                    commandParams.dynamicOption,
                    commandParams.nPrompt,
                    commandParams.nGenerate,
                    commandParams.nRepeat,
                    commandParams.kvCache == "true",
                    testInstance,
                    callback
                )
            }
        } catch (e: Exception) {
            com.alibaba.mnnllm.android.benchmark.BenchmarkResult(
                testInstance = testInstance,
                success = false,
                errorMessage = "benchmark failed: ${e.message}"
            )
        }
    }

    // C++ implementation following llm_bench.cpp approach
    private external fun runBenchmarkNative(
        nativePtr: Long,
        backend: Int,
        threads: Int,
        useMmap: Boolean,
        power: Int,
        precision: Int,
        memory: Int,
        dynamicOption: Int,
        nPrompt: Int,
        nGenerate: Int,
        nRepeat: Int,
        kvCache: Boolean,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult

}