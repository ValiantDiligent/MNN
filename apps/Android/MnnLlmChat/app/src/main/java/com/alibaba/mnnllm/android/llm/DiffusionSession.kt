// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide

class DiffusionSession(
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    private var savedHistory: List<ChatDataItem>? = null
): ChatSession{
    override var supportOmni: Boolean = false
    private val lifecycle = NativeSessionLifecycle(::releaseNative) { provide().removeSession(this) }
    override val debugInfo: String = ""

    override fun load() {
        lifecycle.load {
            initNative(configPath, DiffusionLoadConfigResolver.buildExtraConfigJson(modelId, configPath))
        }
    }

    override fun generate(
        prompt: String,
        params: Map<String, Any>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            if (!lifecycle.isLoaded()) {
                return hashMapOf("error" to true, "message" to "Native diffusion session not initialized")
            }
            return lifecycle.generate { ptr ->
                val output = params["output"] as String
                val iterNum = params["iterNum"] as Int
                val randomSeed = params["randomSeed"] as Int
                submitDiffusionNative(ptr, prompt, output, iterNum, randomSeed, progressListener)
                    ?: hashMapOf("error" to true, "message" to "Native diffusion returned null")
            }
        }
    }

    private external fun submitDiffusionNative(
        instanceId: Long,
        input: String,
        outputPath: String,
        iterNum: Int,
        randomSeed: Int,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>?

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun reset(): String {
        return generateNewSessionId()
    }

    override fun release() {
        lifecycle.release()
    }

    override fun setKeepHistory(keepHistory: Boolean) {
        // Diffusion models don't use keepHistory for generation, but we still store it for UI
        // The history is managed by the UI layer, not the native layer
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        // Diffusion models don't support audio output
    }

    override fun getHistory(): List<ChatDataItem>? {
        Log.d(TAG, "getHistory: returning ${savedHistory?.size ?: 0} items")
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
        Log.d(TAG, "setHistory: setting ${history?.size ?: 0} items")
        savedHistory = history
    }

    override fun updateThinking(thinking: Boolean) {
    }


    private external fun initNative(
        configPath: String,
        extraConfig:String
    ): Long

    private external fun resetNative(instanceId: Long)

    private external fun releaseNative(instanceId: Long)

    companion object {
        const val TAG: String = "DiffusionSession"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}
