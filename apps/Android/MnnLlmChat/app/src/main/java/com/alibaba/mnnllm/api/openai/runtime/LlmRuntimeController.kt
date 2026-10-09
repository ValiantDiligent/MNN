package com.alibaba.mnnllm.api.openai.runtime

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.LlmSession

data class EnsureSessionResult(
    val success: Boolean,
    val session: LlmSession? = null,
    val modelId: String? = null,
    val reason: String? = null,
    val leaseToken: SessionLeaseToken? = null
)

interface LlmRuntimeController {
    /**
     * Each successful call creates an independent lease. UI/API callers must retain and release it,
     * including reuse and failure after deferred load. Switching sessions force-releases the old one.
     * The debug dumpapp controller is administrative: ensure/run intentionally pin until its explicit
     * release/reload command (including isolated-run restore), which invalidates ALL leases.
     */
    fun ensureSession(
        modelId: String,
        forceReload: Boolean = false,
        useAppConfig: Boolean = false,
        configPath: String? = null,
        sessionId: String? = null,
        historyList: List<ChatDataItem>? = null,
        deferLoad: Boolean = false
    ): EnsureSessionResult
    /** Acquires the loaded session without changing its app/API configuration. */
    fun acquireActiveSession(): EnsureSessionResult
    fun getActiveSession(): LlmSession?
    fun getActiveModelId(): String?
    fun getThinkingEnabled(): Boolean?
    fun setThinkingEnabled(enabled: Boolean): Boolean
    /** Administrative stop; asynchronous owners must use the token overload. */
    fun releaseSession()
    fun releaseSession(leaseToken: SessionLeaseToken): Boolean
}
