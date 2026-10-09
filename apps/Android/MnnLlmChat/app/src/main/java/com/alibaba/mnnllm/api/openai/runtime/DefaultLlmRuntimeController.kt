package com.alibaba.mnnllm.api.openai.runtime

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import timber.log.Timber

object DefaultLlmRuntimeController : LlmRuntimeController {
    private val lock = Any()
    private var activeModelId: String? = null
    private val ownership = RuntimeSessionOwnership<LlmSession>()
    private val ownedSession: LlmSession? get() = ownership.session
    private var activeUseAppConfig: Boolean = false

    override fun ensureSession(
        modelId: String,
        forceReload: Boolean,
        useAppConfig: Boolean,
        configPath: String?,
        sessionId: String?,
        historyList: List<ChatDataItem>?,
        deferLoad: Boolean
    ): EnsureSessionResult {
        synchronized(lock) {
            val currentSession = ownedSession
            if (
                currentSession != null &&
                RuntimeSessionReusePolicy.shouldReuse(
                    forceReload = forceReload,
                    activeModelId = activeModelId,
                    requestedModelId = modelId,
                    isSessionLoaded = currentSession.isModelLoaded(),
                    activeUseAppConfig = activeUseAppConfig,
                    requestedUseAppConfig = useAppConfig
                )
            ) {
                return EnsureSessionResult(
                    success = true,
                    session = currentSession,
                    modelId = modelId,
                    leaseToken = ownership.acquire(currentSession)
                )
            }

            if (currentSession != null) {
                releaseSessionLocked()
            }

            val resolvedConfigPath = configPath ?: if (useAppConfig) {
                ModelUtils.getConfigPathForModel(modelId)
            } else {
                ModelConfig.getDefaultConfigFile(modelId)
            } ?: return EnsureSessionResult(
                success = false,
                modelId = modelId,
                reason = "MODEL_CONFIG_NOT_FOUND"
            )

            val modelName = ModelUtils.getModelName(modelId) ?: modelId
            val resolvedSessionId = sessionId ?: "service_runtime_${System.currentTimeMillis()}"

            return runCatching {
                val chatSession = ChatService.provide().createSession(
                    modelId = modelId,
                    modelName = modelName,
                    sessionIdParam = resolvedSessionId,
                    historyList = historyList,
                    configPath = resolvedConfigPath,
                    useNewConfig = !useAppConfig,
                    useCustomConfig = useAppConfig
                )

                val llmSession = chatSession as? LlmSession
                if (llmSession == null) {
                    chatSession.release()
                    return EnsureSessionResult(false, modelId = modelId, reason = "SESSION_NOT_LLM")
                }

                // Register before load so an init exception also closes the created session.
                val leaseToken = ownership.acquire(llmSession)
                activeModelId = modelId
                activeUseAppConfig = useAppConfig
                llmSession.setKeepHistory(true)
                if (!deferLoad) {
                    llmSession.load()
                }
                EnsureSessionResult(
                    success = true,
                    session = llmSession,
                    modelId = modelId,
                    leaseToken = leaseToken
                )
            }.getOrElse { error ->
                Timber.w(error, "ensureSession failed for modelId=%s", modelId)
                releaseSessionLocked()
                EnsureSessionResult(
                    success = false,
                    modelId = modelId,
                    reason = "SESSION_INIT_FAILED"
                )
            }
        }
    }

    override fun acquireActiveSession(): EnsureSessionResult {
        synchronized(lock) {
            val session = getActiveSession()
                ?: return EnsureSessionResult(false, reason = "NO_ACTIVE_SESSION")
            return EnsureSessionResult(
                success = true,
                session = session,
                modelId = activeModelId,
                leaseToken = ownership.acquire(session)
            )
        }
    }

    override fun getActiveSession(): LlmSession? {
        synchronized(lock) {
            val session = ownedSession ?: return null
            return session.takeIf {
                RuntimeSessionReusePolicy.shouldExposeActiveSession(it.isModelLoaded())
            }
        }
    }

    override fun getActiveModelId(): String? {
        synchronized(lock) {
            return activeModelId
        }
    }

    override fun getThinkingEnabled(): Boolean? {
        synchronized(lock) {
            val modelId = activeModelId ?: return null
            val config = ModelConfig.loadConfig(modelId) ?: return null
            return config.jinja?.context?.enableThinking != false
        }
    }

    override fun setThinkingEnabled(enabled: Boolean): Boolean {
        synchronized(lock) {
            val session = getActiveSession() ?: return false
            return runCatching {
                session.updateThinking(enabled)
                true
            }.getOrElse { error ->
                Timber.w(error, "setThinkingEnabled failed enabled=%s", enabled)
                false
            }
        }
    }

    override fun releaseSession() {
        synchronized(lock) {
            releaseSessionLocked()
        }
    }

    override fun releaseSession(leaseToken: SessionLeaseToken): Boolean {
        synchronized(lock) {
            return releaseSessionLocked(leaseToken)
        }
    }

    private fun releaseSessionLocked(leaseToken: SessionLeaseToken? = null): Boolean {
        return ownership.release(leaseToken) { session ->
            activeModelId = null
            activeUseAppConfig = false
            runCatching {
                session.release()
            }.onFailure { error ->
                Timber.w(error, "releaseSession failed")
            }
        }
    }
}
