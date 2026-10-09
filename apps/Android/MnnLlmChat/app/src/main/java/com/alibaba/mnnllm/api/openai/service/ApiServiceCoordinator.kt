package com.alibaba.mnnllm.api.openai.service

import android.content.Context
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ApiNotificationManager
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.network.application.OpenAIApplication
import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import com.alibaba.mnnllm.api.openai.runtime.SessionLeaseToken
import com.alibaba.mnnllm.android.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import timber.log.Timber

/** Coordinates the API foreground service, its network instance and its own runtime lease. */
class ApiServiceCoordinator(private val context: Context) {
    private val TAG = this::class.java.simpleName
    private val stateLock = Any()
    private val networkServiceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val runtime = ServiceLocator.getLlmRuntimeController()
    private var notificationManager: ApiNotificationManager? = null
    @Volatile private var application: OpenAIApplication? = null
    private var leaseToken: SessionLeaseToken? = null
    @Volatile private var bootstrapCount: Int = 0
    @Volatile private var cleanupRequested = false
    private var cleanupJob: Job? = null

    @Volatile private var _isInitialized = false
    val isInitialized: Boolean get() = _isInitialized

    @Volatile private var _isServerRunning = false
    val isServerRunning: Boolean get() = _isServerRunning

    companion object {
        // Notifications and Ktor events are process-global: finish the old network stop before a
        // replacement publishes events. This mutex is only waited on IO, never by UI cleanup.
        private val serverMutex = Mutex()
        private var serverOwner: ApiServiceCoordinator? = null // guarded by serverMutex
    }

    /** * initializecoordinator*/
    fun initialize(): Boolean = synchronized(stateLock) {
        if (cleanupRequested) return false
        if (_isInitialized) return true
        runCatching {
            notificationManager = ApiNotificationManager(context)
            _isInitialized = true
            true
        }.getOrElse { e ->
            Timber.tag(TAG).e(e, "Failed to initialize coordinator")
            false
        }
    }

    /** Load/start and shutdown serialize on IO, including across old/new Service instances. */
    suspend fun startServer(modelId: String? = null): Boolean = withContext(Dispatchers.IO) {
        serverMutex.withLock {
            if (!_isInitialized || cleanupRequested) return@withLock false
            try {
                serverOwner?.takeIf { it !== this@ApiServiceCoordinator }?.stopServerLocked()
                serverOwner = this@ApiServiceCoordinator
                if (!resolveRuntimeLeaseForStart(modelId)) {
                    stopServerLocked()
                    notificationManager?.updateNotification(
                        context.getString(R.string.api_service_not_started),
                        context.getString(R.string.no_active_session)
                    )
                    return@withLock false
                }
                // onDestroy can record close while ensureSession is still in native load.
                if (cleanupRequested) {
                    stopServerLocked()
                    return@withLock false
                }
                if (application != null && _isServerRunning) return@withLock true

                val app = OpenAIApplication(networkServiceScope, context)
                application = app // Retain partial startup for failure rollback too.
                app.start()
                // OpenAIApplication.start catches startup exceptions itself.
                check(app.isRunning()) { "API network did not start" }
                bootstrapCount += 1
                notificationManager?.updateNotification(
                    context.getString(R.string.api_service_running), "", app.getPort()
                )
                _isServerRunning = true
                true
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "Failed to start server")
                withContext(NonCancellable) { stopServerLocked() }
                if (!cleanupRequested) {
                    notificationManager?.updateNotification(
                        context.getString(R.string.api_service_start_failed),
                        context.getString(R.string.api_service_error, e.message)
                    )
                }
                false
            }
        }
    }

    private fun resolveRuntimeLeaseForStart(modelId: String?): Boolean {
        val activeModelId = runtime.getActiveModelId()
        val hasLoadedActiveSession = ServiceLocator.getChatSessionProvider().hasActiveSession()
        if (ApiRuntimeSessionStartPolicy.shouldReuseLoadedSession(modelId, activeModelId, hasLoadedActiveSession)) {
            if (rememberLease(runtime.acquireActiveSession())) return true
        }
        return rememberLease(
            if (modelId.isNullOrBlank()) runtime.acquireActiveSession() else runtime.ensureSession(modelId)
        )
    }

    private fun rememberLease(result: EnsureSessionResult): Boolean {
        if (!result.success || result.leaseToken == null) {
            Timber.tag(TAG).w("Failed to acquire API runtime lease: %s", result.reason)
            return false
        }
        // Acquire before releasing our previous lease, so reuse never drops the session to zero.
        val previous = leaseToken
        leaseToken = result.leaseToken
        if (previous !== result.leaseToken) releaseLease(previous)
        return true
    }

    private fun releaseLease(token: SessionLeaseToken?) {
        if (token == null) return
        runCatching { runtime.releaseSession(token) }.onFailure {
            Timber.tag(TAG).e(it, "Failed to release API runtime lease")
        }
    }

    /** Must hold serverMutex. Release only after this network instance has stopped accepting work. */
    private suspend fun stopServerLocked() {
        val appToStop = application
        val tokenToRelease = leaseToken
        application = null
        leaseToken = null
        _isServerRunning = false
        try {
            appToStop?.stopInternal()
        } finally {
            releaseLease(tokenToRelease)
            // Late old cleanup must not reset shared state or cancel the new owner's notification.
            if (serverOwner === this) {
                serverOwner = null
                ServerEventManager.getInstance().resetRuntimeState()
                notificationManager?.cancelNotification()
            }
        }
    }

    fun stopServer(): Job = networkServiceScope.launch {
        withContext(NonCancellable) {
            serverMutex.withLock { stopServerLocked() }
        }
    }

    /**
     * updatenotificationcontent*/
    fun updateNotification(title: String, content: String, port: Int = 8080) {
        notificationManager?.updateNotification(title, content, port)
    }

    /** * getnotificationobject（forforegroundservice）*/
    fun getNotification(
        title: String = context.getString(R.string.api_service_running),
        content: String = context.getString(R.string.api_service_port, 8080),
        port: Int = 8080
    ) = notificationManager?.buildNotification(title, content, port)

    /** * getserviceport*/
    fun getServerPort(): Int? = application?.getPort()

    fun getBootstrapCount(): Int {
        synchronized(stateLock) {
            return bootstrapCount
        }
    }

    /** * checkservicewhetherrunning*/
    fun checkServerStatus(): Boolean = application?.isRunning() ?: false

    /** * checkservicewhetherready*/
    fun checkServerReady(): Boolean = application?.isReady() ?: false

    /** * getservicestate*/
    fun getServerState() = application?.getServerState()

    /** * getserviceinfo*/
    fun getServerInfo() = application?.getServerInfo()

    /** Main-thread close is intent-only, even while startServer is inside native model load. */
    fun cleanup(): Job = synchronized(stateLock) {
        cleanupJob?.let { return it }
        cleanupRequested = true
        _isInitialized = false
        networkServiceScope.launch {
            try {
                withContext(NonCancellable) {
                    serverMutex.withLock {
                        stopServerLocked()
                        notificationManager = null
                    }
                }
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "API cleanup failed")
            } finally {
                networkServiceScope.cancel()
            }
        }.also { cleanupJob = it }
    }
}
