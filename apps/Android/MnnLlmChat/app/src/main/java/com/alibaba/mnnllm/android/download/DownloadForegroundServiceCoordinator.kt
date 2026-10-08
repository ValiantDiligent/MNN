package com.alibaba.mnnllm.android.download

import android.content.Intent
import android.os.Build
import android.util.Log
import com.alibaba.mls.api.download.DownloadForegroundService
import com.alibaba.mnnllm.android.MnnLlmApplication
import java.util.LinkedHashMap

class DownloadForegroundServiceCoordinator(
    private val gateway: Gateway
) {
    interface Gateway {
        fun show(downloadCount: Int, modelName: String?): Boolean
        fun stop(): Boolean

        /**
         * False while a start is still in flight, i.e. before the service has satisfied Android's
         * requirement to call startForeground() after startForegroundService().
         */
        fun isReady(): Boolean
    }

    private val activeDownloads = LinkedHashMap<String, String?>()
    private var lastDispatchedState: NotificationState? = null
    private var phase = Phase.IDLE

    private enum class Phase {
        IDLE, STARTED, STOPPING, FAILED
    }

    fun onDownloadStateChanged(
        modelId: String,
        modelName: String?,
        isDownloading: Boolean
    ) {
        if (isDownloading) {
            activeDownloads[modelId] = modelName ?: activeDownloads[modelId]
        } else {
            activeDownloads.remove(modelId)
        }
        dispatchDesiredState()
    }

    private fun dispatchDesiredState() {
        // Once stop was requested, only remember the latest desired state. The published
        // instance still belongs to the old service until onDestroy has removed its notification.
        if (phase == Phase.STOPPING || phase == Phase.FAILED) {
            return
        }
        val nextState = activeDownloads.toNotificationState()
        if (nextState == null) {
            lastDispatchedState = null
            if (phase == Phase.STARTED && gateway.isReady()) {
                // Set this before calling out: stop may synchronously call back into us.
                phase = Phase.STOPPING
                val stopped = runCatching { gateway.stop() }.getOrDefault(false)
                if (!stopped && phase == Phase.STOPPING) {
                    // A rejected stop has no guaranteed onDestroy. Allow a later event to retry.
                    phase = Phase.STARTED
                }
            }
            // Before ready, leave the start in flight. Stopping now could violate Android's
            // startForegroundService -> startForeground contract and kill the process.
            return
        }

        if (nextState == lastDispatchedState) {
            return
        }

        val previousPhase = phase
        phase = Phase.STARTED
        lastDispatchedState = nextState
        val shown = runCatching { gateway.show(nextState.downloadCount, nextState.modelName) }.getOrDefault(false)
        if (!shown && lastDispatchedState === nextState) {
            // Do not overwrite state changed by a synchronous ready/failure/download callback.
            lastDispatchedState = null
            phase = previousPhase
        }
    }

    /** Flushes a stop that had to wait for the service to reach the foreground. */
    fun onGatewayReady() {
        if (phase == Phase.STARTED) {
            dispatchDesiredState()
        }
    }

    /** A failed service is tearing down, but its destruction must not trigger an automatic retry. */
    fun onGatewayFailed() {
        lastDispatchedState = null
        phase = Phase.FAILED
    }

    /** Called only after the old instance and its notification have been cleared. */
    fun onGatewayDestroyed() {
        val restartAfterStop = phase == Phase.STOPPING
        phase = Phase.IDLE
        lastDispatchedState = null
        if (restartAfterStop) {
            dispatchDesiredState()
        }
    }

    private fun LinkedHashMap<String, String?>.toNotificationState(): NotificationState? {
        if (isEmpty()) {
            return null
        }
        return NotificationState(
            downloadCount = size,
            modelName = values.firstOrNull()
        )
    }

    private data class NotificationState(
        val downloadCount: Int,
        val modelName: String?
    )
}

object DownloadForegroundServiceManager {
    private const val TAG = "DownloadFgServiceMgr"

    private val coordinator = DownloadForegroundServiceCoordinator(
        object : DownloadForegroundServiceCoordinator.Gateway {
            override fun show(downloadCount: Int, modelName: String?): Boolean {
                return runCatching {
                    val service = DownloadForegroundService.getInstance()
                    if (service != null) {
                        Log.d(
                            TAG,
                            "Updating download foreground service count=$downloadCount modelName=$modelName"
                        )
                        service.updateNotification(downloadCount, modelName)
                    } else {
                        val context = MnnLlmApplication.getAppContext()
                        val intent = Intent(context, DownloadForegroundService::class.java).apply {
                            putExtra(DownloadForegroundService.EXTRA_DOWNLOAD_COUNT, downloadCount)
                            putExtra(DownloadForegroundService.EXTRA_MODEL_NAME, modelName)
                        }
                        Log.d(
                            TAG,
                            "Starting download foreground service count=$downloadCount modelName=$modelName"
                        )
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            context.startForegroundService(intent) != null
                        } else {
                            context.startService(intent) != null
                        }
                    }
                }.getOrElse { error ->
                    Log.e(
                        TAG,
                        "Failed to show download foreground service count=$downloadCount modelName=$modelName",
                        error
                    )
                    false
                }
            }

            override fun stop(): Boolean {
                return runCatching {
                    val context = MnnLlmApplication.getAppContext()
                    Log.d(TAG, "Stopping download foreground service")
                    context.stopService(Intent(context, DownloadForegroundService::class.java))
                }.getOrElse { error ->
                    Log.e(TAG, "Failed to stop download foreground service", error)
                    false
                }
            }

            // The instance is published only once startForeground() has succeeded, so this is
            // exactly the "safe to tear down" condition the coordinator needs.
            override fun isReady(): Boolean = DownloadForegroundService.getInstance() != null
        }
    )

    @Synchronized
    fun onDownloadStateChanged(
        modelId: String,
        modelName: String?,
        isDownloading: Boolean
    ) {
        coordinator.onDownloadStateChanged(modelId, modelName, isDownloading)
    }

    /** Called by the service once it has entered the foreground and can safely be stopped. */
    @Synchronized
    fun onServiceReady() {
        coordinator.onGatewayReady()
    }

    @Synchronized
    fun onServiceFailed() {
        coordinator.onGatewayFailed()
    }

    @Synchronized
    fun onServiceDestroyed() {
        coordinator.onGatewayDestroyed()
    }
}
