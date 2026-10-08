package com.alibaba.mls.api.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.alibaba.mnnllm.android.R as AppR
import com.alibaba.mnnllm.android.download.DownloadForegroundServiceManager
import com.alibaba.mnnllm.android.main.MainActivity

class DownloadForegroundService : Service() {
    private lateinit var notificationManager: NotificationManager
    private var currentDownloadCount = 0
    private var currentModelName: String? = null
    private var foregroundEntered = false
    private var foregroundFailed = false

    override fun onCreate() {
        super.onCreate()
        // Android allows only a few seconds between startForegroundService() and
        // startForeground(), and a stopService() queued in between can prevent onStartCommand()
        // from ever being delivered. Entering the foreground here satisfies the contract
        // whatever happens to the start command afterwards.
        enterForeground()
        Log.d(TAG, "onCreate")
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(AppR.string.download_service_title),
            NotificationManager.IMPORTANCE_LOW
        )
        channel.enableLights(false)
        channel.enableVibration(false)
        channel.setSound(null, null)
        notificationManager.createNotificationChannel(channel)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Extract download count and model name from intent if available
        currentDownloadCount = intent?.getIntExtra(EXTRA_DOWNLOAD_COUNT, 0) ?: 0
        currentModelName = intent?.getStringExtra(EXTRA_MODEL_NAME)
        Log.d(
            TAG,
            "onStartCommand startId=$startId flags=$flags count=$currentDownloadCount modelName=$currentModelName"
        )

        // A null intent and a rejected startForeground() both used to return from here without
        // the service ever being in the foreground, which makes the system kill the process with
        // ForegroundServiceDidNotStartInTimeException.
        val enteredForeground = enterForeground()
        if (intent == null) {
            Log.w(TAG, "onStartCommand received null intent startId=$startId")
        }
        if (intent == null || !enteredForeground) {
            stopSelf(startId)
        }
        return START_NOT_STICKY
    }

    /** Publishes the notification, or reports failure and tears this instance down without retrying. */
    private fun enterForeground(): Boolean {
        if (foregroundFailed) {
            return false
        }
        return try {
            if (!foregroundEntered) {
                notificationManager = getSystemService(NotificationManager::class.java)
                createNotificationChannel()
            }
            val notification = createNotification()
            if (foregroundEntered) {
                notificationManager.notify(SERVICE_ID, notification)
            } else {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    startForeground(SERVICE_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                } else {
                    startForeground(SERVICE_ID, notification)
                }
                foregroundEntered = true
                // Publish only after promotion, so a deferred stop cannot violate the contract.
                instance = this
                Log.d(TAG, "startForeground succeeded")
                DownloadForegroundServiceManager.onServiceReady()
            }
            true
        } catch (e: Exception) {
            foregroundFailed = true
            Log.e(
                TAG,
                "Foreground notification failed count=$currentDownloadCount modelName=$currentModelName",
                e
            )
            // Channel creation, notification construction, promotion and notify failures all
            // invalidate the dispatched state. A later download event may retry after destruction.
            DownloadForegroundServiceManager.onServiceFailed()
            stopSelf()
            false
        }
    }

    private fun createNotification(): Notification {
        val contentTitle = getString(AppR.string.download_service_title)
        val contentText = when {
            currentDownloadCount <= 0 -> getString(AppR.string.downloading_please_wait)
            currentDownloadCount == 1 && currentModelName != null -> {
                getString(AppR.string.downloading_single_model, currentModelName)
            }
            else -> {
                getString(AppR.string.downloading_multiple_models, currentDownloadCount)
            }
        }

        // Create intent to open MainActivity and select ModelMarketFragment
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra(MainActivity.EXTRA_SELECT_TAB, MainActivity.TAB_MODEL_MARKET)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            // Use android system download icon if available, or a fallback
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(pendingIntent)
            .setAutoCancel(false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            
        return builder.build()
    }

    fun updateNotification(downloadCount: Int, modelName: String? = null): Boolean {
        currentDownloadCount = downloadCount
        currentModelName = modelName
        Log.d(TAG, "updateNotification count=$downloadCount modelName=$modelName")
        return enterForeground()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance === this) {
            instance = null
        }
        Log.d(TAG, "onDestroy")
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to remove foreground notification", e)
            DownloadForegroundServiceManager.onServiceFailed()
        } finally {
            // Restart, if needed, only after the old instance can no longer remove the new
            // notification. Failed/unexpected destruction does not automatically restart.
            DownloadForegroundServiceManager.onServiceDestroyed()
        }
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    companion object {
        private const val TAG = "DownloadForegroundSvc"
        private const val CHANNEL_ID = "DownloadServiceChannel"
        private const val SERVICE_ID = 8888
        const val EXTRA_DOWNLOAD_COUNT = "download_count"
        const val EXTRA_MODEL_NAME = "model_name"
        
        @Volatile
        private var instance: DownloadForegroundService? = null

        fun getInstance(): DownloadForegroundService? = instance

        fun updateNotification(downloadCount: Int, modelName: String? = null): Boolean {
            return instance?.updateNotification(downloadCount, modelName) ?: false
        }
    }
}
