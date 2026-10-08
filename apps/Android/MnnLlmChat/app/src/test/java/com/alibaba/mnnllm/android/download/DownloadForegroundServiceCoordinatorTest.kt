package com.alibaba.mnnllm.android.download

import org.junit.Assert.assertEquals
import org.junit.Test

class DownloadForegroundServiceCoordinatorTest {

    private val gateway = FakeGateway()
    private val coordinator = DownloadForegroundServiceCoordinator(gateway).also {
        gateway.coordinator = it
    }

    @Test
    fun downloadingStateStartsForegroundServiceOnce() {
        coordinator.onDownloadStateChanged(
            modelId = "model-a",
            modelName = "Model A",
            isDownloading = true
        )

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A")), gateway.events)
    }

    @Test
    fun repeatedDownloadingStateDoesNotRestartForegroundService() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A")), gateway.events)
    }

    @Test
    fun additionalDownloadUpdatesForegroundNotificationCount() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Show(2, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun endingLastDownloadStopsForegroundService() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Stop
            ),
            gateway.events
        )
    }

    @Test
    fun endingPrimaryDownloadPromotesNextActiveModelName() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Show(2, "Model A"),
                FakeGateway.Event.Show(1, "Model B")
            ),
            gateway.events
        )
    }

    @Test
    fun stopWaitsWhileTheServiceHasNotReachedTheForeground() {
        // Stopping inside that window is what kills the process with
        // ForegroundServiceDidNotStartInTimeException.
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A")), gateway.events)
    }

    @Test
    fun deferredStopRunsOnceTheServiceReachesTheForeground() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.ready = true
        coordinator.onGatewayReady()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Stop
            ),
            gateway.events
        )
    }

    @Test
    fun deferredStopIsDroppedWhenADownloadResumesFirst() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.ready = true
        coordinator.onGatewayReady()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Show(1, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun resumedDownloadDoesNotUpdateTheStoppingInstance() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(
            listOf(FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop),
            gateway.events
        )
    }

    @Test
    fun stopAfterResumeRestartsOnlyAfterDestructionWithLatestDownloads() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.becomeReady() // A duplicate ready must not show on the stopping instance.
        gateway.destroy()
        gateway.becomeReady()
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Ready, FakeGateway.Event.Destroy,
                FakeGateway.Event.Show(1, "Model B"), FakeGateway.Event.Ready
            ),
            gateway.events
        )
    }

    @Test
    fun downloadThatEndsAgainWhileStoppingDoesNotRestart() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", false)
        gateway.destroy()

        assertEquals(
            listOf(FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop, FakeGateway.Event.Destroy),
            gateway.events
        )
    }

    @Test
    fun duplicateReadyStopsOnlyOnce() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.becomeReady()
        gateway.becomeReady()
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.destroy()
        gateway.becomeReady()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Ready,
                FakeGateway.Event.Stop, FakeGateway.Event.Ready,
                FakeGateway.Event.Destroy, FakeGateway.Event.Ready
            ),
            gateway.events
        )
    }

    @Test
    fun rejectedShowAllowsRetryWithIdenticalState() {
        gateway.showResult = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.showResult = true
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(List(2) { FakeGateway.Event.Show(1, "Model A") }, gateway.events)
    }

    @Test
    fun throwingShowAllowsRetryWithIdenticalState() {
        gateway.showThrows = true
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.showThrows = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(List(2) { FakeGateway.Event.Show(1, "Model A") }, gateway.events)
    }

    @Test
    fun rejectedInitialShowDoesNotLeaveADeferredStop() {
        gateway.ready = false
        gateway.showResult = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.becomeReady()

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Ready), gateway.events)
    }

    @Test
    fun asynchronousPromotionFailureAllowsSameStateRetryAfterDestruction() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.fail()
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.becomeReady() // A stale ready must not revive a failed instance.
        gateway.destroy()
        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Failed,
                FakeGateway.Event.Ready, FakeGateway.Event.Destroy
            ),
            gateway.events
        )
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        assertEquals(FakeGateway.Event.Show(1, "Model A"), gateway.events.last())
        assertEquals(5, gateway.events.size)
    }

    @Test
    fun failureClearsDeferredStopWithoutAutomaticallyRestarting() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.fail()
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()
        gateway.becomeReady()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Failed,
                FakeGateway.Event.Destroy, FakeGateway.Event.Ready
            ),
            gateway.events
        )
    }

    @Test
    fun failedRestartDoesNotLoopOnDestroy() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy() // The one restart justified by a normal stop.
        gateway.fail()
        gateway.destroy()
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Destroy, FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Failed, FakeGateway.Event.Destroy, FakeGateway.Event.Destroy
            ),
            gateway.events
        )
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        assertEquals(8, gateway.events.size)
        assertEquals(FakeGateway.Event.Show(1, "Model A"), gateway.events.last())
    }

    @Test
    fun unexpectedDestroyDoesNotRestartButAllowsSameStateRetry() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()
        gateway.becomeReady()
        assertEquals(
            listOf(FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Destroy, FakeGateway.Event.Ready),
            gateway.events
        )
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        assertEquals(4, gateway.events.size)
    }

    @Test
    fun failureWhileStoppingDisarmsAutomaticRestart() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.fail()
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Failed, FakeGateway.Event.Destroy
            ),
            gateway.events
        )
    }

    @Test
    fun rejectedStopCanBeRetriedWithoutWaitingForDestroy() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.stopResult = false
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.stopResult = true
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop, FakeGateway.Event.Stop,
                FakeGateway.Event.Destroy, FakeGateway.Event.Show(1, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun throwingDeferredStopDoesNotPreventResumeOrLaterStop() {
        gateway.ready = false
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.stopThrows = true
        gateway.becomeReady()
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.stopThrows = false
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Ready, FakeGateway.Event.Stop,
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop, FakeGateway.Event.Destroy
            ),
            gateway.events
        )
    }

    @Test
    fun rejectedStopDoesNotCauseAutomaticRestartOnUnexpectedDestroy() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.stopResult = false
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Destroy
            ),
            gateway.events
        )
    }

    @Test
    fun readyAndDownloadCallbacksInsideShowObserveAlreadyDispatchedState() {
        gateway.ready = false
        gateway.onShow = {
            gateway.becomeReady()
            coordinator.onDownloadStateChanged("model-a", "Model A", true)
            coordinator.onDownloadStateChanged("model-a", "Model A", false)
        }
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Ready,
                FakeGateway.Event.Stop, FakeGateway.Event.Destroy
            ),
            gateway.events
        )
    }

    @Test
    fun synchronousFailureInsideShowIsNotOverwrittenBySuccessfulDispatch() {
        gateway.onShow = { gateway.fail() }
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.destroy()
        gateway.onShow = null
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Failed,
                FakeGateway.Event.Destroy, FakeGateway.Event.Show(1, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun failedOuterShowDoesNotEraseNewerReentrantDispatch() {
        gateway.showResult = false
        gateway.onShow = {
            gateway.onShow = null
            gateway.showResult = true
            coordinator.onDownloadStateChanged("model-b", "Model B", true)
        }
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Show(2, "Model A")),
            gateway.events
        )
    }

    @Test
    fun synchronousResumeAndDestroyInsideStopStartsReplacementOnce() {
        gateway.onStop = {
            coordinator.onDownloadStateChanged("model-b", "Model B", true)
            gateway.destroy()
        }
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        gateway.becomeReady()
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Destroy, FakeGateway.Event.Show(1, "Model B"), FakeGateway.Event.Ready
            ),
            gateway.events
        )
    }

    @Test
    fun rejectedNotificationUpdateAllowsSameStateRetry() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.showResult = false
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        gateway.showResult = true
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Show(2, "Model A"),
                FakeGateway.Event.Show(2, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun rejectedNotificationUpdateStillAllowsStoppingTheExistingService() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.showResult = false
        coordinator.onDownloadStateChanged("model-a", "Renamed A", true)
        coordinator.onDownloadStateChanged("model-a", "Renamed A", false)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Show(1, "Renamed A"),
                FakeGateway.Event.Stop
            ),
            gateway.events
        )
    }

    @Test
    fun notificationFailureCallbackBlocksOldInstanceAndAllowsSameStateRetryAfterDestroy() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        gateway.onShow = { gateway.fail() }
        gateway.showResult = false
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        gateway.destroy()
        gateway.onShow = null
        gateway.showResult = true
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Show(2, "Model A"),
                FakeGateway.Event.Failed, FakeGateway.Event.Destroy, FakeGateway.Event.Show(2, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun stopExceptionAfterSynchronousDestroyDoesNotEraseReplacementState() {
        gateway.onStop = {
            coordinator.onDownloadStateChanged("model-b", "Model B", true)
            gateway.destroy()
            throw IllegalStateException("stop failed after callback")
        }
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        gateway.becomeReady()

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"), FakeGateway.Event.Stop,
                FakeGateway.Event.Destroy, FakeGateway.Event.Show(1, "Model B"), FakeGateway.Event.Ready
            ),
            gateway.events
        )
    }

    private class FakeGateway : DownloadForegroundServiceCoordinator.Gateway {
        lateinit var coordinator: DownloadForegroundServiceCoordinator
        val events = mutableListOf<Event>()
        var ready = true
        var showResult = true
        var stopResult = true
        var showThrows = false
        var stopThrows = false
        var onShow: (() -> Unit)? = null
        var onStop: (() -> Unit)? = null

        override fun show(downloadCount: Int, modelName: String?): Boolean {
            events += Event.Show(downloadCount, modelName)
            val result = showResult
            if (showThrows) throw IllegalStateException("show rejected")
            onShow?.invoke()
            return result
        }

        override fun stop(): Boolean {
            events += Event.Stop
            val result = stopResult
            if (stopThrows) throw IllegalStateException("stop rejected")
            onStop?.invoke()
            return result
        }

        fun becomeReady() {
            ready = true
            events += Event.Ready
            coordinator.onGatewayReady()
        }

        fun fail() {
            events += Event.Failed
            coordinator.onGatewayFailed()
        }

        fun destroy() {
            ready = false
            events += Event.Destroy
            coordinator.onGatewayDestroyed()
        }

        override fun isReady(): Boolean = ready

        sealed interface Event {
            data class Show(val downloadCount: Int, val modelName: String?) : Event
            data object Stop : Event
            data object Ready : Event
            data object Failed : Event
            data object Destroy : Event
        }
    }
}
