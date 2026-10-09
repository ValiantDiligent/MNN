package com.alibaba.mnnllm.api.openai.network.queue

import kotlinx.coroutines.*
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class RequestQueueManagerTest {
    @Test
    fun cancelledWaitingRequestIsSkippedAndMetricsNeverReorderFifo() = runTest {
        val queue = RequestQueueManager(StandardTestDispatcher(testScheduler))
        val release = CompletableDeferred<Unit>()
        val order = mutableListOf<Int>()
        var concurrent = 0
        var maxConcurrent = 0
        val requests = (1..4).map { index ->
            launch {
                queue.submitRequest("$index", "trace", task = {
                    concurrent++
                    maxConcurrent = maxOf(maxConcurrent, concurrent)
                    try {
                        order.add(index)
                        if (index == 1) release.await()
                    } finally {
                        concurrent--
                    }
                })
            }.also { runCurrent() }
        }
        try {
            assertEquals(listOf(1), order)
            assertEquals(3, queue.getQueueSize())
            requests[1].cancelAndJoin()
            assertEquals(2, queue.getQueueSize())
            repeat(10_000) {
                assertEquals(2, queue.getQueueSize())
                val stats = queue.getQueueStats()
                assertEquals(2L, stats.pendingRequests)
                assertTrue(stats.isProcessing)
            }
            release.complete(Unit)
            advanceUntilIdle()
            requests.forEach { it.join() }
            assertEquals(listOf(1, 3, 4), order)
            assertEquals(1, maxConcurrent)
            val stats = queue.getQueueStats()
            assertEquals(4L, stats.totalRequests)
            assertEquals(3L, stats.completedRequests)
            assertEquals(0L, stats.failedRequests)
            assertEquals(0, queue.getQueueSize())
            assertFalse(stats.isProcessing)
        } finally {
            release.complete(Unit)
            queue.shutdown()
        }
    }

    @Test
    fun failureAndThrowingObserverDoNotStallQueueOrCancelCaller() = runTest {
        val queue = RequestQueueManager(StandardTestDispatcher(testScheduler))
        var errors = 0
        var laterRan = false
        try {
            val failed = async {
                try {
                    queue.submitRequest("failed", "trace", task = { throw IllegalStateException("native failure") },
                        onError = { errors++; throw IllegalStateException("observer failure") })
                    fail("Failure must reach submitter")
                } catch (e: IllegalStateException) {
                    assertEquals("native failure", e.message)
                    assertTrue(currentCoroutineContext().isActive)
                }
            }
            val later = launch {
                queue.submitRequest("later", "trace", task = { laterRan = true },
                    onComplete = { throw IllegalStateException("observer failure") })
            }
            advanceUntilIdle()
            failed.await()
            later.join()
            assertEquals(1, errors)
            assertTrue(laterRan)
            assertEquals(1L, queue.getQueueStats().failedRequests)
            assertEquals(1L, queue.getQueueStats().completedRequests)
            assertFalse(queue.getQueueStats().isProcessing)
        } finally {
            queue.shutdown()
        }
    }

    @Test
    fun clearAndShutdownResolvePendingWaiters() = runTest {
        val queue = RequestQueueManager(StandardTestDispatcher(testScheduler))
        val gate = CompletableDeferred<Unit>()
        var waitingRan = false
        val running = launch { queue.submitRequest("running", "trace", task = { gate.await() }) }
        runCurrent()
        val cleared = launch { queue.submitRequest("cleared", "trace", task = { waitingRan = true }) }
        runCurrent()
        queue.clearQueue()
        runCurrent()
        assertTrue(cleared.isCancelled)
        assertEquals(0, queue.getQueueSize())
        assertTrue(queue.getQueueStats().isProcessing)
        val shutdown = launch { queue.submitRequest("shutdown", "trace", task = { waitingRan = true }) }
        runCurrent()
        queue.shutdown()
        advanceUntilIdle()
        assertTrue(running.isCancelled)
        assertTrue(shutdown.isCancelled)
        assertFalse(waitingRan)
        assertEquals(0, queue.getQueueSize())
        assertFalse(queue.getQueueStats().isProcessing)
    }
}
