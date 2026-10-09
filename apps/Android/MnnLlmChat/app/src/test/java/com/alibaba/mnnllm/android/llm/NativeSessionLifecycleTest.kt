package com.alibaba.mnnllm.android.llm

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/** Tests the production helper used by LlmSession (both submit paths), DiffusionSession and SanaSession. */
class NativeSessionLifecycleTest {
    private class Fixture {
        val releases = AtomicInteger()
        val closed = AtomicInteger()
        val lifecycle = NativeSessionLifecycle(
            { ptr -> assertEquals(42L, ptr); releases.incrementAndGet() },
            { closed.incrementAndGet() }
        )
    }

    private class Worker(block: () -> Unit) {
        val failure = AtomicReference<Throwable?>()
        val thread = Thread {
            try { block() } catch (error: Throwable) { failure.set(error) }
        }.apply { isDaemon = true; start() }
        fun join() {
            thread.join(2000)
            assertFalse("Worker failed to terminate", thread.isAlive)
            failure.get()?.let { throw AssertionError("Worker failed", it) }
        }
    }

    private fun CountDownLatch.awaitReady() {
        assertTrue("Timed out waiting for test gate", await(2, TimeUnit.SECONDS))
    }

    private fun awaitWaiting(worker: Worker) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
        while (worker.thread.state != Thread.State.WAITING && worker.thread.isAlive && System.nanoTime() < deadline) {
            Thread.yield()
        }
        assertEquals("Release must park until native work exits", Thread.State.WAITING, worker.thread.state)
    }

    private fun expectFailure(block: () -> Unit) {
        try {
            block()
            fail("Expected failure")
        } catch (_: IllegalStateException) {
        }
    }

    @Test(timeout = 5000)
    fun loadExceptionWakesBlockingLlmRelease() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val loader = Worker {
            expectFailure {
                f.lifecycle.load {
                    entered.countDown()
                    finish.awaitReady()
                    error("JNI load failed")
                }
            }
        }
        entered.awaitReady()
        val closer = Worker { f.lifecycle.release(waitForIdle = true) }
        try {
            awaitWaiting(closer)
        } finally {
            finish.countDown()
        }
        loader.join()
        closer.join()
        assertFalse(f.lifecycle.isLoaded())
        assertEquals(0, f.releases.get())
        assertEquals(1, f.closed.get())
        expectFailure { f.lifecycle.load { fail("Closed object must not enter JNI"); 42L } }
    }

    @Test(timeout = 5000)
    fun allBlockingReleasersWakeAfterLoadFailure() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val loader = Worker {
            expectFailure { f.lifecycle.load { entered.countDown(); finish.awaitReady(); error("load") } }
        }
        entered.awaitReady()
        val closers = List(3) { Worker { f.lifecycle.release(waitForIdle = true) } }
        try {
            closers.forEach { awaitWaiting(it) }
        } finally {
            finish.countDown()
        }
        loader.join()
        closers.forEach { it.join() }
        assertEquals(0, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test(timeout = 5000)
    fun failedLoadCanBeReleasedWithoutWaitingForever() {
        val f = Fixture()
        expectFailure { f.lifecycle.load { error("JNI exception") } }
        Worker { f.lifecycle.release(waitForIdle = true) }.join()
        f.lifecycle.release(waitForIdle = true)
        assertEquals(1, f.closed.get())
    }

    @Test(timeout = 5000)
    fun blockingReleaseRejectsSpuriousWakeupsAndRestoresInterruptOnlyOnExit() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val loader = Worker { f.lifecycle.load { entered.countDown(); finish.awaitReady(); 42L } }
        entered.awaitReady()
        val closer = Worker {
            f.lifecycle.release(waitForIdle = true)
            assertTrue(Thread.currentThread().isInterrupted)
        }
        try {
            awaitWaiting(closer)
            closer.thread.interrupt()
            // Test-only reflection: no JNI bypass or monitor hook is exposed by production.
            val monitor = NativeSessionLifecycle::class.java.getDeclaredField("monitor").let {
                it.isAccessible = true
                it.get(f.lifecycle) as Object
            }
            synchronized(monitor) { monitor.notifyAll() }
            awaitWaiting(closer)
            assertEquals(0, f.releases.get())
            assertEquals(0, f.closed.get())
        } finally {
            finish.countDown()
        }
        loader.join()
        closer.join()
        assertFalse(f.lifecycle.isLoaded())
        assertEquals(1, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test(timeout = 5000)
    fun imageReleaseDuringLoadDisposesLateHandleExactlyOnce() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val loader = Worker { f.lifecycle.load { entered.countDown(); finish.awaitReady(); 42L } }
        entered.awaitReady()
        try {
            Worker { f.lifecycle.release(); f.lifecycle.release() }.join()
            assertFalse(f.lifecycle.isLoaded())
            assertEquals(0, f.releases.get())
        } finally {
            finish.countDown()
        }
        loader.join()
        f.lifecycle.release()
        assertFalse(f.lifecycle.isLoaded())
        assertEquals(1, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test(timeout = 5000)
    fun imageLoadExceptionAfterCloseStillRemovesRegistration() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val loader = Worker {
            expectFailure { f.lifecycle.load { entered.countDown(); finish.awaitReady(); error("load") } }
        }
        entered.awaitReady()
        f.lifecycle.release()
        finish.countDown()
        loader.join()
        assertEquals(0, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test(timeout = 5000)
    fun generationExceptionsWakeReleaseInBothReleaseModes() {
        for (blocking in listOf(false, true)) {
            val f = Fixture()
            f.lifecycle.load { 42L }
            val entered = CountDownLatch(1)
            val finish = CountDownLatch(1)
            val generator = Worker {
                expectFailure {
                    f.lifecycle.generate { ptr ->
                        assertEquals(42L, ptr)
                        entered.countDown()
                        finish.awaitReady()
                        error("submit or progress listener failed")
                    }
                }
            }
            entered.awaitReady()
            val closer = Worker { f.lifecycle.release(waitForIdle = blocking) }
            try {
                if (blocking) awaitWaiting(closer) else closer.join()
                assertEquals(0, f.releases.get())
            } finally {
                finish.countDown()
            }
            generator.join()
            closer.join()
            assertEquals(1, f.releases.get())
            assertEquals(1, f.closed.get())
            expectFailure { f.lifecycle.generate { fail("Cannot submit after close") } }
        }
    }

    @Test(timeout = 5000)
    fun generationFailureDoesNotLeaveBusyFlagSet() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        expectFailure { f.lifecycle.generate { error("submit") } }
        assertEquals(42L, f.lifecycle.generate { it })
        Worker { f.lifecycle.release(waitForIdle = true) }.join()
        assertEquals(1, f.releases.get())
    }

    @Test(timeout = 5000)
    fun synchronousProgressCallbackCanRequestRelease() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        Worker {
            f.lifecycle.generate {
                f.lifecycle.release(waitForIdle = true)
                assertEquals(0, f.releases.get())
            }
        }.join()
        assertEquals(1, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test
    fun closeBeforeLoadIsTerminalAndIdempotent() {
        val f = Fixture()
        f.lifecycle.release()
        f.lifecycle.release(waitForIdle = true)
        expectFailure { f.lifecycle.load { fail("init must not run"); 42L } }
        assertEquals(0, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test
    fun zeroHandleIsFailureAndStillClosable() {
        val f = Fixture()
        expectFailure { f.lifecycle.load { 0L } }
        assertFalse(f.lifecycle.isLoaded())
        f.lifecycle.release()
        assertEquals(0, f.releases.get())
        assertEquals(1, f.closed.get())
    }

    @Test
    fun duplicateLoadDoesNotReplaceLiveHandle() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        f.lifecycle.load { fail("Already loaded"); 43L }
        f.lifecycle.release()
        assertEquals(1, f.releases.get())
    }

    @Test(timeout = 5000)
    fun settersDuringLongGenerationReturnWithoutEnteringNativeAndDrainInOrder() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val calls = mutableListOf<String>()
        val generator = Worker {
            f.lifecycle.generate {
                entered.countDown()
                finish.awaitReady()
                assertTrue(calls.isEmpty())
            }
        }
        entered.awaitReady()
        val setters = listOf("audioCallback", "config", "systemPrompt", "maxTokens", "audioEnable", "config2")
        try {
            Worker {
                setters.forEach { name ->
                    f.lifecycle.mutate { ptr -> assertEquals(42L, ptr); calls.add(name) }
                }
                assertEquals(0L, f.lifecycle.withIdleHandle { it })
            }.join() // Must complete while the native generation is still gated.
            assertTrue(calls.isEmpty())
        } finally {
            finish.countDown()
        }
        generator.join()
        assertEquals(setters, calls)
        f.lifecycle.generate { assertEquals(setters, calls) }
        f.lifecycle.release()
    }

    @Test(timeout = 5000)
    fun settersDuringLoadDrainOnlyAfterHandleIsPublished() {
        val f = Fixture()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val calls = mutableListOf<Int>()
        val loader = Worker { f.lifecycle.load { entered.countDown(); finish.awaitReady(); 42L } }
        entered.awaitReady()
        try {
            Worker {
                f.lifecycle.mutate { ptr -> assertEquals(42L, ptr); calls.add(1) }
                f.lifecycle.mutate { calls.add(2) }
                assertEquals(0L, f.lifecycle.withIdleHandle { it })
            }.join()
            assertTrue(calls.isEmpty())
        } finally {
            finish.countDown()
        }
        loader.join()
        assertEquals(listOf(1, 2), calls)
        f.lifecycle.release()
    }

    @Test(timeout = 5000)
    fun releaseDiscardsPendingMutationsDuringLoadAndGeneration() {
        for (loading in listOf(false, true)) {
            val f = Fixture()
            if (!loading) f.lifecycle.load { 42L }
            val entered = CountDownLatch(1)
            val finish = CountDownLatch(1)
            val worker = Worker {
                if (loading) f.lifecycle.load { entered.countDown(); finish.awaitReady(); 42L }
                else f.lifecycle.generate { entered.countDown(); finish.awaitReady() }
            }
            entered.awaitReady()
            try {
                f.lifecycle.mutate { fail("Close must cancel queued setters") }
                f.lifecycle.release()
                f.lifecycle.mutate { fail("Close must reject new setters") }
                assertEquals(0, f.releases.get())
            } finally {
                finish.countDown()
            }
            worker.join()
            assertEquals(1, f.releases.get())
            assertEquals(1, f.closed.get())
        }
    }

    @Test
    fun synchronousCallbackCannotReplaceCallbackOrReadNativeStringsInFlight() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        var callbackVersion = 1
        f.lifecycle.generate {
            f.lifecycle.mutate { callbackVersion = 2 }
            assertEquals(1, callbackVersion)
            assertEquals(0L, f.lifecycle.withIdleHandle { it })
        }
        assertEquals(2, callbackVersion)
        assertEquals(42L, f.lifecycle.withIdleHandle { it })
        f.lifecycle.release()
    }

    @Test
    fun generationFailureStillDrainsAcceptedSetters() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        var applied = false
        expectFailure {
            f.lifecycle.generate {
                f.lifecycle.mutate { applied = true }
                error("generation")
            }
        }
        assertTrue(applied)
        f.lifecycle.release()
    }

    @Test
    fun failedLoadDropsPendingSettersRatherThanApplyingThemOnRetry() {
        val f = Fixture()
        expectFailure {
            f.lifecycle.load {
                f.lifecycle.mutate { fail("Failed load must discard pending work") }
                error("load")
            }
        }
        f.lifecycle.load { 42L }
        f.lifecycle.release()
    }

    @Test
    fun idleSettersAreSynchronousAndUnloadedSettersAreNoOps() {
        val f = Fixture()
        f.lifecycle.mutate { fail("No handle before load") }
        f.lifecycle.load { 42L }
        var value = 0
        f.lifecycle.mutate { value = 1 }
        assertEquals(1, value)
        f.lifecycle.release()
        f.lifecycle.mutate { fail("No handle after close") }
    }

    @Test
    fun setterFailureDoesNotStrandOtherPendingSettersOrLeaveBusyState() {
        val f = Fixture()
        f.lifecycle.load { 42L }
        var applied = false
        expectFailure {
            f.lifecycle.generate {
                f.lifecycle.mutate { error("setter") }
                f.lifecycle.mutate { applied = true }
            }
        }
        assertTrue(applied)
        assertEquals(42L, f.lifecycle.generate { it })
        f.lifecycle.release(waitForIdle = true)
        assertEquals(1, f.releases.get())
    }

    @Test
    fun nativeReleaseExceptionDoesNotDoubleFreeOrSkipRegistryCleanup() {
        val calls = AtomicInteger()
        val closed = AtomicInteger()
        val lifecycle = NativeSessionLifecycle({ calls.incrementAndGet(); error("release") }, { closed.incrementAndGet() })
        lifecycle.load { 42L }
        expectFailure { lifecycle.release() }
        lifecycle.release()
        assertFalse(lifecycle.isLoaded())
        assertEquals(1, calls.get())
        assertEquals(1, closed.get())
    }
}
