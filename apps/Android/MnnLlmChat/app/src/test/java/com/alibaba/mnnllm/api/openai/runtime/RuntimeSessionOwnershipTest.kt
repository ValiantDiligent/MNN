package com.alibaba.mnnllm.api.openai.runtime

import com.alibaba.mnnllm.android.llm.NativeSessionLifecycle
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Uses the same ownership gate as DefaultLlmRuntimeController, with a controllable native session. */
class RuntimeSessionOwnershipTest {
    private class Session(val model: String) {
        var releaseCount = 0
        val lifecycle = NativeSessionLifecycle({ releaseCount++ }, {})
        fun load() = lifecycle.load { 1L }
        fun release() = lifecycle.release(waitForIdle = true)
    }

    private fun delayedCleanup(sameModel: Boolean) {
        val lock = Any() // Same serialization boundary as DefaultLlmRuntimeController.
        val ownership = RuntimeSessionOwnership<Session>()
        val old = Session("old").also { it.load() }
        val token = synchronized(lock) { ownership.acquire(old) }
        val queued = CountDownLatch(1)
        val resume = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>()
        val cleanup = Thread {
            try {
                queued.countDown()
                assertTrue(resume.await(2, TimeUnit.SECONDS))
                synchronized(lock) {
                    assertEquals(sameModel, ownership.release(token) { fail("Another lease still owns the session") })
                    assertFalse(ownership.release(token) { fail("Duplicate cleanup") })
                }
            } catch (error: Throwable) {
                failure.set(error)
            }
        }.apply { isDaemon = true; start() }
        assertTrue(queued.await(2, TimeUnit.SECONDS))
        val next = if (sameModel) old else Session("new").also { it.load() }
        val newToken = synchronized(lock) {
            if (!sameModel) ownership.release { it.release() }
            ownership.acquire(next)
        }
        assertNotSame(token, newToken)
        resume.countDown()
        cleanup.join(2000)
        assertFalse(cleanup.isAlive)
        failure.get()?.let { throw AssertionError("Cleanup failed", it) }
        synchronized(lock) {
            assertSame(next, ownership.session)
            assertTrue(next.lifecycle.isLoaded())
            assertEquals(0, next.releaseCount)
            assertTrue(ownership.release(newToken) { it.release() })
            assertFalse(ownership.release(newToken) { fail("Repeated release") })
        }
        assertEquals(1, next.releaseCount)
    }

    @Test(timeout = 5000)
    fun oldOwnerOfSameModelCannotReleaseReusedSession() = delayedCleanup(sameModel = true)

    @Test(timeout = 5000)
    fun oldOwnerOfDifferentModelCannotReleaseReplacementSession() = delayedCleanup(sameModel = false)

    @Test
    fun apiAcquiringLoadedChatSessionKeepsItAliveAfterPageCleanupWithoutReload() {
        val ownership = RuntimeSessionOwnership<Session>()
        val session = Session("app-config-model").also { it.load() }
        val pageToken = ownership.acquire(session)
        // acquireActiveSession deliberately preserves the loaded object and its configuration.
        val apiToken = ownership.acquire(session)
        assertNotSame(pageToken, apiToken)
        assertTrue(ownership.release(pageToken) { fail("Would stop new API session") })
        assertFalse(ownership.release(pageToken) { fail("Duplicate page cleanup") })
        assertSame(session, ownership.session)
        assertTrue(session.lifecycle.isLoaded())
        assertEquals(0, session.releaseCount)
        assertTrue(ownership.release(apiToken) { it.release() })
        assertEquals(1, session.releaseCount)
        assertNull(ownership.session)
        assertFalse(ownership.release(apiToken) { fail("Duplicate API stop") })
    }

    private fun apiAndPageStop(pageFirst: Boolean) {
        val ownership = RuntimeSessionOwnership<Session>()
        val session = Session("api-model").also { it.load() }
        val api = ownership.acquire(session)
        val page = ownership.acquire(session) // Page can exit before notifying the API of reuse.
        val first = if (pageFirst) page else api
        val last = if (pageFirst) api else page
        assertTrue(ownership.release(first) { fail("Other owner must keep the model alive") })
        assertTrue(session.lifecycle.isLoaded())
        assertSame(session, ownership.session)
        assertEquals(0, session.releaseCount)
        assertFalse(ownership.release(first) { fail("Duplicate close") })
        assertTrue(ownership.release(last) { it.release() })
        assertEquals(1, session.releaseCount)
        assertNull(ownership.session)
        assertFalse(ownership.release(last) { fail("Duplicate last close") })
    }

    @Test
    fun apiThenPageExitBeforeLoadNotificationThenApiStopReleasesExactlyOnce() = apiAndPageStop(pageFirst = true)

    @Test
    fun apiStopBeforePageExitReleasesExactlyOnce() = apiAndPageStop(pageFirst = false)

    @Test
    fun administrativeReleaseInvalidatesAllLeasesEvenWhenReacquiringSameObject() {
        val ownership = RuntimeSessionOwnership<Session>()
        val session = Session("model").also { it.load() }
        val a = ownership.acquire(session)
        val b = ownership.acquire(session)
        assertTrue(ownership.release(null) { it.release() })
        assertEquals(1, session.releaseCount)
        assertNull(ownership.session)
        val next = ownership.acquire(session)
        assertFalse(ownership.release(a) { fail("Administrative release invalidates A") })
        assertFalse(ownership.release(b) { fail("Administrative release invalidates B") })
        assertSame(session, ownership.session)
        var callbacks = 0
        assertTrue(ownership.release(next) { callbacks++ })
        assertEquals(1, callbacks)
        assertFalse(ownership.release(null) { fail("Already detached") })
    }

    @Test
    fun differentSessionMustBeExplicitlyReleasedBeforeAcquire() {
        val ownership = RuntimeSessionOwnership<Session>()
        val old = Session("old").also { it.load() }
        val a = ownership.acquire(old)
        val b = ownership.acquire(old)
        val next = Session("new").also { it.load() }
        try {
            ownership.acquire(next)
            fail("Must not silently replace a leased session")
        } catch (_: IllegalStateException) {
        }
        assertSame(old, ownership.session)
        assertEquals(0, old.releaseCount)
        assertTrue(ownership.release(null) { it.release() })
        assertEquals(1, old.releaseCount)
        val c = ownership.acquire(next)
        assertFalse(ownership.release(a) { fail("Old model A") })
        assertFalse(ownership.release(b) { fail("Old model B") })
        assertSame(next, ownership.session)
        assertTrue(next.lifecycle.isLoaded())
        assertTrue(ownership.release(c) { it.release() })
        assertEquals(1, next.releaseCount)
    }

    @Test
    fun repeatedAcquireBeforeReleaseDoesNotAccumulateLeasesOrUnload() {
        val ownership = RuntimeSessionOwnership<Session>()
        val session = Session("model").also { it.load() }
        var token = ownership.acquire(session)
        repeat(100) {
            val next = ownership.acquire(session)
            assertNotSame(token, next)
            assertTrue(ownership.release(token) { fail("Reuse must never drop to zero") })
            assertFalse(ownership.release(token) { fail("Duplicate previous lease") })
            token = next
        }
        assertTrue(session.lifecycle.isLoaded())
        assertTrue(ownership.release(token) { it.release() })
        assertNull(ownership.session)
        assertEquals(1, session.releaseCount)
    }

    @Test(timeout = 5000)
    fun delayedLoadFailureRollbackCannotReleaseNewOwner() {
        val lock = Any()
        val ownership = RuntimeSessionOwnership<Session>()
        val failed = Session("failed")
        val oldToken = ownership.acquire(failed)
        val failedLoad = CountDownLatch(1)
        val rollback = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>()
        val loader = Thread {
            try {
                try {
                    failed.lifecycle.load { error("JNI load exception") }
                } catch (_: IllegalStateException) {
                    failedLoad.countDown()
                    assertTrue(rollback.await(2, TimeUnit.SECONDS))
                    synchronized(lock) {
                        assertFalse(ownership.release(oldToken) { fail("Late failure released a newer owner") })
                    }
                }
            } catch (error: Throwable) { failure.set(error) }
        }.apply { isDaemon = true; start() }
        assertTrue(failedLoad.await(2, TimeUnit.SECONDS))
        val next = Session("replacement").also { it.load() }
        synchronized(lock) {
            assertTrue(ownership.release { it.release() }) // Must not deadlock on failed load.
            ownership.acquire(next)
        }
        rollback.countDown()
        loader.join(2000)
        assertFalse(loader.isAlive)
        failure.get()?.let { throw AssertionError("Rollback failed", it) }
        assertSame(next, ownership.session)
        assertTrue(next.lifecycle.isLoaded())
        assertEquals(0, next.releaseCount)
        ownership.release { it.release() }
    }

    @Test
    fun releaseDetachesOwnershipEvenIfCleanupThrows() {
        val ownership = RuntimeSessionOwnership<Session>()
        val token = ownership.acquire(Session("model"))
        try {
            ownership.release(token) { error("cleanup") }
            fail("Expected cleanup failure")
        } catch (_: IllegalStateException) {
        }
        assertNull(ownership.session)
        assertFalse(ownership.release(token) { fail("Must not retry old cleanup") })
    }
}
