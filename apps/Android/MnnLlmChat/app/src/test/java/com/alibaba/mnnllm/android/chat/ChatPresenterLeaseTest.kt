package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.lifecycle.LifecycleRegistry
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import com.alibaba.mnnllm.api.openai.runtime.LlmRuntimeController
import com.alibaba.mnnllm.api.openai.runtime.RuntimeSessionOwnership
import com.alibaba.mnnllm.api.openai.runtime.SessionLeaseToken
import io.mockk.*
import kotlinx.coroutines.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Real Presenter entry points with the same ownership gate as the runtime, without JNI or a model. */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class ChatPresenterLeaseTest {
    private val runtime = mockk<LlmRuntimeController>(relaxed = true)
    private val session = mockk<LlmSession>(relaxed = true)
    private val ownership = RuntimeSessionOwnership<LlmSession>()
    private val lock = Any()
    private val releases = ConcurrentHashMap<SessionLeaseToken, CountDownLatch>()
    private val presenters = mutableListOf<ChatPresenter>()
    private lateinit var lastToken: SessionLeaseToken
    private var nativeReleases = 0

    @Before
    fun setUp() {
        mockkObject(ChatDataManager.Companion)
        every { ChatDataManager.getInstance(any()) } returns mockk(relaxed = true)
        every { session.sessionId } returns "session"
        every { session.isModelLoaded() } returns true
        ServiceLocator.setLlmRuntimeController(runtime)
        every { runtime.ensureSession(any(), any(), any(), any(), any(), any(), any()) } answers { acquire() }
        every { runtime.releaseSession(any()) } answers {
            val token = firstArg<SessionLeaseToken>()
            synchronized(lock) {
                ownership.release(token) { nativeReleases++ }
            }.also { releases[token]?.countDown() }
        }
        // Reproduces the old keepForApi leak even before the page reports load completion.
        ServerEventManager.getInstance().handleApplicationStarted()
    }

    private fun acquire(): EnsureSessionResult = synchronized(lock) {
        lastToken = ownership.acquire(session)
        releases[lastToken] = CountDownLatch(1)
        EnsureSessionResult(true, session, "model", leaseToken = lastToken)
    }

    private fun presenter(): ChatPresenter {
        val activity = mockk<ChatActivity>(relaxed = true)
        every { activity.intent } returns Intent().putExtra("configFilePath", "/unused/config.json")
        every { activity.lifecycle } returns LifecycleRegistry(activity)
        return ChatPresenter(activity, "Qwen", "model").also { presenters.add(it) }
    }

    private fun awaitRelease(token: SessionLeaseToken) {
        assertTrue("Presenter must release its lease", releases.getValue(token).await(3, TimeUnit.SECONDS))
    }

    private fun assertAlive() = synchronized(lock) {
        assertSame(session, ownership.session)
        assertEquals(0, nativeReleases)
    }

    private fun assertClosed() = synchronized(lock) {
        assertNull(ownership.session)
        assertEquals(1, nativeReleases)
    }

    @After
    fun tearDown() {
        presenters.forEach { it.destroy() }
        releases.keys.forEach { runtime.releaseSession(it) }
        ServerEventManager.getInstance().resetRuntimeState()
        ServiceLocator.reset()
        unmockkAll()
    }

    @Test
    fun apiLeaseThenPageExitBeforeLoadNotificationThenApiStopReleasesExactlyOnce() {
        val api = acquire().leaseToken!!
        val page = presenter()
        page.createSession()
        val ui = lastToken
        page.destroy()
        awaitRelease(ui)
        assertAlive()
        assertTrue(runtime.releaseSession(api))
        assertClosed()
        page.destroy()
        assertFalse(runtime.releaseSession(ui))
        assertFalse(runtime.releaseSession(api))
        assertClosed()
        verify(exactly = 0) { runtime.releaseSession() }
    }

    @Test
    fun apiStopsFirstAndPageExitReleasesLastLease() {
        val api = acquire().leaseToken!!
        val page = presenter()
        page.createSession()
        val ui = lastToken
        assertTrue(runtime.releaseSession(api))
        assertAlive()
        page.destroy()
        awaitRelease(ui)
        assertClosed()
    }

    @Test
    fun repeatedRememberReleasesPreviousLeaseWithoutUnloadingSameSession() {
        val page = presenter()
        page.createSession()
        repeat(20) {
            val previous = lastToken
            page.createSession()
            assertNotSame(previous, lastToken)
            awaitRelease(previous)
            assertAlive()
        }
        val last = lastToken
        page.destroy()
        awaitRelease(last)
        assertClosed()
    }

    @Test
    fun lateCleanupOfFirstPageKeepsSecondPageLeaseAlive() {
        val first = presenter()
        first.createSession()
        val a = lastToken
        val second = presenter()
        second.createSession()
        val b = lastToken
        first.destroy()
        awaitRelease(a)
        assertAlive()
        assertFalse(runtime.releaseSession(a))
        second.destroy()
        awaitRelease(b)
        assertClosed()
    }

    @Test
    fun ensureReturningAfterDestroyDropsLateLeaseEvenWhileApiRuns() = runBlocking {
        val api = acquire().leaseToken!!
        val page = presenter()
        val entered = CountDownLatch(1)
        val resume = CountDownLatch(1)
        every { runtime.ensureSession(any(), any(), any(), any(), any(), any(), any()) } answers {
            entered.countDown()
            check(resume.await(3, TimeUnit.SECONDS))
            acquire()
        }
        val create = async(Dispatchers.IO) {
            try {
                page.createSession()
                fail("Late creation must not survive destroy")
            } catch (_: CancellationException) {
            }
        }
        try {
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            page.destroy()
        } finally {
            resume.countDown()
        }
        create.await()
        awaitRelease(lastToken)
        assertAlive()
        assertTrue(runtime.releaseSession(api))
        assertClosed()
    }

    @Test
    fun loadFailureReleasesOnlyPageLeaseEvenWhileApiRuns() {
        val api = acquire().leaseToken!!
        val page = presenter()
        page.createSession()
        val ui = lastToken
        every { session.isModelLoaded() } returns false
        every { session.load() } throws IllegalStateException("load failed")
        page.load()
        awaitRelease(ui)
        assertAlive()
        page.destroy()
        assertTrue(runtime.releaseSession(api))
        assertClosed()
    }

    @Test
    fun lateLoadFailureCannotClearReplacementOwnerOfSamePresenter() {
        val page = presenter()
        page.createSession()
        val old = lastToken
        val entered = CountDownLatch(1)
        val resume = CountDownLatch(1)
        val rollback = CountDownLatch(1)
        every { session.isModelLoaded() } returns false
        every { session.load() } answers {
            entered.countDown()
            check(resume.await(3, TimeUnit.SECONDS))
            throw IllegalStateException("late load failure")
        }
        page.load()
        try {
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            page.createSession()
            awaitRelease(old)
            releases[old] = rollback // Observe the delayed failure's second, idempotent release.
        } finally {
            resume.countDown()
        }
        assertTrue(rollback.await(3, TimeUnit.SECONDS))
        assertAlive()
        val next = lastToken
        page.destroy()
        awaitRelease(next)
        assertClosed()
    }

    @Test
    fun loadFailureWithoutApiReleasesLastLease() {
        val page = presenter()
        page.createSession()
        val ui = lastToken
        every { session.isModelLoaded() } returns false
        every { session.load() } throws IllegalStateException("load failed")
        page.load()
        awaitRelease(ui)
        assertClosed()
        page.destroy()
        assertFalse(runtime.releaseSession(ui))
    }
}
