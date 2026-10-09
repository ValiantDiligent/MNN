package com.alibaba.mnnllm.api.openai.service

import androidx.test.core.app.ApplicationProvider
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider
import com.alibaba.mnnllm.api.openai.manager.ApiNotificationManager
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.network.application.OpenAIApplication
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class ApiServiceCoordinatorTest {
    private val runtime = mockk<LlmRuntimeController>(relaxed = true)
    private val ownership = RuntimeSessionOwnership<Any>()
    private val coordinators = mutableListOf<ApiServiceCoordinator>()
    private lateinit var acquiredToken: SessionLeaseToken
    private var released = 0

    @Before
    fun setUp() {
        ServiceLocator.setLlmRuntimeController(runtime)
        val provider = mockk<ChatSessionProvider>()
        every { provider.hasActiveSession() } returns false
        ServiceLocator.setChatSessionProvider(provider)
        every { runtime.ensureSession(any()) } answers { acquire() }
        every { runtime.acquireActiveSession() } answers { acquire() }
        every { runtime.releaseSession(any()) } answers {
            ownership.release(firstArg<SessionLeaseToken>()) { released++ }
        }
        mockkConstructor(ApiNotificationManager::class, OpenAIApplication::class)
        every { anyConstructed<ApiNotificationManager>().updateNotification(any(), any(), any()) } just Runs
        every { anyConstructed<ApiNotificationManager>().cancelNotification() } just Runs
        every { anyConstructed<OpenAIApplication>().start() } answers {
            ServerEventManager.getInstance().handleApplicationStarted()
        }
        every { anyConstructed<OpenAIApplication>().isRunning() } returns true
        every { anyConstructed<OpenAIApplication>().getPort() } returns 8080
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } just Runs
    }

    private fun acquire(): EnsureSessionResult {
        acquiredToken = ownership.acquire(ownership.session ?: Any())
        return EnsureSessionResult(success = true, modelId = "model", leaseToken = acquiredToken)
    }

    private fun coordinator() = ApiServiceCoordinator(ApplicationProvider.getApplicationContext()).also {
        assertTrue(it.initialize())
        coordinators.add(it)
    }

    @After
    fun tearDown() {
        runBlocking { coordinators.forEach { it.cleanup().join() } }
        ServiceLocator.reset()
        unmockkAll()
    }

    @Test
    fun cleanupReleasesCapturedLeaseOnlyAfterNetworkStopAndOnlyOnce() = runBlocking {
        val api = coordinator()
        assertTrue(api.startServer("model"))
        val token = acquiredToken
        val entered = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
            entered.complete(Unit)
            finish.await()
        }
        val cleanup = api.cleanup()
        try {
            withTimeout(2000) { entered.await() }
            assertSame(cleanup, api.cleanup())
            assertEquals(0, released)
            verify(exactly = 0) { runtime.releaseSession(any()) }
        } finally {
            finish.complete(Unit)
        }
        cleanup.join()
        assertEquals(1, released)
        verify(exactly = 1) { runtime.releaseSession(token) }
        verify(exactly = 0) { runtime.releaseSession() }
    }

    @Test
    fun lateOldApiCleanupCannotReleaseNewOwnerOfSameOrDifferentModel() = runBlocking {
        for (reuseModel in listOf(true, false)) {
            val api = coordinator()
            assertTrue(api.startServer("model"))
            val oldToken = acquiredToken
            val entered = CompletableDeferred<Unit>()
            val finish = CompletableDeferred<Unit>()
            coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
                entered.complete(Unit)
                finish.await()
            }
            val cleanup = api.cleanup()
            val newSession: Any
            lateinit var newToken: SessionLeaseToken
            val beforeRelease = released
            try {
                withTimeout(2000) { entered.await() }
                newSession = if (reuseModel) ownership.session!! else Any()
                if (!reuseModel) assertTrue(ownership.release(null) { released++ })
                newToken = ownership.acquire(newSession)
            } finally {
                finish.complete(Unit)
            }
            cleanup.join()
            assertSame(newSession, ownership.session)
            assertEquals(beforeRelease + if (reuseModel) 0 else 1, released)
            verify(exactly = 1) { runtime.releaseSession(oldToken) }
            assertFalse(ownership.release(oldToken) { fail("Duplicate old cleanup") })
            assertTrue(ownership.release(newToken) { released++ })
            assertNull(ownership.session)
            assertEquals(beforeRelease + if (reuseModel) 1 else 2, released)
        }
        verify(exactly = 0) { runtime.releaseSession() }
    }

    @Test
    fun reacquireLoadedSessionRemembersTheLatestLease() = runBlocking {
        val api = coordinator()
        assertTrue(api.startServer("model"))
        val firstToken = acquiredToken
        every { ServiceLocator.getChatSessionProvider().hasActiveSession() } returns true
        every { runtime.getActiveModelId() } returns "model"
        assertTrue(api.startServer("model"))
        val latestToken = acquiredToken
        assertNotSame(firstToken, latestToken)
        assertEquals(0, released)
        assertNotNull(ownership.session)
        verify(exactly = 1) { runtime.releaseSession(firstToken) }
        api.cleanup().join()
        verify(exactly = 1) { runtime.acquireActiveSession() }
        verify(exactly = 1) { runtime.releaseSession(latestToken) }
        assertEquals(1, released)
        assertNull(ownership.session)
    }

    @Test
    fun repeatedStartsIncludingNullModelReplaceOnlyOwnLeaseWithoutLeaking() = runBlocking {
        val api = coordinator()
        assertTrue(api.startServer("model"))
        val session = ownership.session!!
        val pageToken = ownership.acquire(session)
        val tokens = mutableListOf(acquiredToken)
        repeat(20) {
            assertTrue(api.startServer(if (it % 2 == 0) null else "model"))
            assertNotSame(tokens.last(), acquiredToken)
            tokens.add(acquiredToken)
            assertSame(session, ownership.session)
            assertEquals(0, released)
        }
        assertEquals(1, api.getBootstrapCount())
        tokens.dropLast(1).forEach { token -> verify(exactly = 1) { runtime.releaseSession(token) } }
        assertTrue(ownership.release(pageToken) { fail("API must keep its final lease") })
        api.cleanup().join()
        verify(exactly = 1) { runtime.releaseSession(tokens.last()) }
        assertEquals(1, released)
        assertNull(ownership.session)
    }

    @Test
    fun apiAndPageCanStopInEitherOrderWithoutReloadOrLeakedLease() = runBlocking {
        for (pageFirst in listOf(true, false)) {
            val before = released
            val api = coordinator()
            assertTrue(api.startServer("model"))
            val session = ownership.session!!
            val page = ownership.acquire(session)
            if (pageFirst) {
                assertTrue(ownership.release(page) { fail("API lease must survive page exit") })
            } else {
                api.cleanup().join()
            }
            assertSame(session, ownership.session)
            assertEquals(before, released)
            if (pageFirst) api.cleanup().join() else assertTrue(ownership.release(page) { released++ })
            assertEquals(before + 1, released)
            assertNull(ownership.session)
            assertFalse(ownership.release(page) { fail("Duplicate page cleanup") })
        }
    }

    @Test
    fun failedStartReleasesItsCurrentLeaseOnce() = runBlocking {
        val api = coordinator()
        every { anyConstructed<OpenAIApplication>().isRunning() } returns false
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
            assertEquals(0, released)
        }
        assertFalse(api.startServer("model"))
        assertEquals(1, released)
        assertNull(ownership.session)
        api.cleanup().join()
        verify(exactly = 1) { runtime.releaseSession(acquiredToken) }
        verify(exactly = 0) { runtime.releaseSession() }
    }

    @Test
    fun explicitStopStillStopsNetworkAndReleasesLease() = runBlocking {
        val api = coordinator()
        assertTrue(api.startServer("model"))
        api.stopServer().join()
        assertFalse(api.isServerRunning)
        assertEquals(1, released)
        coVerify(exactly = 1) { anyConstructed<OpenAIApplication>().stopInternal() }
        api.cleanup().join()
        verify(exactly = 1) { runtime.releaseSession(acquiredToken) }
    }

    @Test
    fun failedStartRollsBackItsLeaseAfterStopWithoutReleasingANewOwner() = runBlocking {
        val api = coordinator()
        every { anyConstructed<OpenAIApplication>().isRunning() } returns false
        lateinit var page: SessionLeaseToken
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
            assertEquals(0, released)
            page = ownership.acquire(ownership.session!!) // A page shares the failed API startup lease.
        }
        assertFalse(api.startServer("model"))
        verify(exactly = 1) { runtime.releaseSession(acquiredToken) }
        verify(exactly = 0) { runtime.releaseSession() }
        assertEquals(0, released)
        assertNotNull(ownership.session)
        api.cleanup().join()
        verify(exactly = 1) { runtime.releaseSession(any()) }
        assertTrue(ownership.release(page) { released++ })
        assertEquals(1, released)
        assertNull(ownership.session)
    }

    @Test
    fun failedStartRollbackCannotReleaseAdministrativelyReplacedModel() = runBlocking {
        val api = coordinator()
        every { anyConstructed<OpenAIApplication>().isRunning() } returns false
        lateinit var next: SessionLeaseToken
        val replacement = Any()
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
            assertTrue(ownership.release(null) { released++ })
            next = ownership.acquire(replacement)
        }
        assertFalse(api.startServer("model"))
        verify(exactly = 1) { runtime.releaseSession(acquiredToken) }
        assertSame(replacement, ownership.session)
        assertEquals(1, released)
        api.cleanup().join()
        assertSame(replacement, ownership.session)
        assertTrue(ownership.release(next) { released++ })
        assertEquals(2, released)
        assertNull(ownership.session)
    }

    @Test
    fun replacementWaitsForOldNetworkAndLateCleanupCannotResetNewGlobalState() = runBlocking {
        val old = coordinator()
        assertTrue(old.startServer("old"))
        val next = coordinator()
        val entered = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        coEvery { anyConstructed<OpenAIApplication>().stopInternal() } coAnswers {
            entered.complete(Unit)
            finish.await()
        }
        val start = async { next.startServer("new") }
        try {
            withTimeout(2000) { entered.await() }
            assertFalse(start.isCompleted)
            verify(exactly = 1) { anyConstructed<OpenAIApplication>().start() }
        } finally {
            finish.complete(Unit)
        }
        assertTrue(start.await())
        val nextToken = acquiredToken
        old.cleanup().join()
        assertEquals(ServerEventManager.ServerState.STARTED, ServerEventManager.getInstance().getCurrentState())
        verify(exactly = 1) { anyConstructed<ApiNotificationManager>().cancelNotification() }
        verify(exactly = 0) { runtime.releaseSession(nextToken) }
        assertNotNull(ownership.session)
    }

    @Test
    fun cleanupDuringSlowEnsureReturnsImmediatelyAndRollsBackLateLease() = runBlocking {
        val api = coordinator()
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        every { runtime.ensureSession(any()) } answers {
            entered.countDown()
            check(finish.await(2, TimeUnit.SECONDS))
            acquire()
        }
        val start = async(Dispatchers.IO) { api.startServer("model") }
        assertTrue(entered.await(2, TimeUnit.SECONDS))
        val cleanup: Job
        try {
            cleanup = api.cleanup() // Must not wait for the native load/start lock.
            assertFalse(cleanup.isCompleted)
            assertFalse(api.isInitialized)
        } finally {
            finish.countDown()
        }
        assertFalse(start.await())
        cleanup.join()
        verify(exactly = 0) { anyConstructed<OpenAIApplication>().start() }
        assertEquals(1, released)
    }
}
