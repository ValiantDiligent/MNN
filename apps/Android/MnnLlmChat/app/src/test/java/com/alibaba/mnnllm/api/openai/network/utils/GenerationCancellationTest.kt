package com.alibaba.mnnllm.api.openai.network.utils

import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import java.io.IOException
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

@RunWith(Parameterized::class)
class GenerationCancellationTest(private val protocolName: String, private val stream: Boolean) {
    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0} stream={1}")
        fun cases() = listOf("OPENAI", "ANTHROPIC").flatMap { protocol ->
            listOf(false, true).map { arrayOf<Any>(protocol, it) }
        }
    }

    @Test
    fun requestCancellationStopsProgressButKeepsQueueSlotUntilNativeReturns() = runBlocking {
        val queue = RequestQueueManager()
        val started = CompletableDeferred<Unit>()
        val stopped = CompletableDeferred<Unit>()
        val checkStop = CountDownLatch(1)
        val exitNative = CountDownLatch(1)
        val active = AtomicInteger()
        val maxConcurrent = AtomicInteger()
        val order = Collections.synchronizedList(mutableListOf<String>())
        val submitter = GenerationSubmitter { progress ->
            val concurrency = active.incrementAndGet()
            maxConcurrent.updateAndGet { maxOf(it, concurrency) }
            try {
                order.add("first-enter")
                assertFalse(progress("partial"))
                started.complete(Unit)
                check(checkStop.await(5, TimeUnit.SECONDS)) { "No cancellation signal from test" }
                assertTrue("text callback must stop JNI", progress("discard"))
                assertTrue("null callback must also observe cancellation", progress(null))
                stopped.complete(Unit)
                check(exitNative.await(5, TimeUnit.SECONDS)) { "Native exit was not released" }
                order.add("first-exit")
                hashMapOf<String, Any>("prompt_len" to 1L, "decode_len" to 1L)
            } finally {
                active.decrementAndGet()
            }
        }
        try {
            val first = launch(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("first", "trace", task = {
                    val response = GenerationResponse(GenerationProtocol.valueOf(protocolName), "test-id")
                    if (stream) {
                        response.stream(submitter, RequestCancellation.from(currentCoroutineContext())) {}
                    } else {
                        response.nonStream(submitter)
                    }
                    fail("Cancelled generation must not return success")
                })
            }
            withTimeout(5_000) { started.await() }
            val second = launch(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("second", "trace", task = {
                    val concurrency = active.incrementAndGet()
                    maxConcurrent.updateAndGet { maxOf(it, concurrency) }
                    order.add("second")
                    active.decrementAndGet()
                })
            }
            assertEquals(1, queue.getQueueSize())
            first.cancel()
            checkStop.countDown()
            withTimeout(5_000) { stopped.await() }
            repeat(1_000) {
                assertEquals(1, queue.getQueueSize())
                assertTrue(queue.getQueueStats().isProcessing)
                assertFalse(second.isCompleted)
            }
            assertEquals(listOf("first-enter"), order.toList())
            exitNative.countDown()
            withTimeout(5_000) { first.join(); second.join() }
            assertEquals(listOf("first-enter", "first-exit", "second"), order.toList())
            assertEquals(1, maxConcurrent.get())
            assertEquals(0, active.get())
        } finally {
            checkStop.countDown()
            exitNative.countDown()
            queue.shutdown()
        }
    }
}

class GenerationWriterFailureTest {
    @Test
    fun writerIOExceptionStopsBothProducersAndWaitsForTheirNativeExit() = runBlocking {
        for (protocol in GenerationProtocol.values()) {
            val entered = CompletableDeferred<Unit>()
            val writerFailed = CountDownLatch(1)
            val stopped = CompletableDeferred<Unit>()
            val exitNative = CountDownLatch(1)
            val nativeExited = CompletableDeferred<Unit>()
            val responseReturned = CompletableDeferred<Unit>()
            val submitter = GenerationSubmitter { progress ->
                entered.complete(Unit)
                progress("partial")
                check(writerFailed.await(5, TimeUnit.SECONDS))
                // The consumer's finally must make the callback return true without relying on JNI suspension.
                val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
                var stop = progress(null)
                while (!stop && System.nanoTime() < deadline) {
                    Thread.yield()
                    stop = progress(null)
                }
                assertTrue("Writer failure must request native stop", stop)
                stopped.complete(Unit)
                check(exitNative.await(5, TimeUnit.SECONDS))
                nativeExited.complete(Unit)
                hashMapOf<String, Any>()
            }
            val request = launch {
                try {
                    GenerationResponse(protocol, "id").stream(
                        submitter, RequestCancellation.from(currentCoroutineContext())
                    ) {
                        entered.await()
                        writerFailed.countDown()
                        throw IOException("simulated broken writer")
                    }
                    fail("IOException must propagate from the consumer")
                } catch (e: IOException) {
                    assertEquals("simulated broken writer", e.message)
                    assertTrue(nativeExited.isCompleted)
                } finally {
                    responseReturned.complete(Unit)
                }
            }
            try {
                withTimeout(5_000) { stopped.await() }
                assertFalse(responseReturned.isCompleted)
                exitNative.countDown()
                withTimeout(5_000) { request.join() }
                assertTrue(nativeExited.isCompleted)
            } finally {
                writerFailed.countDown()
                exitNative.countDown()
                request.cancelAndJoin()
            }
        }
    }
}
