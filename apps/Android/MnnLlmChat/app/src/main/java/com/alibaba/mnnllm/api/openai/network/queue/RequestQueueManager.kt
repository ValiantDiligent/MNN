package com.alibaba.mnnllm.api.openai.network.queue

import com.alibaba.mnnllm.api.openai.network.utils.RequestCancellation
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import timber.log.Timber
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/** FIFO queue: cancellation cannot release a slot until synchronous generation has exited. */
class RequestQueueManager internal constructor(dispatcher: CoroutineDispatcher = Dispatchers.IO) {
    companion object {
        @Volatile
        private var INSTANCE: RequestQueueManager? = null

        fun getInstance(): RequestQueueManager = INSTANCE ?: synchronized(this) {
            INSTANCE ?: RequestQueueManager().also { INSTANCE = it }
        }
    }

    private val pendingRequests = AtomicInteger(0)
    private val isProcessing = AtomicBoolean(false)
    private val totalRequests = AtomicLong(0)
    private val completedRequests = AtomicLong(0)
    private val failedRequests = AtomicLong(0)
    private val queueMutex = Mutex()
    private val queueScope = CoroutineScope(dispatcher + SupervisorJob())
    private val requestQueue = Channel<QueuedRequest>(Channel.UNLIMITED, onUndeliveredElement = { it.cancel() })

    private class QueuedRequest(
        val requestId: String,
        val traceId: String,
        val task: suspend () -> Unit,
        val completion: CompletableDeferred<Result<Unit>>,
        val cancellation: RequestCancellation,
        val pendingCount: AtomicInteger,
        val onComplete: () -> Unit,
        val onError: (Exception) -> Unit
    ) {
        private val pending = AtomicBoolean(true)

        fun leaveQueue() {
            if (pending.compareAndSet(true, false)) pendingCount.decrementAndGet()
        }

        fun cancel() {
            leaveQueue()
            cancellation.cancel()
            completion.cancel()
        }
    }

    init {
        queueScope.launch {
            for (request in requestQueue) {
                request.leaveQueue()
                if (request.completion.isCancelled || request.cancellation.isCancelled) {
                    request.cancel()
                    continue
                }
                processRequest(request)
            }
        }
    }

    /** Priority is retained for call-site compatibility; requests are deliberately scheduled FIFO. */
    suspend fun submitRequest(
        requestId: String,
        traceId: String,
        task: suspend () -> Unit,
        onComplete: () -> Unit = {},
        onError: (Exception) -> Unit = {},
        priority: Int = 0
    ) {
        val context = currentCoroutineContext()
        context.ensureActive()
        // Parent cancellation reaches this completion immediately. Failures are values so they do not
        // cancel the HTTP request before the caller has a chance to report its protocol error.
        val completion = CompletableDeferred<Result<Unit>>(context[Job])
        val request = QueuedRequest(requestId, traceId, task, completion,
            RequestCancellation.from(context), pendingRequests, onComplete, onError)
        pendingRequests.incrementAndGet()
        val cancellationHandle = completion.invokeOnCompletion {
            if (completion.isCancelled) request.cancel()
        }
        try {
            queueMutex.withLock {
                context.ensureActive()
                requestQueue.trySend(request).getOrThrow()
                totalRequests.incrementAndGet()
                Timber.tag("RequestQueue").d("Queued requestId=$requestId, traceId=$traceId, priority=$priority")
            }
            completion.await().getOrThrow()
        } finally {
            cancellationHandle.dispose()
            if (!completion.isCompleted) request.cancel()
            // Also covers failed enqueue or cancellation before the processor dequeues a tombstone.
            request.leaveQueue()
        }
    }

    private suspend fun processRequest(request: QueuedRequest) = coroutineScope {
        isProcessing.set(true)
        val execution = launch(request.cancellation, start = CoroutineStart.LAZY) {
            try {
                request.cancellation.ensureActive()
                request.task()
                currentCoroutineContext().ensureActive()
                request.cancellation.ensureActive()
                completedRequests.incrementAndGet()
                notifySafely { request.onComplete() }
                request.completion.complete(Result.success(Unit))
            } catch (e: CancellationException) {
                request.cancel()
            } catch (e: Exception) {
                failedRequests.incrementAndGet()
                notifySafely { request.onError(e) }
                request.completion.complete(Result.failure(e))
            }
        }
        val cancellationHandle = request.completion.invokeOnCompletion {
            if (request.completion.isCancelled) execution.cancel()
        }
        try {
            execution.start()
            execution.join()
        } finally {
            // Cancellation changes Job state, not the execution state of a blocking JNI call.
            // Join the worker and its structured SSE children before advancing FIFO.
            execution.cancel()
            withContext(NonCancellable) { execution.join() }
            cancellationHandle.dispose()
            if (!request.completion.isCompleted) request.cancel()
            isProcessing.set(false)
        }
    }

    private fun notifySafely(callback: () -> Unit) {
        try {
            callback()
        } catch (e: Exception) {
            Timber.tag("RequestQueue").e(e, "Queue callback failed")
        }
    }

    /** Non-blocking, observational only. Never receives from or sends to the queue. */
    fun getQueueSize(): Int = pendingRequests.get()

    fun getQueueStats() = QueueStats(
        totalRequests = totalRequests.get(),
        completedRequests = completedRequests.get(),
        failedRequests = failedRequests.get(),
        pendingRequests = getQueueSize().toLong(),
        isProcessing = isProcessing.get()
    )

    data class QueueStats(
        val totalRequests: Long,
        val completedRequests: Long,
        val failedRequests: Long,
        val pendingRequests: Long,
        val isProcessing: Boolean
    ) {
        val successRate: Double
            get() = if (totalRequests > 0) completedRequests.toDouble() / totalRequests * 100 else 0.0
    }

    /** Resolve discarded waiters rather than leaving submitRequest suspended forever. */
    suspend fun clearQueue() {
        queueMutex.withLock {
            while (true) {
                val request = requestQueue.tryReceive().getOrNull() ?: break
                request.cancel()
            }
        }
    }

    fun shutdown() {
        requestQueue.cancel()
        queueScope.cancel()
    }
}