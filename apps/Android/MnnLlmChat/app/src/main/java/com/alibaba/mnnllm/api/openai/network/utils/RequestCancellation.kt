package com.alibaba.mnnllm.api.openai.network.utils

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

/** A request-local stop signal that synchronous native progress callbacks can read. */
internal class RequestCancellation(
    private val job: Job? = null,
    private val parent: RequestCancellation? = null
) : AbstractCoroutineContextElement(Key) {
    companion object Key : CoroutineContext.Key<RequestCancellation> {
        fun from(context: CoroutineContext) = RequestCancellation(context[Job], context[Key])
    }

    private val stopped = AtomicBoolean(false)

    val isCancelled: Boolean
        get() = stopped.get() || job?.isActive == false || parent?.isCancelled == true

    fun cancel() {
        stopped.set(true)
    }

    fun ensureActive() {
        if (isCancelled) throw CancellationException("Generation request cancelled")
    }
}
