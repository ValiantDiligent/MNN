package com.alibaba.mnnllm.android.llm

import java.util.ArrayDeque

/**
 * Owns a native handle from initialization through the last JNI call. Long JNI calls run outside
 * the monitor so UI setters/close can record their intent without waiting for inference or load.
 * Short mutations drain in FIFO order under the monitor before another generation can enter JNI.
 */
internal class NativeSessionLifecycle(
    private val releaseNative: (Long) -> Unit,
    private val onClosed: () -> Unit
) {
    private val monitor = Object()
    private var nativePtr = 0L
    private var modelLoading = false
    private var generating = false
    private var closing = false
    private var closed = false
    private var operationThread: Thread? = null
    private val pendingMutations = ArrayDeque<(Long) -> Unit>()

    fun isLoaded(): Boolean = synchronized(monitor) { nativePtr != 0L && !modelLoading && !closing }

    fun load(initialize: () -> Long) {
        synchronized(monitor) {
            check(!closing) { "Session is closed" }
            if (nativePtr != 0L) return
            check(!modelLoading) { "Session is already loading" }
            modelLoading = true
            operationThread = Thread.currentThread()
        }
        try {
            val loadedPtr = initialize()
            check(loadedPtr != 0L) { "Model load failed - the model module could not be loaded" }
            synchronized(monitor) {
                if (closing) {
                    // Never publish a handle to an object whose owner has already closed it.
                    releaseNative(loadedPtr)
                } else {
                    nativePtr = loadedPtr
                }
            }
        } finally {
            synchronized(monitor) {
                modelLoading = false
                operationThread = null
                try {
                    drainMutations()
                } finally {
                    try {
                        closeIfIdle()
                    } finally {
                        monitor.notifyAll()
                    }
                }
            }
        }
    }

    fun <T> generate(block: (Long) -> T): T {
        val ptr = synchronized(monitor) {
            check(!closing && nativePtr != 0L && !modelLoading) { "Session is not loaded or is closed" }
            check(!generating) { "Session is already generating" }
            generating = true
            operationThread = Thread.currentThread()
            nativePtr
        }
        try {
            return block(ptr)
        } finally {
            synchronized(monitor) {
                generating = false
                operationThread = null
                try {
                    drainMutations()
                } finally {
                    try {
                        closeIfIdle()
                    } finally {
                        monitor.notifyAll()
                    }
                }
            }
        }
    }

    /**
     * Only short setters belong here. Busy setters enqueue and return, including from callbacks.
     * Never-loaded/closed sessions retain the old no-op behavior; a failed load drops its queue.
     */
    fun mutate(block: (Long) -> Unit) {
        synchronized(monitor) {
            if (closing) return
            if (modelLoading || generating) {
                pendingMutations.addLast(block)
            } else if (nativePtr != 0L) {
                block(nativePtr)
            }
        }
    }

    /** Reads of native strings/config must not overlap native writes during load or generation. */
    fun <T> withIdleHandle(block: (Long) -> T): T = synchronized(monitor) {
        block(if (closing || modelLoading || generating) 0L else nativePtr)
    }

    private fun drainMutations() {
        if (closing || nativePtr == 0L) {
            pendingMutations.clear()
            return
        }
        // Drain remaining accepted setters even if one throws; leave no work for a later owner.
        var failure: Throwable? = null
        while (!closing && pendingMutations.isNotEmpty()) {
            try {
                pendingMutations.removeFirst()(nativePtr)
            } catch (error: Throwable) {
                if (failure == null) failure = error else failure.addSuppressed(error)
            }
        }
        failure?.let { throw it }
    }

    fun release(waitForIdle: Boolean = false) {
        var interrupted = false
        try {
            synchronized(monitor) {
                closing = true // Includes never-loaded and already-released sessions.
                pendingMutations.clear() // Do not retain listeners or mutate a closing native session.
                // A synchronous JNI callback may request close. It cannot wait for itself.
                if (waitForIdle && operationThread !== Thread.currentThread()) {
                    while (modelLoading || generating) {
                        try {
                            monitor.wait()
                        } catch (_: InterruptedException) {
                            // Restore once, after leaving the wait loop, not before the next wait.
                            interrupted = true
                        }
                    }
                }
                closeIfIdle()
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt()
        }
    }

    private fun closeIfIdle() {
        if (!closing || closed || modelLoading || generating) return
        closed = true
        val ptr = nativePtr
        nativePtr = 0L // Detach before JNI, even if native destruction throws.
        try {
            if (ptr != 0L) releaseNative(ptr)
        } finally {
            onClosed()
        }
    }
}
