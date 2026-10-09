package com.alibaba.mnnllm.api.openai.network.utils

import com.alibaba.mnnllm.api.openai.network.compat.AnthropicAdapter
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicUsage
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/** The only native seam needed by protocol tests; submitting is synchronous. */
internal fun interface GenerationSubmitter {
    fun submit(onProgress: (String?) -> Boolean): HashMap<String, Any>
}

internal enum class GenerationProtocol { OPENAI, ANTHROPIC }

internal class GenerationException(message: String, val status: Int = 500) : Exception(message)

internal data class GenerationHttpResponse(val body: String, val status: Int = 200)

/** Shared result validation and request lifetime for both wire protocols. No native library is loaded here. */
internal class GenerationResponse(
    private val protocol: GenerationProtocol,
    private val responseId: String,
    private val created: Long = System.currentTimeMillis() / 1000,
    private val model: String = "mnn-local"
) {
    private val formatter = ChatResponseFormatter()

    fun errorResponse(error: Exception): GenerationHttpResponse {
        val body = buildJsonObject {
            if (protocol == GenerationProtocol.ANTHROPIC) put("type", "error")
            put("error", buildJsonObject {
                put("message", error.message?.takeIf { it.isNotBlank() } ?: "Generation failed")
                put("type", if (protocol == GenerationProtocol.ANTHROPIC) "api_error" else "server_error")
                if (protocol == GenerationProtocol.OPENAI) {
                    put("param", JsonNull)
                    put("code", "generation_error")
                }
            })
        }.toString()
        return GenerationHttpResponse(body, (error as? GenerationException)?.status ?: 500)
    }

    suspend fun nonStream(submitter: GenerationSubmitter): GenerationHttpResponse {
        val cancellation = RequestCancellation.from(currentCoroutineContext())
        return try {
            val text = StringBuilder()
            val metrics = withContext(Dispatchers.IO) {
                submitChecked(submitter, cancellation) { text.append(it) }
            }
            cancellation.ensureActive()
            val body = when (protocol) {
                GenerationProtocol.OPENAI -> formatter.createChatCompletionResponse(
                    responseId, created, text.toString(), formatter.createUsageFromMetrics(metrics)
                )
                GenerationProtocol.ANTHROPIC -> AnthropicAdapter.createNonStreamResponse(
                    responseId, model, text.toString(), usageFromMetrics(metrics)
                )
            }
            GenerationHttpResponse(body)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            errorResponse(e)
        }
    }

    suspend fun stream(
        submitter: GenerationSubmitter,
        request: RequestCancellation,
        send: suspend (ServerSentEvent) -> Unit
    ) = coroutineScope {
        // Keep this child signal local: normal stream cleanup must not cancel the enclosing queue request.
        val cancellation = RequestCancellation(currentCoroutineContext()[kotlinx.coroutines.Job], request)
        val channel = Channel<ServerSentEvent>(Channel.UNLIMITED)
        val producer = launch(Dispatchers.IO) {
            try {
                cancellation.ensureActive()
                startEvents().forEach { channel.send(it) }
                val metrics = submitChecked(submitter, cancellation) { text ->
                    if (channel.trySend(deltaEvent(text)).isFailure) cancellation.cancel()
                }
                // A null progress callback is not proof of success. Only a successful submit result is.
                successEvents(metrics).forEach { channel.send(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (!cancellation.isCancelled) {
                    channel.trySend(ServerSentEvent(data = errorResponse(e).body, event = "error"))
                }
            } finally {
                channel.close()
            }
        }
        try {
            for (event in channel) {
                cancellation.ensureActive()
                send(event)
            }
        } finally {
            // Includes IOException, SSE session cancellation and a failed/closed writer. Do not leave a
            // producer generating into an unread channel, or release the queue while JNI is still running.
            cancellation.cancel()
            channel.cancel()
            producer.cancel()
            withContext(NonCancellable) { producer.join() }
        }
    }

    private fun submitChecked(
        submitter: GenerationSubmitter,
        cancellation: RequestCancellation,
        onText: (String) -> Unit
    ): HashMap<String, Any> {
        cancellation.ensureActive()
        var callbackError: Exception? = null
        val metrics = submitter.submit { progress ->
            if (cancellation.isCancelled || callbackError != null) {
                true
            } else {
                try {
                    if (progress != null) onText(progress)
                } catch (e: Exception) {
                    // JNI may swallow callback exceptions. Retain the cause and request a cooperative stop.
                    callbackError = e
                }
                cancellation.isCancelled || callbackError != null
            }
        }
        cancellation.ensureActive()
        if (metrics["error"] == true) {
            throw GenerationException(metrics["message"]?.toString()?.takeIf { it.isNotBlank() } ?: "Generation failed")
        }
        callbackError?.let { throw it }
        return metrics
    }

    private fun startEvents(): List<ServerSentEvent> {
        if (protocol == GenerationProtocol.OPENAI) return emptyList()
        val idJson = JsonPrimitive(responseId)
        val modelJson = JsonPrimitive(model)
        return listOf(
            ServerSentEvent(event = "message_start", data = "{\"type\":\"message_start\",\"message\":{\"id\":$idJson,\"type\":\"message\",\"role\":\"assistant\",\"model\":$modelJson,\"content\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}"),
            ServerSentEvent(event = "content_block_start", data = "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}")
        )
    }

    private fun deltaEvent(text: String): ServerSentEvent = when (protocol) {
        GenerationProtocol.OPENAI -> ServerSentEvent(data = formatter.createDeltaResponse(responseId, created, text))
        GenerationProtocol.ANTHROPIC -> ServerSentEvent(
            event = "content_block_delta",
            data = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":${JsonPrimitive(text)}}}"
        )
    }

    private fun successEvents(metrics: HashMap<String, Any>): List<ServerSentEvent> = when (protocol) {
        GenerationProtocol.OPENAI -> listOf(
            ServerSentEvent(data = formatter.createDeltaResponse(responseId, created, "", true,
                usage = formatter.createUsageFromMetrics(metrics))),
            ServerSentEvent(data = "[DONE]")
        )
        GenerationProtocol.ANTHROPIC -> {
            val usage = usageFromMetrics(metrics)
            listOf(
                ServerSentEvent(event = "content_block_stop", data = "{\"type\":\"content_block_stop\",\"index\":0}"),
                ServerSentEvent(event = "message_delta", data = "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":${usage.input_tokens},\"output_tokens\":${usage.output_tokens}}}"),
                ServerSentEvent(event = "message_stop", data = "{\"type\":\"message_stop\"}")
            )
        }
    }

    private fun usageFromMetrics(metrics: HashMap<String, Any>) = AnthropicUsage(
        input_tokens = (metrics["prompt_len"] as? Long)?.toInt() ?: 0,
        output_tokens = (metrics["decode_len"] as? Long)?.toInt() ?: 0
    )
}
