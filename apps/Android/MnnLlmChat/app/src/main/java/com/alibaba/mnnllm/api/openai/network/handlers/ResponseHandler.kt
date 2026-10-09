package com.alibaba.mnnllm.api.openai.network.handlers

import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.utils.GenerationException
import com.alibaba.mnnllm.api.openai.network.utils.GenerationProtocol
import com.alibaba.mnnllm.api.openai.network.utils.GenerationResponse
import com.alibaba.mnnllm.api.openai.network.utils.GenerationSubmitter
import com.alibaba.mnnllm.api.openai.network.utils.RequestCancellation
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.sse.SSEServerContent
import io.ktor.server.sse.heartbeat
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlin.time.Duration.Companion.seconds

/** OpenAI response transport. Generation and error semantics are shared with Anthropic. */
class ResponseHandler {
    private val logger = ChatLogger()

    suspend fun handleStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        traceId: String
    ) {
        val response = createResponse()
        // Capture the queue/request context before Ktor enters its SSE writer context.
        val cancellation = RequestCancellation.from(currentCoroutineContext())
        call.respond(SSEServerContent(call) {
            heartbeat {
                period = 3.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }
            try {
                response.stream(submitter(history), cancellation) { send(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // The writer is unusable. stream's finally has already stopped and joined JNI.
                logger.logWarning(traceId, "Client connection closed during SSE send", e)
            }
        })
    }

    suspend fun handleNonStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>
    ) {
        val response = createResponse().nonStream(submitter(history))
        call.respondText(response.body, ContentType.Application.Json, HttpStatusCode.fromValue(response.status))
    }

    private fun createResponse(): GenerationResponse {
        val timestamp = System.currentTimeMillis()
        return GenerationResponse(GenerationProtocol.OPENAI, "chatcmpl-$timestamp", timestamp / 1000)
    }

    private fun submitter(history: List<android.util.Pair<String, String>>) = GenerationSubmitter { onProgress ->
        val session = ServiceLocator.getChatSessionProvider().getLlmSession()
            ?: throw GenerationException("LlmSession not available", 503)
        session.submitFullHistory(history, object : GenerateProgressListener {
            override fun onProgress(progress: String?) = onProgress.invoke(progress)
        })
    }
}