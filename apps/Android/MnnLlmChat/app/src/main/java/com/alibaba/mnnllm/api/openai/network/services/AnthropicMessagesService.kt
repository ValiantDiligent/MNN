package com.alibaba.mnnllm.api.openai.network.services

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicAdapter
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicMessagesRequest
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import com.alibaba.mnnllm.api.openai.network.utils.GenerationException
import com.alibaba.mnnllm.api.openai.network.utils.GenerationProtocol
import com.alibaba.mnnllm.api.openai.network.utils.GenerationResponse
import com.alibaba.mnnllm.api.openai.network.utils.GenerationSubmitter
import com.alibaba.mnnllm.api.openai.network.utils.RequestCancellation
import com.alibaba.mnnllm.api.openai.network.utils.MessageTransformer
import com.alibaba.mnnllm.api.openai.network.validators.ChatRequestValidator
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
import java.util.UUID
import kotlin.time.Duration.Companion.seconds

class AnthropicMessagesService {
    private val messageTransformer = MessageTransformer()
    private val chatRequestValidator = ChatRequestValidator()
    private val logger = ChatLogger()
    private val requestQueueManager = RequestQueueManager.getInstance()
    private val chatSessionProvider = ServiceLocator.getChatSessionProvider()
    private val imageProcessor = MnnImageProcessor.getInstance(MnnLlmApplication.getAppContext())

    suspend fun processMessages(
        call: ApplicationCall,
        request: AnthropicMessagesRequest,
        traceId: String
    ) {
        val requestId = UUID.randomUUID().toString()
        val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)

        val validationResult = chatRequestValidator.validateChatRequest(openAiRequest)
        if (!validationResult.isValid) {
            call.respond(
                HttpStatusCode.BadRequest,
                mapOf("error" to mapOf("message" to validationResult.errorMessage))
            )
            return
        }

        try {
            requestQueueManager.submitRequest(
                requestId = requestId,
                traceId = traceId,
                task = {
                    processGeneration(call, request, traceId)
                },
                onComplete = {
                    logger.logInfo(traceId, "Anthropic queue completed, requestId=$requestId")
                },
                onError = { exception ->
                    logger.logError(traceId, exception, "Anthropic queue failed, requestId=$requestId")
                }
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            logger.logError(traceId, e, "Anthropic queue submit failed")
            if (!call.response.isCommitted) respondError(call, e)
        }
    }

    private suspend fun processGeneration(
        call: ApplicationCall,
        request: AnthropicMessagesRequest,
        traceId: String
    ) {
        try {
            val llmSession = getLlmSession()
            if (llmSession == null) {
                throw GenerationException("No active LLM session available. Please ensure a model is loaded.", 503)
            }

            val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)
            val unifiedHistory = messageTransformer.convertToUnifiedMnnHistory(
                openAiRequest.messages,
                imageProcessor,
                llmSession
            )

            if (request.stream == true) {
                handleStreamResponse(call, unifiedHistory, request.model ?: "mnn-local", traceId, llmSession)
            } else {
                handleNonStreamResponse(call, unifiedHistory, request.model ?: "mnn-local", llmSession)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            logger.logError(traceId, e, "Anthropic generation failed")
            if (!call.response.isCommitted) respondError(call, e)
        }
    }

    private suspend fun handleNonStreamResponse(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        model: String,
        llmSession: LlmSession
    ) {
        val response = createResponse(model).nonStream(submitter(llmSession, history))
        call.respondText(response.body, ContentType.Application.Json, HttpStatusCode.fromValue(response.status))
    }

    private suspend fun handleStreamResponse(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        model: String,
        traceId: String,
        llmSession: LlmSession
    ) {
        val response = createResponse(model)
        val cancellation = RequestCancellation.from(currentCoroutineContext())
        call.respond(SSEServerContent(call) {
            heartbeat {
                period = 3.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }
            try {
                response.stream(submitter(llmSession, history), cancellation) { send(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                logger.logWarning(traceId, "Client connection closed during Anthropic SSE send", e)
            }
        })
    }

    private fun createResponse(model: String = "mnn-local") = GenerationResponse(
        GenerationProtocol.ANTHROPIC, "msg_${System.currentTimeMillis()}", model = model
    )

    private fun submitter(llmSession: LlmSession, history: List<android.util.Pair<String, String>>) =
        GenerationSubmitter { onProgress ->
            llmSession.submitFullHistory(history, object : GenerateProgressListener {
                override fun onProgress(progress: String?) = onProgress.invoke(progress)
            })
        }

    private suspend fun respondError(call: ApplicationCall, error: Exception) {
        val response = createResponse().errorResponse(error)
        call.respondText(response.body, ContentType.Application.Json, HttpStatusCode.fromValue(response.status))
    }

    private fun getLlmSession(): LlmSession? {
        return chatSessionProvider.getLlmSession()
    }
}
