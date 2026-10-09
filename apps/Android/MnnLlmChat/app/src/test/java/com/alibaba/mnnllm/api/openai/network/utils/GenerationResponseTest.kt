package com.alibaba.mnnllm.api.openai.network.utils

import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized

/** Exercises the same injectable submit path used by both production transports, without loading JNI. */
@RunWith(Parameterized::class)
class GenerationResponseTest(private val protocolName: String, private val stream: Boolean, private val scenario: String) {
    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0} stream={1} {2}")
        fun cases(): List<Array<Any>> = listOf("OPENAI", "ANTHROPIC").flatMap { protocol ->
            listOf(false, true).flatMap { stream ->
                listOf("emptyError", "partialError", "nullThenError", "emptyException", "partialException",
                    "success", "emptySuccess", "successWithoutNull").map { arrayOf<Any>(protocol, stream, it) }
            }
        }
    }

    @Test
    fun nativeResultControlsWireResponse() = runBlocking {
        val protocol = GenerationProtocol.valueOf(protocolName)
        val response = GenerationResponse(protocol, "response-id", 123, "model\"\tname")
        val text = "partial \"text\"\n\r\t\b\u0001"
        val reason = "native failure: \"decode\"\n\r\t\u0001"
        val success = scenario.startsWith("success") || scenario == "emptySuccess"
        val hasText = scenario.startsWith("partial") || scenario.startsWith("success") || scenario == "nullThenError"
        val submitter = GenerationSubmitter { progress ->
            if (hasText) assertFalse(progress(text))
            if (scenario == "nullThenError" || (success && scenario != "successWithoutNull")) {
                assertFalse(progress(null))
            }
            if (scenario.endsWith("Exception")) throw IllegalStateException(reason)
            if (success) {
                hashMapOf("error" to false, "prompt_len" to 7L, "decode_len" to 3L)
            } else {
                object : HashMap<String, Any>() {
                    override fun get(key: String): Any? {
                        check(key != "prompt_len" && key != "decode_len") { "Read metrics before checking native error" }
                        return super.get(key)
                    }
                }.apply { put("error", true); put("message", reason) }
            }
        }
        if (stream) {
            val events = mutableListOf<ServerSentEvent>()
            response.stream(submitter, RequestCancellation.from(currentCoroutineContext())) { events.add(it) }
            val documents = events.filter { it.data != "[DONE]" }.map { Json.parseToJsonElement(it.data!!).jsonObject }
            val errors = events.filter { it.event == "error" }
            if (success) {
                assertTrue(errors.isEmpty())
                if (protocol == GenerationProtocol.OPENAI) {
                    assertEquals("[DONE]", events.last().data)
                    val finish = documents.last()
                    assertEquals("stop", finish["choices"]!!.jsonArray[0].jsonObject["finish_reason"]!!.jsonPrimitive.content)
                    assertEquals("10", finish["usage"]!!.jsonObject["total_tokens"]!!.jsonPrimitive.content)
                } else {
                    assertEquals(listOf("content_block_stop", "message_delta", "message_stop"), events.takeLast(3).map { it.event })
                    val finish = documents[documents.lastIndex - 1]
                    assertEquals("end_turn", finish["delta"]!!.jsonObject["stop_reason"]!!.jsonPrimitive.content)
                    assertEquals("3", finish["usage"]!!.jsonObject["output_tokens"]!!.jsonPrimitive.content)
                }
            } else {
                assertEquals(1, errors.size)
                assertEquals(errors.single(), events.last())
                val error = Json.parseToJsonElement(errors.single().data!!).jsonObject
                assertEquals(reason, error["error"]!!.jsonObject["message"]!!.jsonPrimitive.content)
                assertEquals(if (protocol == GenerationProtocol.OPENAI) "server_error" else "api_error",
                    error["error"]!!.jsonObject["type"]!!.jsonPrimitive.content)
                assertTrue(events.none { it.data == "[DONE]" || it.event in listOf("message_stop", "message_delta", "content_block_stop") })
                assertTrue(documents.none { doc -> doc["choices"]?.jsonArray?.any {
                    it.jsonObject["finish_reason"]?.jsonPrimitive?.content == "stop"
                } == true })
            }
            val deltas = if (protocol == GenerationProtocol.OPENAI) {
                documents.mapNotNull { it["choices"]?.jsonArray?.get(0)?.jsonObject?.get("delta")
                    ?.jsonObject?.get("content")?.jsonPrimitive?.content }.filter { it.isNotEmpty() }
            } else {
                documents.filter { it["type"]?.jsonPrimitive?.content == "content_block_delta" }
                    .map { it["delta"]!!.jsonObject["text"]!!.jsonPrimitive.content }
            }
            assertEquals(if (hasText) listOf(text) else emptyList<String>(), deltas)
        } else {
            val result = response.nonStream(submitter)
            val body = Json.parseToJsonElement(result.body).jsonObject
            if (success) {
                assertEquals(200, result.status)
                assertFalse(body.containsKey("error"))
                if (protocol == GenerationProtocol.OPENAI) {
                    val choice = body["choices"]!!.jsonArray[0].jsonObject
                    assertEquals("stop", choice["finish_reason"]!!.jsonPrimitive.content)
                    assertEquals(if (hasText) text else "", choice["message"]!!.jsonObject["content"]!!.jsonPrimitive.content)
                    assertEquals("10", body["usage"]!!.jsonObject["total_tokens"]!!.jsonPrimitive.content)
                } else {
                    // The adapter's default-field serialization contract is unchanged.
                    assertEquals(if (hasText) text else "", body["content"]!!.jsonArray[0].jsonObject["text"]!!.jsonPrimitive.content)
                    assertEquals("3", body["usage"]!!.jsonObject["output_tokens"]!!.jsonPrimitive.content)
                }
            } else {
                assertEquals(500, result.status)
                assertEquals(reason, body["error"]!!.jsonObject["message"]!!.jsonPrimitive.content)
                assertFalse(body.containsKey("choices"))
                assertFalse(body.containsKey("content"))
                assertFalse(body.containsKey("usage"))
                assertFalse(body.containsKey("stop_reason"))
            }
        }
    }
}
