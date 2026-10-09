package com.alibaba.mnnllm.android.llm

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.*
import org.junit.Test

class ChatServiceRegistryTest {
    private class Session(override var sessionId: String) : ChatSession {
        override val debugInfo = ""
        override val supportOmni = false
        override fun load() = Unit
        override fun reset(): String = "changed".also { sessionId = it }
        override fun release() = Unit
        override fun generate(prompt: String, params: Map<String, Any>, progressListener: GenerateProgressListener) =
            hashMapOf<String, Any>()
        override fun setKeepHistory(keepHistory: Boolean) = Unit
        override fun setEnableAudioOutput(enable: Boolean) = Unit
        override fun getHistory(): List<ChatDataItem>? = null
        override fun setHistory(history: List<ChatDataItem>?) = Unit
        override fun updateThinking(thinking: Boolean) = Unit
    }

    @Suppress("UNCHECKED_CAST")
    private fun registry(service: ChatService, name: String): MutableMap<String, ChatSession> =
        ChatService::class.java.getDeclaredField(name).let {
            it.isAccessible = true
            it.get(service) as MutableMap<String, ChatSession>
        }

    @Test
    fun resetDoesNotLeaveOriginalRegistrationOrRemoveAnotherSession() {
        val service = ChatService()
        val old = Session("original")
        val other = Session("changed")
        registry(service, "diffusionSessionMap")["original"] = old
        registry(service, "transformerSessionMap")["changed"] = other
        old.reset()
        service.removeSession(old)
        assertNull(service.getSession("original"))
        assertSame(other, service.getSession("changed"))
        service.removeSession(old)
        assertSame(other, service.getSession("changed"))
    }

    @Test
    fun repeatedLateCleanupDoesNotRemoveReplacementAtSameRegistrationKey() {
        val service = ChatService()
        val old = Session("shared-key")
        val replacement = Session("shared-key")
        val sessions = registry(service, "transformerSessionMap")
        sessions["shared-key"] = old
        sessions["shared-key"] = replacement
        service.removeSession(old)
        service.removeSession(old)
        assertSame(replacement, service.getSession("shared-key"))
        service.removeSession(replacement)
        assertNull(service.getSession("shared-key"))
    }
}
