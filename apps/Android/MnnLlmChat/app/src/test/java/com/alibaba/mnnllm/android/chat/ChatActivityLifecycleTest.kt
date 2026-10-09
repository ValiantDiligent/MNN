package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ChatActivityLifecycleTest {

    @Test
    @Config(sdk = [34], application = android.app.Application::class)
    fun leavingChatDoesNotStopIndependentApiService() {
        val manager = com.alibaba.mnnllm.api.openai.manager.ApiServiceManager
        io.mockk.mockkObject(manager)
        try {
            io.mockk.every { manager.stopApiService(any()) } returns true
            val intent = Intent(ApplicationProvider.getApplicationContext(), ChatActivity::class.java)
            Robolectric.buildActivity(ChatActivity::class.java, intent).create().destroy()
            // Also catches an unscoped MainScope stop posted by the old onDestroy implementation.
            org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idle()
            io.mockk.verify(exactly = 0) { manager.stopApiService(any()) }
        } finally {
            io.mockk.unmockkObject(manager)
        }
    }

    @Test
    fun destroy_withoutModelExtras_doesNotCrash() {
        val intent = Intent(ApplicationProvider.getApplicationContext(), ChatActivity::class.java)

        Robolectric.buildActivity(ChatActivity::class.java, intent)
            .create()
            .destroy()
    }
}
