package com.alibaba.mnnllm.api.openai.service

import android.content.ComponentName
import android.content.Context
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class OpenAIServiceLifecycleTest {

    @After
    fun tearDown() {
    }

    @Test
    @Config(sdk = [34], application = android.app.Application::class)
    fun lateOldServiceDestroyDoesNotClearReplacementModelOrReleaseItsSession() {
        val runtime = mockk<com.alibaba.mnnllm.api.openai.runtime.LlmRuntimeController>(relaxed = true)
        val locator = com.alibaba.mnnllm.api.openai.di.ServiceLocator
        val models = com.alibaba.mnnllm.api.openai.manager.CurrentModelManager
        locator.setLlmRuntimeController(runtime)
        val old = org.robolectric.Robolectric.buildService(OpenAIService::class.java).create()
        val replacement = org.robolectric.Robolectric.buildService(OpenAIService::class.java).create()
        try {
            models.setCurrentModelId("new-model")
            old.destroy()
            org.junit.Assert.assertSame(replacement.get(), OpenAIService.getInstance())
            org.junit.Assert.assertEquals("new-model", models.getCurrentModelId())
            verify(exactly = 0) { runtime.releaseSession() }
            verify(exactly = 0) { runtime.releaseSession(any()) }
        } finally {
            replacement.destroy()
            locator.reset()
        }
        org.junit.Assert.assertNull(models.getCurrentModelId())
        org.junit.Assert.assertNull(OpenAIService.getInstance())
    }

    @Test
    fun startService_doesNotBindService() {
        val context = mockk<Context>(relaxed = true)
        every { context.startForegroundService(any()) } returns mockk<ComponentName>(relaxed = true)

        OpenAIService.startService(context, "ModelScope/MNN/Qwen3.5-0.8B-MNN")

        verify(exactly = 1) { context.startForegroundService(any()) }
        verify(exactly = 0) {
            context.bindService(any<android.content.Intent>(), any(), any<Int>())
        }
    }
}
