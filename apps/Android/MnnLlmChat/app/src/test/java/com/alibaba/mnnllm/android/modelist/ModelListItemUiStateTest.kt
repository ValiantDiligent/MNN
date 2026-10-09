package com.alibaba.mnnllm.android.modelist

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import io.mockk.every
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class ModelListItemUiStateTest {
    private val modelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
    private lateinit var application: Application
    private lateinit var downloadManager: ModelDownloadManager
    private lateinit var downloadInfo: DownloadInfo
    private lateinit var wrapper: ModelItemWrapper

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        downloadManager = mockk()
        downloadInfo = DownloadInfo(downloadState = DownloadState.FAILED)
        every { downloadManager.getDownloadedFile(modelId) } returns null
        every { downloadManager.getDownloadInfo(modelId) } returns downloadInfo
        wrapper = ModelItemWrapper(
            modelItem = ModelItem().apply {
                modelId = this@ModelListItemUiStateTest.modelId
                modelName = "Qwen3.5-0.8B-MNN"
            },
            downloadSize = 1024L,
            hasUpdate = true
        )
    }

    @Test
    fun `failed update shows localized failure and enabled retry without exposing error details`() {
        val privateUri = "https://example.invalid/model?token=private-download-token"
        downloadInfo.errorMessage = "Unable to download $privateUri"
        downloadInfo.errorException = IllegalStateException("Request failed: $privateUri")

        val state = uiState()

        assertTrue(state.updateButtonVisible)
        assertTrue(state.updateButtonEnabled)
        assertEquals(application.getString(R.string.download_resume), state.updateButtonText)
        assertEquals(
            "${wrapper.formattedSize} (${application.getString(R.string.download_state_failed)})",
            state.statusText
        )
        assertFalse(state.statusText.contains(privateUri))
        assertFalse(state.statusText.contains("private-download-token"))
        assertFalse(state.updateButtonText.contains("private-download-token"))
    }

    @Test
    fun `failed update without known size still shows failure and retry`() {
        val state = uiState(wrapper.copy(downloadSize = 0L))

        assertEquals(application.getString(R.string.download_state_failed), state.statusText)
        assertTrue(state.updateButtonVisible)
        assertTrue(state.updateButtonEnabled)
        assertEquals(application.getString(R.string.download_resume), state.updateButtonText)
    }

    @Test
    fun `retry preparing and downloading disable update button again`() {
        assertTrue(uiState().updateButtonEnabled)

        downloadInfo.downloadState = DownloadState.PREPARING
        val preparing = uiState()
        assertTrue(preparing.updateButtonVisible)
        assertFalse(preparing.updateButtonEnabled)
        assertEquals(application.getString(R.string.download_pending), preparing.updateButtonText)
        assertEquals(
            "${wrapper.formattedSize} (${application.getString(R.string.download_preparing)})",
            preparing.statusText
        )

        downloadInfo.downloadState = DownloadState.DOWNLOADING
        val downloading = uiState()
        assertTrue(downloading.updateButtonVisible)
        assertFalse(downloading.updateButtonEnabled)
        assertEquals(application.getString(R.string.download_state_updating), downloading.updateButtonText)
        assertEquals(
            "${wrapper.formattedSize} (${application.getString(R.string.download_state_updating)})",
            downloading.statusText
        )
    }

    @Test
    fun `available update without failure retains update action`() {
        downloadInfo.downloadState = DownloadState.COMPLETED
        val state = uiState()

        assertTrue(state.updateButtonVisible)
        assertTrue(state.updateButtonEnabled)
        assertEquals(application.getString(R.string.update), state.updateButtonText)
        assertEquals(
            application.getString(R.string.downloaded_update_available, wrapper.formattedSize),
            state.statusText
        )
    }

    @Test
    fun `failure without update keeps downloaded row and hides retry`() {
        val state = uiState(wrapper.copy(hasUpdate = false))

        assertFalse(state.updateButtonVisible)
        assertTrue(state.updateButtonEnabled)
        assertEquals("", state.updateButtonText)
        assertEquals(
            application.getString(R.string.downloaded_click_to_chat, wrapper.formattedSize),
            state.statusText
        )
    }

    private fun uiState(modelWrapper: ModelItemWrapper = wrapper): ModelListItemUiState =
        ModelListItemUiStateFactory.fromModelWrapper(application, modelWrapper, downloadManager)
}
