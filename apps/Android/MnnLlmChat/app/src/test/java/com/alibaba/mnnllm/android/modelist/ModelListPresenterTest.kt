package com.alibaba.mnnllm.android.modelist

import android.app.Application
import android.os.Looper
import androidx.recyclerview.widget.RecyclerView
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.slot
import io.mockk.unmockkObject
import io.mockk.verify
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
@LooperMode(LooperMode.Mode.PAUSED)
class ModelListPresenterTest {
    private val modelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
    private val downloadListener = slot<DownloadListener>()
    private val refreshedPositions = mutableListOf<Int>()
    private val renderedStates = mutableListOf<ModelListItemUiState>()
    private lateinit var application: Application
    private lateinit var downloadManager: ModelDownloadManager
    private lateinit var downloadInfo: DownloadInfo
    private lateinit var presenter: ModelListPresenter

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        downloadManager = mockk(relaxed = true)
        downloadInfo = DownloadInfo(downloadState = DownloadState.PREPARING)
        every { downloadManager.getDownloadedFile(modelId) } returns null
        every { downloadManager.getDownloadInfo(modelId) } returns downloadInfo
        every { downloadManager.addListener(capture(downloadListener)) } returns Unit
        mockkObject(ModelDownloadManager)
        every { ModelDownloadManager.getInstance(application) } returns downloadManager

        val wrapper = ModelItemWrapper(
            modelItem = ModelItem().apply {
                modelId = this@ModelListPresenterTest.modelId
                modelName = "Qwen3.5-0.8B-MNN"
            },
            hasUpdate = true
        )
        val otherWrapper = ModelItemWrapper(ModelItem().apply { modelId = "Builtin/other" })
        val adapter = ModelListAdapter(mutableListOf(otherWrapper, wrapper))
        adapter.registerAdapterDataObserver(object : RecyclerView.AdapterDataObserver() {
            override fun onItemRangeChanged(positionStart: Int, itemCount: Int, payload: Any?) {
                assertEquals(Looper.getMainLooper(), Looper.myLooper())
                assertEquals(1, itemCount)
                refreshedPositions.add(positionStart)
                renderedStates.add(
                    ModelListItemUiStateFactory.fromModelWrapper(application, wrapper, downloadManager)
                )
            }
        })
        val view = mockk<ModelListContract.View>(relaxed = true)
        every { view.adapter } returns adapter
        presenter = ModelListPresenter(application, view)
    }

    @After
    fun tearDown() {
        if (::presenter.isInitialized) presenter.onDestroy()
        unmockkObject(ModelDownloadManager)
    }

    @Test
    fun `download start then failure refreshes item on main thread and enables retry`() {
        downloadListener.captured.onDownloadStart(modelId)
        assertTrue(renderedStates.isEmpty())
        shadowOf(Looper.getMainLooper()).idle()
        assertEquals(listOf(1), refreshedPositions)
        assertFalse(renderedStates.single().updateButtonEnabled)
        assertEquals(application.getString(R.string.download_pending), renderedStates.single().updateButtonText)

        downloadInfo.downloadState = DownloadState.FAILED
        downloadListener.captured.onDownloadFailed(modelId, IllegalStateException("Download failed"))
        assertEquals(1, renderedStates.size)
        shadowOf(Looper.getMainLooper()).idle()

        assertEquals(listOf(1, 1), refreshedPositions)
        assertTrue(renderedStates.last().updateButtonVisible)
        assertTrue(renderedStates.last().updateButtonEnabled)
        assertEquals(application.getString(R.string.download_resume), renderedStates.last().updateButtonText)
        assertEquals(application.getString(R.string.download_state_failed), renderedStates.last().statusText)
    }

    @Test
    fun `download failure after destroy does not refresh UI`() {
        presenter.onDestroy()
        verify(exactly = 1) { downloadManager.removeListener(presenter) }

        downloadInfo.downloadState = DownloadState.FAILED
        downloadListener.captured.onDownloadFailed(modelId, IllegalStateException("Download failed"))
        shadowOf(Looper.getMainLooper()).idle()

        assertTrue(refreshedPositions.isEmpty())
        assertTrue(renderedStates.isEmpty())
    }

    @Test
    fun `destroy removes queued download failure refresh`() {
        downloadListener.captured.onDownloadStart(modelId)
        downloadInfo.downloadState = DownloadState.FAILED
        downloadListener.captured.onDownloadFailed(modelId, IllegalStateException("Download failed"))
        assertTrue(renderedStates.isEmpty())

        presenter.onDestroy()
        shadowOf(Looper.getMainLooper()).idle()

        assertTrue(refreshedPositions.isEmpty())
        assertTrue(renderedStates.isEmpty())
    }

    @Test
    fun `shouldCheckForUpdate returns true for remote source prefixes`() {
        assertTrue(ModelListPresenter.shouldCheckForUpdate("ModelScope/MNN/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("HuggingFace/taobao-mnn/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("Huggingface/taobao-mnn/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("Modelers/MNN/Qwen3.5-0.8B-MNN"))
    }

    @Test
    fun `shouldCheckForUpdate returns false for local and unknown ids`() {
        assertFalse(ModelListPresenter.shouldCheckForUpdate("local//data/local/tmp/mnn_models/Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate("Builtin/Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate("Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate(null))
        assertFalse(ModelListPresenter.shouldCheckForUpdate(""))
    }
}
