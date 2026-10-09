package com.alibaba.mnnllm.android.chat.input

import android.app.Application
import android.content.ClipData
import android.content.Intent
import android.net.Uri
import com.alibaba.mnnllm.android.utils.FileUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.io.IOException
import java.util.concurrent.CancellationException

/** Exercises the production import boundary without constructing native-backed ChatActivity. */
@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE, application = Application::class, sdk = [28])
class AttachmentImportTest {
    private val source = Uri.parse("content://attachment-test/new")
    private val existing = Uri.parse("content://attachment-test/existing")
    private val copied = Uri.parse("content://attachment-test/copied")

    private inner class ImportState {
        val preview = mutableListOf(existing)
        val failures = mutableListOf<Exception>()
        val copiedSources = mutableListOf<Uri>()
        var pickedCalls = 0

        fun import(
            data: Intent?,
            sessionId: String? = "test-session",
            multiple: Boolean = false,
            copy: (Uri, String) -> Uri = { _, _ -> copied }
        ) {
            importAttachmentResult(
                data, sessionId, multiple,
                copy = { uri, session ->
                    copiedSources.add(uri)
                    copy(uri, session)
                },
                onImported = { uris ->
                    if (!multiple) preview.clear()
                    preview.addAll(uris)
                    pickedCalls++
                },
                onFailure = { failures.add(it) }
            )
        }

        fun assertUnchanged() {
            assertEquals(listOf(existing), preview)
            assertEquals(0, pickedCalls)
        }
    }

    @Test
    fun `null intent fails for single media and multiple images without publishing`() {
        for (multiple in listOf(false, true)) {
            val state = ImportState()
            state.import(null, multiple = multiple)
            state.assertUnchanged()
            assertTrue(state.copiedSources.isEmpty())
            assertTrue(state.failures.single() is IOException)
        }
    }

    @Test
    fun `missing or empty uri fails without copying or publishing`() {
        for (multiple in listOf(false, true)) {
            for (data in listOf(Intent(), Intent().setData(Uri.EMPTY))) {
                val state = ImportState()
                state.import(data, multiple = multiple)
                state.assertUnchanged()
                assertTrue(state.copiedSources.isEmpty())
                assertTrue(state.failures.single() is IOException)
            }
        }
    }

    @Test
    fun `missing or blank session fails even with a valid uri`() {
        for (multiple in listOf(false, true)) {
            for (sessionId in listOf(null, "", " ")) {
                val state = ImportState()
                state.import(Intent().setData(source), sessionId, multiple)
                state.assertUnchanged()
                assertTrue(state.copiedSources.isEmpty())
                assertTrue(state.failures.single() is IOException)
            }
        }
    }

    @Test
    fun `text only clip item fails without a null uri escaping`() {
        val state = ImportState()
        state.import(Intent().apply { clipData = ClipData.newPlainText("test", "not a URI") }, multiple = true)
        state.assertUnchanged()
        assertTrue(state.copiedSources.isEmpty())
        assertTrue(state.failures.single() is IOException)
    }

    @Test
    fun `provider null stream fails without replacing existing preview`() {
        val state = ImportState()
        state.import(Intent().setData(source)) { _, _ ->
            // Opening fails before a destination is even considered; no filesystem access.
            val destination = File(System.getProperty("user.dir"), "build/attachment-copy-tests/null-stream.bin")
            FileUtils.copyInputStreamToPath(destination.absolutePath) { null }
            fail("A null stream must not count as an imported attachment")
            copied
        }
        state.assertUnchanged()
        assertTrue(state.failures.single() is IOException)
    }

    @Test
    fun `io and security failures keep existing preview and do not call picked`() {
        for (multiple in listOf(false, true)) {
            for (error in listOf(IOException("provider unavailable"), SecurityException("permission revoked"))) {
                val state = ImportState()
                state.import(Intent().setData(source), multiple = multiple) { _, _ -> throw error }
                state.assertUnchanged()
                assertEquals(listOf(source), state.copiedSources)
                assertSame(error, state.failures.single())
            }
        }
    }

    @Test
    fun `successful single audio or video import publishes the copied uri once`() {
        val state = ImportState()
        state.import(Intent().setData(source)) { uri, sessionId ->
            assertEquals(source, uri)
            assertEquals("test-session", sessionId)
            state.assertUnchanged()
            copied
        }
        assertEquals(listOf(copied), state.preview)
        assertEquals(1, state.pickedCalls)
        assertTrue(state.failures.isEmpty())
    }

    @Test
    fun `single image data appends to existing preview`() {
        val state = ImportState()
        state.import(Intent().setData(source), multiple = true)
        assertEquals(listOf(existing, copied), state.preview)
        assertEquals(1, state.pickedCalls)
        assertTrue(state.failures.isEmpty())
    }

    @Test
    fun `partial images retain successes continue after failures and report failure once`() {
        val denied = Uri.parse("content://attachment-test/denied")
        val broken = Uri.parse("content://attachment-test/broken")
        val last = Uri.parse("content://attachment-test/last")
        val data = Intent().apply {
            clipData = ClipData.newRawUri("test", source).apply {
                addItem(ClipData.Item(denied))
                addItem(ClipData.Item("missing URI"))
                addItem(ClipData.Item(broken))
                addItem(ClipData.Item(last))
            }
        }
        val state = ImportState()
        state.import(data, multiple = true) { uri, _ ->
            state.assertUnchanged()
            when (uri) {
                denied -> throw SecurityException("permission revoked")
                broken -> throw IOException("provider read failed")
                source -> copied
                else -> last
            }
        }
        assertEquals(listOf(source, denied, broken, last), state.copiedSources)
        assertEquals(listOf(existing, copied, last), state.preview)
        assertEquals(1, state.pickedCalls)
        assertEquals(1, state.failures.size)
    }

    @Test
    fun `all images failing never publishes an empty replacement`() {
        val data = Intent().apply {
            clipData = ClipData.newRawUri("test", source).apply { addItem(ClipData.Item(source)) }
        }
        val state = ImportState()
        state.import(data, multiple = true) { _, _ -> throw IOException("unavailable") }
        state.assertUnchanged()
        assertEquals(2, state.copiedSources.size)
        assertEquals(1, state.failures.size)
    }

    @Test
    fun `cancellation and programming errors escape instead of becoming import failures`() {
        for (error in listOf(CancellationException("cancelled"), IllegalStateException("bug"))) {
            val state = ImportState()
            try {
                state.import(Intent().setData(source)) { _, _ -> throw error }
                fail("Expected the original exception")
            } catch (actual: RuntimeException) {
                assertSame(error, actual)
            }
            state.assertUnchanged()
            assertTrue(state.failures.isEmpty())
        }
    }

    @Test
    fun `exceptions in publication are not misreported as copy failures`() {
        val error = IOException("callback failure")
        val failures = mutableListOf<Exception>()
        try {
            importAttachmentResult(
                Intent().setData(source), "test-session", false,
                copy = { _, _ -> copied },
                onImported = { throw error },
                onFailure = { failures.add(it) }
            )
            fail("Expected callback failure to escape")
        } catch (actual: IOException) {
            assertSame(error, actual)
        }
        assertTrue(failures.isEmpty())
    }
}
