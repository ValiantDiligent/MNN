package com.alibaba.mls.api.download

import android.util.Log
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.HfFileMetadata
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.MockedStatic
import org.mockito.Mockito.mockStatic
import java.io.Closeable
import java.io.File
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.security.MessageDigest
import java.util.Collections
import java.util.Locale
import java.util.concurrent.CancellationException

/** Real OkHttp + loopback HTTP; only Android logging is mocked. All fixture files are retained. */
class ModelFileDownloaderTest {
    private lateinit var log: MockedStatic<Log>
    private lateinit var directory: File
    private lateinit var server: LocalServer
    private val downloader = ModelFileDownloader()
    private val listener = object : ModelFileDownloader.FileDownloadListener {
        override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long) = false
    }

    @Before
    fun setUp() {
        log = mockStatic(Log::class.java)
        val root = File(System.getProperty(
            "mnn.downloader.testRoot", File("build/downloader-protocol-test-data").absolutePath
        )).absoluteFile
        Files.createDirectories(root.toPath())
        directory = Files.createTempDirectory(root.toPath(), "case-").toFile()
        println("Retained downloader fixture: ${directory.absolutePath}")
        server = LocalServer()
    }

    @After
    fun tearDown() {
        try {
            if (::server.isInitialized) server.close()
        } finally {
            if (::log.isInitialized) log.close()
        }
        // Deliberately no delete/deleteOnExit/TemporaryFolder: retain all test data.
    }

    @Test
    fun full200WithoutEtagPublishesExactBytes() {
        val task = task()
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(1, server.requests.size)
        assertEquals("GET", server.requests.single().method)
        assertNull(server.requests.single().headers["range"])
        assertEquals("identity", server.requests.single().headers["accept-encoding"])
    }

    @Test
    fun ignoredRange200RestartsAtZeroInsteadOfAppending() {
        val task = task(partial = "ABC", cachedSize = 3)
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals("bytes=3-", server.requests.single().headers["range"])
    }

    @Test
    fun correct206UsesActualLengthNotStaleCachedSize() {
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        val task = task(partial = "ABC", cachedSize = 6)
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals("bytes=3-", server.requests.single().headers["range"])
    }

    @Test
    fun missingTempIgnoresCachedCompletedSize() {
        val task = task(cachedSize = 6)
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertNull(server.requests.single().headers["range"])
    }

    @Test
    fun correct206CanStartAtZero() {
        server.reply = { _, _ -> Reply(206, "ABCDEF", mapOf("Content-Range" to "bytes 0-5/6")) }
        val task = task()
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
    }

    @Test
    fun validSubrangesAreCompletedBeforePublication() {
        val task = task(partial = "ABC")
        server.reply = { _, index ->
            assertNotPublished(task)
            if (index == 0) Reply(206, "D", mapOf("Content-Range" to "bytes 3-3/6"))
            else Reply(206, "EF", mapOf("Content-Range" to "bytes 4-5/6"))
        }
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf("bytes=3-", "bytes=4-"), server.requests.map { it.headers["range"] })
    }

    @Test fun wrongRangeStartIsNotRetriedOrWritten() = rejectRange("bytes 2-4/6")
    @Test fun wrongRangeTotalIsNotRetriedOrWritten() = rejectRange("bytes 3-5/7")
    @Test fun endOutsideTotalIsNotRetriedOrWritten() = rejectRange("bytes 3-6/6")
    @Test fun reversedRangeIsNotRetriedOrWritten() = rejectRange("bytes 3-2/6")
    @Test fun endDisagreesWithContentLength() = rejectRange("bytes 3-4/6")
    @Test fun unknownRangeTotalIsRejected() = rejectRange("bytes 3-5/*")
    @Test fun overflowingRangeIsRejected() = rejectRange("bytes 3-5/9223372036854775808")
    @Test fun missingContentRangeIsRejected() = rejectRange(null)
    @Test fun malformedContentRangeIsRejected() = rejectRange("not-a-range")

    private fun rejectRange(range: String?) {
        server.reply = { _, _ -> Reply(206, "DEF", range?.let { mapOf("Content-Range" to it) } ?: emptyMap()) }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun response416NeverAppendsErrorBody() {
        server.reply = { _, _ -> Reply(416, "ERR", mapOf("Content-Range" to "bytes */6")) }
        val task = task(partial = "ABC", cachedSize = 3)
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun response416WithDifferentServerTotalStillFails() {
        server.reply = { _, _ -> Reply(416, "ERR", mapOf("Content-Range" to "bytes */3")) }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun shortContentLengthFailsBeforeWriting() {
        server.reply = { _, _ -> Reply(text = "ABC") }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertFalse(task.blobPathIncomplete!!.exists())
        assertEquals(1, server.requests.size)
    }

    @Test
    fun longContentLengthFailsBeforeWriting() {
        server.reply = { _, _ -> Reply(text = "ABCDEFG") }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun shortChunkedBodyAtNormalEofNeverPublishesAndExhaustsRetryBudget() {
        server.reply = { _, _ -> Reply(text = "ABC", chunked = true) }
        val task = task()
        val error = assertThrows(IOException::class.java) { downloader.downloadFile(task, listener) }
        assertFalse(error is DownloadProtocolException)
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(10, server.requests.size)
    }

    @Test
    fun shortChunkedBodyCanResumeOnNextAttempt() {
        server.reply = { _, index ->
            if (index == 0) Reply(text = "ABC", chunked = true)
            else Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true)
        }
        val task = task()
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf(null, "bytes=3-"), server.requests.map { it.headers["range"] })
    }

    @Test
    fun interruptedFixedLengthConnectionIsRetriedFromActualLength() {
        server.reply = { _, index ->
            if (index == 0) Reply(text = "ABC", declaredLength = 6)
            else Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"))
        }
        val task = task()
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf(null, "bytes=3-"), server.requests.map { it.headers["range"] })
    }

    @Test
    fun incompleteChunkFramingDoesNotLeaveACompletedNoEtagFile() {
        val task = task()
        server.reply = { _, index ->
            if (index == 0) Reply(chunked = true, terminateChunks = false)
            else {
                assertNotPublished(task)
                assertTemp(task, "")
                Reply()
            }
        }
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf(null, null), server.requests.map { it.headers["range"] })
    }

    @Test
    fun short206CannotBeMistakenForFinishedInterval() {
        server.reply = { _, index ->
            if (index == 0) Reply(206, "D", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true)
            else Reply(206, "EF", mapOf("Content-Range" to "bytes 4-5/6"), chunked = true)
        }
        val task = task(partial = "ABC")
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf("bytes=3-", "bytes=4-"), server.requests.map { it.headers["range"] })
    }

    @Test
    fun longChunked200DoesNotWritePastMetadataOrLeavePublishablePrefix() {
        server.reply = { _, _ -> Reply(text = "ABCDEFG", chunked = true) }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun extraChunkAfterExactLengthIsRejectedAndRolledBack() {
        server.reply = { _, _ -> Reply(text = "ABCDEF", chunked = true, extraChunk = "G") }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun longChunked206PreservesOnlyPreviouslyVerifiedPrefix() {
        server.reply = { _, _ ->
            Reply(206, "DEFG", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true)
        }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun correctQuotedSha256Publishes() {
        val task = task(etag = "\"${digest("ABCDEF")}\"")
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
    }

    @Test
    fun correctGitBlobSha1Publishes() {
        val task = task(etag = digest("blob 6\u0000ABCDEF", "SHA-1"))
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
    }

    @Test
    fun checksumFailureRetriesFreshThenPublishesOnlyValidBytes() {
        val task = task(etag = digest("ABCDEF"))
        server.reply = { _, index ->
            assertNotPublished(task)
            Reply(text = if (index == 0) "XXXXXX" else "ABCDEF")
        }
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf(null, null), server.requests.map { it.headers["range"] })
    }

    @Test
    fun persistentChecksumFailureThrowsWithoutSymlinkOrHashingDeletedFile() {
        val task = task(etag = digest("XXXXXX"))
        assertThrows(DownloadIntegrityException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(10, server.requests.size)
        assertTrue(server.requests.all { it.headers["range"] == null })
    }

    @Test
    fun invalidChecksumMetadataIsNotRetried() {
        val task = task(etag = "invalid-hash")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABCDEF")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun exactLocalLengthWithoutEtagMustRefetchDespiteStaleCounter() {
        val task = task(partial = "ABCDEF", cachedSize = 0)
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(1, server.requests.size)
        assertNull(server.requests.single().headers["range"])
    }

    @Test
    fun completeLocalFileWithWrongHashRestartsBeforePublication() {
        val task = task(partial = "XXXXXX", etag = digest("ABCDEF"))
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertNull(server.requests.single().headers["range"])
    }

    @Test
    fun oversizedLocalFileWithoutEtagIsResetNotPublished() {
        val task = task(partial = "ABCABCDEF", cachedSize = 6)
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertNull(server.requests.single().headers["range"])
    }

    @Test
    fun zeroMetadataAndEmptyBodyProduceRealEmptyFile() {
        server.reply = { _, _ -> Reply(text = "") }
        val task = task(size = 0, cachedSize = 30, etag = digest(""))
        downloader.downloadFile(task, listener)
        assertPublished(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun zeroMetadataRejectsNonemptyChunkedBody() {
        server.reply = { _, _ -> Reply(text = "ABC", chunked = true) }
        val task = task(size = 0)
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun zeroMetadataWithStaleNonemptyTempResetsSafely() {
        server.reply = { _, _ -> Reply(text = "") }
        val task = task(size = 0, partial = "ABC")
        downloader.downloadFile(task, listener)
        assertPublished(task, "")
        assertNull(server.requests.single().headers["range"])
    }

    @Test
    fun negativeMetadataIsRejectedWithoutNetworkOrPublication() {
        val task = task(size = -1)
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertEquals(0, server.requests.size)
    }

    @Test
    fun relativeRedirectUsesOkHttpResolutionAndRetainsRange() {
        server.reply = { request, _ ->
            if (request.path == "/model") Reply(302, "", mapOf("Location" to "nested/../weights?raw=1"))
            else Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"))
        }
        val task = task(partial = "ABC")
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(listOf("/model", "/weights?raw=1"), server.requests.map { it.path })
        assertTrue(server.requests.all { it.headers["range"] == "bytes=3-" })
    }

    @Test
    fun redirectWithoutLocationFailsWithoutNpeOrPublication() {
        server.reply = { _, _ -> Reply(302, "") }
        val task = task(partial = "ABC")
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun invalidRedirectLocationFailsWithoutNpe() {
        server.reply = { _, _ -> Reply(307, "", mapOf("Location" to "http://[invalid")) }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertEquals(1, server.requests.size)
    }

    @Test
    fun redirectLoopIsBoundedWithoutRetryingProtocolError() {
        server.reply = { _, _ -> Reply(308, "", mapOf("Location" to "/model")) }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertEquals(11, server.requests.size)
    }

    @Test
    fun status304IsNotTreatedAsRedirect() {
        server.reply = { _, _ -> Reply(304, "") }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertEquals(1, server.requests.size)
    }

    @Test
    fun transientHttpFailureRetriesBut404DoesNot() {
        server.reply = { _, index -> if (index == 0) Reply(503, "ERR") else Reply() }
        val task = task()
        downloader.downloadFile(task, listener)
        assertPublished(task, "ABCDEF")
        assertEquals(2, server.requests.size)
    }

    @Test
    fun permanentHttpFailureIsNotRetried() {
        server.reply = { _, _ -> Reply(404, "ERR") }
        val task = task()
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertFalse(task.blobPathIncomplete!!.exists())
        assertEquals(1, server.requests.size)
    }

    @Test
    fun pauseRetainsPartialBytesAndResumeUsesActualFileLength() {
        val text = "0123456789ABCDEF".repeat(8192)
        server.reply = { request, _ ->
            val offset = request.headers["range"]?.removePrefix("bytes=")?.removeSuffix("-")?.toInt() ?: 0
            if (offset == 0) Reply(text = text)
            else Reply(206, text.substring(offset), mapOf("Content-Range" to "bytes $offset-${text.length - 1}/${text.length}"))
        }
        val task = task(size = text.length.toLong())
        val pause = object : ModelFileDownloader.FileDownloadListener {
            override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long) = true
        }
        assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, pause) }
        assertNotPublished(task)
        val length = task.blobPathIncomplete!!.length().toInt()
        assertTrue(length in 1 until text.length)
        assertTemp(task, text.substring(0, length))
        assertEquals(1, server.requests.size)
        task.downloadedSize = 0
        downloader.downloadFile(task, listener)
        assertPublished(task, text)
        assertEquals("bytes=$length-", server.requests.last().headers["range"])
    }

    @Test
    fun cancellationIsPropagatedWithoutRetriesOrPublication() {
        val task = task()
        val cancelled = CancellationException("test cancellation")
        val cancel = object : ModelFileDownloader.FileDownloadListener {
            override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long): Boolean {
                throw cancelled
            }
        }
        assertSame(cancelled, assertThrows(CancellationException::class.java) { downloader.downloadFile(task, cancel) })
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun interruptedThreadIsCancelledBeforePublishingCompleteTemp() {
        val task = task(partial = "ABCDEF", cachedSize = 6)
        try {
            Thread.currentThread().interrupt()
            assertThrows(CancellationException::class.java) { downloader.downloadFile(task, listener) }
            assertTrue(Thread.currentThread().isInterrupted)
        } finally {
            Thread.interrupted()
        }
        assertNotPublished(task)
        assertTemp(task, "ABCDEF")
        assertEquals(0, server.requests.size)
    }

    @Test
    fun existingValidPointerIsPreservedWithoutDownloading() {
        val task = task()
        task.blobPath!!.writeText("ABCDEF")
        Files.createDirectories(task.pointerPath!!.parentFile!!.toPath())
        Files.createSymbolicLink(task.pointerPath!!.toPath(), task.blobPath!!.toPath())
        downloader.downloadFile(task, listener)
        assertEquals("ABCDEF", task.blobPath!!.readText())
        assertEquals("ABCDEF", task.pointerPath!!.readText())
        assertEquals(0, server.requests.size)
    }

    @Test
    fun invalidExistingPointerFailsWithoutChangingIt() {
        val task = task()
        Files.createDirectories(task.pointerPath!!.parentFile!!.toPath())
        task.pointerPath!!.writeText("OLD")
        assertThrows(FileDownloadException::class.java) { downloader.downloadFile(task, listener) }
        assertEquals("OLD", task.pointerPath!!.readText())
        assertFalse(task.blobPath!!.exists())
        assertEquals(0, server.requests.size)
    }

    @Test
    fun existingValidDestinationIsNeverResetByIgnoredRangeResponse() {
        val task = task(partial = "ABC", cachedSize = 3, etag = digest("ABCDEF"))
        task.blobPath!!.writeText("ABCDEF")
        downloader.downloadFile(task, listener)
        assertEquals("ABCDEF", task.blobPath!!.readText())
        assertEquals("ABCDEF", task.pointerPath!!.readText())
        assertTemp(task, "ABC")
        assertEquals(0, server.requests.size)
    }

    @Test
    fun invalidExistingDestinationIsPreservedAndNotPublished() {
        val task = task()
        task.blobPath!!.writeText("OLD")
        assertThrows(FileDownloadException::class.java) { downloader.downloadFile(task, listener) }
        assertEquals("OLD", task.blobPath!!.readText())
        assertFalse(Files.exists(task.pointerPath!!.toPath(), LinkOption.NOFOLLOW_LINKS))
        assertEquals(0, server.requests.size)
    }

    @Test
    fun temporarySymlinkIsNotTruncated() {
        val task = task()
        val sentinel = File(directory, "sentinel").apply { writeText("UNCHANGED") }
        Files.createSymbolicLink(task.blobPathIncomplete!!.toPath(), sentinel.toPath())
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, listener) }
        assertNotPublished(task)
        assertEquals("UNCHANGED", sentinel.readText())
        assertTrue(Files.isSymbolicLink(task.blobPathIncomplete!!.toPath()))
        assertEquals(0, server.requests.size)
    }

    @Test
    fun destinationAppearingDuringDownloadIsNotOverwritten() {
        val task = task()
        val createDestination = object : ModelFileDownloader.FileDownloadListener {
            override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long): Boolean {
                task.blobPath!!.writeText("UNCHANGED")
                return false
            }
        }
        assertThrows(FileDownloadException::class.java) { downloader.downloadFile(task, createDestination) }
        assertEquals("UNCHANGED", task.blobPath!!.readText())
        assertTemp(task, "ABCDEF")
        assertFalse(Files.exists(task.pointerPath!!.toPath(), LinkOption.NOFOLLOW_LINKS))
        assertEquals(1, server.requests.size)
    }

    @Test
    fun noHashCompleteTempMustSurviveFailedRevalidationWithoutPublishing() {
        val task = task(partial = "ABCDEF", cachedSize = 6)
        val progress = AggregateListener(task)
        server.reply = { _, _ -> Reply(404, "ERR") }
        assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, progress) }
        assertNotPublished(task)
        assertTemp(task, "")
        progress.assertSaved(0)
        server.reply = { _, _ -> Reply() }
        ModelFileDownloader().downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(-6L, 6L), progress.deltas)
        assertEquals(listOf(null, null), server.requests.map { it.headers["range"] })
    }

    @Test
    fun validHashCompleteTempCanPublishOfflineAndSynchronizeStaleCounter() {
        val task = task(partial = "ABCDEF", cachedSize = 3, etag = digest("ABCDEF"))
        val progress = AggregateListener(task)
        server.reply = { _, _ -> throw AssertionError("Verified local hash should not need HTTP") }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(3L), progress.deltas)
        assertTrue(server.requests.isEmpty())
    }

    @Test
    fun preexistingEmptyNoHashTempRequiresHttpEof() {
        val task = task(size = 0, partial = "")
        val progress = AggregateListener(task)
        server.reply = { _, _ -> Reply(text = "") }
        downloader.downloadFile(task, progress)
        assertPublished(task, "")
        progress.assertSaved(0)
        assertEquals(1, server.requests.size)
    }

    @Test
    fun preexistingEmptyHashedTempStillRequiresHttpEofForZeroMetadata() {
        val task = task(size = 0, partial = "", etag = digest(""))
        server.reply = { _, _ -> Reply(text = "") }
        downloader.downloadFile(task, listener)
        assertPublished(task, "")
        assertEquals(1, server.requests.size)
    }

    @Test
    fun zeroMetadataRejectedBodyIsStillRejectedOnNextDownload() {
        val task = task(size = 0)
        val progress = AggregateListener(task)
        server.reply = { _, _ -> Reply(text = "ABC", chunked = true) }
        repeat(2) {
            assertThrows(DownloadProtocolException::class.java) { ModelFileDownloader().downloadFile(task, progress) }
            assertNotPublished(task)
            assertTemp(task, "")
            progress.assertSaved(0)
            assertEquals(it + 1, server.requests.size)
        }
        server.reply = { _, _ -> Reply(text = "", chunked = true) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "")
        assertEquals(3, server.requests.size)
    }

    @Test
    fun unknownSizeRemainsFailClosedAcrossDownloadsWithEmptyTemp() {
        val task = task(size = -1, partial = "")
        repeat(2) {
            assertThrows(DownloadProtocolException::class.java) { ModelFileDownloader().downloadFile(task, listener) }
            assertNotPublished(task)
            assertTemp(task, "")
        }
        assertTrue(server.requests.isEmpty())
    }

    @Test
    fun ignoredRangeCorrectsAggregateBeforeCountingReplacementBody() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task, otherFiles = 11)
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(-3L, 6L), progress.deltas)
    }

    @Test
    fun shaRetrySubtractsRejectedBytesFromAggregate() {
        val task = task(partial = "ABC", cachedSize = 3, etag = digest("ABCDEF"))
        val progress = AggregateListener(task)
        server.reply = { _, index ->
            if (index == 0) Reply(206, "XXX", mapOf("Content-Range" to "bytes 3-5/6"))
            else {
                progress.assertSaved(0)
                assertTemp(task, "")
                Reply()
            }
        }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(3L, -6L, 6L), progress.deltas)
        assertEquals(listOf("bytes=3-", null), server.requests.map { it.headers["range"] })
    }

    @Test
    fun shortBodyResumeCountsOnlyRetainedBytes() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        server.reply = { _, index ->
            if (index == 0) Reply(206, "D", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true)
            else {
                progress.assertSaved(4)
                assertTemp(task, "ABCD")
                Reply(206, "EF", mapOf("Content-Range" to "bytes 4-5/6"))
            }
        }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(1L, 2L), progress.deltas)
    }

    @Test fun staleHighTaskCounterIsCorrectedOnce() = correctedCounter(partial = "ABC", cachedSize = 6, correction = -3)
    @Test fun staleLowTaskCounterIsCorrectedOnce() = correctedCounter(partial = "ABC", cachedSize = 0, correction = 3)
    @Test fun missingTempCounterIsCorrectedOnce() = correctedCounter(partial = null, cachedSize = 3, correction = -3)

    private fun correctedCounter(partial: String?, cachedSize: Long, correction: Long) {
        val task = task(partial = partial, cachedSize = cachedSize)
        val progress = AggregateListener(task)
        server.reply = { _, _ ->
            if (partial == null) Reply()
            else Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"))
        }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(correction, if (partial == null) 6L else 3L), progress.deltas)
    }

    @Test
    fun oversizedTempReconciliationDoesNotSubtractUncountedBytes() {
        val task = task(partial = "ABCABCDEF", cachedSize = 3)
        val progress = AggregateListener(task)
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(6L, -9L, 6L), progress.deltas)
    }

    @Test
    fun secondDownloadAfterFinalChunkPauseCannotBypassFailingHttp() {
        val task = task()
        server.reply = { _, _ -> Reply(chunked = true, terminateChunks = false) }
        val pause = object : ModelFileDownloader.FileDownloadListener {
            override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long) = true
        }
        assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, pause) }
        assertNotPublished(task)
        // Intentionally exercise the second call before inspecting the rolled-back temp:
        // the old implementation silently published here without even requesting HTTP.
        server.reply = { _, _ -> Reply(404, "ERR") }
        assertThrows(DownloadProtocolException::class.java) { ModelFileDownloader().downloadFile(task, listener) }
        assertNotPublished(task)
        assertTemp(task, "")
        assertEquals(2, server.requests.size)
    }

    @Test fun finalChunkPauseRollsBackBeforeNextDownload() = finalChunkAbort(resume = false, pause = true)
    @Test fun finalChunkCancellationRollsBackBeforeNextDownload() = finalChunkAbort(resume = false, pause = false)
    @Test fun final206ChunkPauseRollsBackBeforeNextDownload() = finalChunkAbort(resume = true, pause = true)
    @Test fun final206ChunkCancellationRollsBackBeforeNextDownload() = finalChunkAbort(resume = true, pause = false)

    private fun finalChunkAbort(resume: Boolean, pause: Boolean) {
        val prefix = if (resume) "ABC" else ""
        val task = task(partial = prefix, cachedSize = prefix.length.toLong())
        val progress = AggregateListener(task)
        val cancelled = CancellationException("original cancellation")
        val rollbackFailure = IllegalStateException("callback still failing during rollback")
        progress.action = { bytes, delta ->
            if (delta < 0 && !pause) throw rollbackFailure
            if (bytes == 6L && !pause) throw cancelled
            pause
        }
        server.reply = { _, _ ->
            if (resume) Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, terminateChunks = false)
            else Reply(chunked = true, terminateChunks = false)
        }
        if (pause) {
            assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, progress) }
        } else {
            val error = assertThrows(CancellationException::class.java) { downloader.downloadFile(task, progress) }
            assertSame(cancelled, error)
            assertTrue(error.suppressed.contains(rollbackFailure))
        }
        assertNotPublished(task)
        assertTemp(task, prefix)
        progress.assertSaved(prefix.length.toLong())
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        server.reply = { _, _ ->
            if (resume) Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) else Reply()
        }
        ModelFileDownloader().downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        val remaining = 6L - prefix.length
        assertEquals(listOf(remaining, -remaining, remaining), progress.deltas)
        assertEquals(2, server.requests.size)
    }

    @Test
    fun finalChunkThreadInterruptRollsBackAndRemainsInterrupted() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        progress.action = { _, delta -> if (delta > 0) Thread.currentThread().interrupt(); false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, terminateChunks = false) }
        try {
            assertThrows(CancellationException::class.java) { downloader.downloadFile(task, progress) }
            assertTrue(Thread.currentThread().isInterrupted)
        } finally {
            Thread.interrupted()
        }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        progress.assertSaved(3)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(2, server.requests.size)
    }

    @Test
    fun finalChunkRuntimeFailureRollsBackWithoutRetryAndCanResume() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        val original = IllegalStateException("listener failed")
        progress.action = { _, _ -> throw original }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, terminateChunks = false) }
        assertSame(original, assertThrows(IllegalStateException::class.java) { downloader.downloadFile(task, progress) })
        assertNotPublished(task)
        assertTemp(task, "ABC")
        progress.assertSaved(3)
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
    }

    @Test
    fun framingFailureRollbackCorrectsAggregateBeforeRetry() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        server.reply = { _, index ->
            if (index > 0) {
                assertTemp(task, "ABC")
                progress.assertSaved(3)
            }
            Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, terminateChunks = index > 0)
        }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf(3L, -3L, 3L), progress.deltas)
    }

    @Test fun framingRollbackPauseDoesNotHideIoErrorOrRetry() = framingRollbackAbort(pause = true)
    @Test fun framingRollbackCancellationDoesNotHideIoErrorOrRetry() = framingRollbackAbort(pause = false)

    private fun framingRollbackAbort(pause: Boolean) {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        val secondary = CancellationException("rollback cancelled")
        progress.action = { _, delta -> if (delta < 0 && !pause) throw secondary; delta < 0 && pause }
        server.reply = { _, _ ->
            Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, terminateChunks = false)
        }
        val error = assertThrows(IOException::class.java) { downloader.downloadFile(task, progress) }
        assertFalse(error is DownloadProtocolException)
        if (pause) assertTrue(error.suppressed.any { it is DownloadPausedException })
        else assertTrue(error.suppressed.contains(secondary))
        assertNotPublished(task)
        assertTemp(task, "ABC")
        progress.assertSaved(3)
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
    }

    @Test
    fun pauseAtEndOfSubrangeRollsBackUnconfirmedInterval() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        progress.action = { _, _ -> true }
        server.reply = { _, _ ->
            Reply(206, "D", mapOf("Content-Range" to "bytes 3-3/6"), chunked = true, terminateChunks = false)
        }
        assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, progress) }
        assertNotPublished(task)
        assertTemp(task, "ABC")
        progress.assertSaved(3)
        assertEquals(listOf(1L, -1L), progress.deltas)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
        assertEquals(listOf("bytes=3-", "bytes=3-"), server.requests.map { it.headers["range"] })
    }

    @Test fun excessBodyRollbackPreservesProtocolErrorWhenCallbackPauses() = excessBodyRollback(pause = true)
    @Test fun excessBodyRollbackPreservesProtocolErrorWhenCallbackThrows() = excessBodyRollback(pause = false)

    private fun excessBodyRollback(pause: Boolean) {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        val secondary = CancellationException("rollback callback")
        progress.action = { _, delta -> if (delta < 0 && !pause) throw secondary; delta < 0 && pause }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6"), chunked = true, extraChunk = "G") }
        val error = assertThrows(DownloadProtocolException::class.java) { downloader.downloadFile(task, progress) }
        assertTrue(error.message!!.contains("exceeds"))
        if (!pause) assertTrue(error.suppressed.contains(secondary))
        assertNotPublished(task)
        assertTemp(task, "ABC")
        progress.assertSaved(3)
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply(206, "DEF", mapOf("Content-Range" to "bytes 3-5/6")) }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
    }

    @Test
    fun pauseOnIgnoredRangeCorrectionIsNotSwallowed() {
        val task = task(partial = "ABC", cachedSize = 3)
        val progress = AggregateListener(task)
        progress.action = { _, delta -> delta < 0 }
        assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, progress) }
        assertNotPublished(task)
        assertTemp(task, "")
        progress.assertSaved(0)
        assertEquals(listOf(-3L), progress.deltas)
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
    }

    @Test
    fun pauseOnShaResetIsNotSwallowedOrRetried() {
        val task = task(partial = "ABC", cachedSize = 3, etag = digest("ABCDEF"))
        val progress = AggregateListener(task)
        progress.action = { _, delta -> delta < 0 }
        server.reply = { _, _ -> Reply(206, "XXX", mapOf("Content-Range" to "bytes 3-5/6")) }
        assertThrows(DownloadPausedException::class.java) { downloader.downloadFile(task, progress) }
        assertNotPublished(task)
        assertTemp(task, "")
        progress.assertSaved(0)
        assertEquals(1, server.requests.size)
        progress.action = { _, _ -> false }
        server.reply = { _, _ -> Reply() }
        downloader.downloadFile(task, progress)
        assertPublished(task, "ABCDEF")
        progress.assertSaved(6)
    }

    /** Same += delta contract as HF/MS/ML, including updates made before a callback throws. */
    private class AggregateListener(private val task: FileDownloadTask, private val otherFiles: Long = 0) :
        ModelFileDownloader.FileDownloadListener {
        private var saved = otherFiles + task.downloadedSize
        val deltas = mutableListOf<Long>()
        var action: (Long, Long) -> Boolean = { _, _ -> false }

        override fun onDownloadDelta(fileName: String?, downloadedBytes: Long, totalBytes: Long, delta: Long): Boolean {
            assertTrue("must not subtract uncounted file bytes", delta >= -(saved - otherFiles))
            saved += delta
            deltas.add(delta)
            assertTrue("aggregate cannot be negative", saved >= 0)
            assertEquals(task.blobPathIncomplete!!.length(), task.downloadedSize)
            assertEquals(task.downloadedSize, downloadedBytes)
            assertEquals(otherFiles + task.blobPathIncomplete!!.length(), saved)
            return action(downloadedBytes, delta)
        }

        fun assertSaved(fileBytes: Long) { assertEquals(otherFiles + fileBytes, saved) }
    }

    private fun task(size: Long = 6, partial: String? = null, cachedSize: Long = 0, etag: String? = null): FileDownloadTask {
        return FileDownloadTask().apply {
            this.etag = etag
            relativePath = "model.bin"
            fileMetadata = HfFileMetadata().also { it.size = size; it.location = server.url; it.etag = etag }
            blobPath = File(directory, "blob")
            blobPathIncomplete = File(directory, "blob.incomplete")
            pointerPath = File(directory, "snapshot/model.bin")
            downloadedSize = cachedSize
            if (partial != null) blobPathIncomplete!!.writeText(partial)
        }
    }

    private fun assertNotPublished(task: FileDownloadTask) {
        assertFalse("blob published", Files.exists(task.blobPath!!.toPath(), LinkOption.NOFOLLOW_LINKS))
        assertFalse("pointer published", Files.exists(task.pointerPath!!.toPath(), LinkOption.NOFOLLOW_LINKS))
    }

    private fun assertTemp(task: FileDownloadTask, text: String) {
        assertArrayEquals(text.toByteArray(), task.blobPathIncomplete!!.readBytes())
        assertEquals(text.toByteArray().size.toLong(), task.blobPathIncomplete!!.length())
        assertEquals(task.blobPathIncomplete!!.length(), task.downloadedSize)
    }

    private fun assertPublished(task: FileDownloadTask, text: String) {
        assertArrayEquals(text.toByteArray(), task.blobPath!!.readBytes())
        assertArrayEquals(text.toByteArray(), task.pointerPath!!.readBytes())
        assertTrue(Files.isSymbolicLink(task.pointerPath!!.toPath()))
        assertEquals(task.blobPath!!.toPath(), Files.readSymbolicLink(task.pointerPath!!.toPath()))
        assertEquals(text.toByteArray().size.toLong(), task.downloadedSize)
        assertFalse(task.blobPathIncomplete!!.exists())
    }

    private fun digest(text: String, algorithm: String = "SHA-256") =
        MessageDigest.getInstance(algorithm).digest(text.toByteArray()).joinToString("") { "%02x".format(it) }

    private data class HttpRequest(val method: String, val path: String, val headers: Map<String, String>)
    private data class Reply(
        val code: Int = 200,
        val text: String = "ABCDEF",
        val headers: Map<String, String> = emptyMap(),
        val chunked: Boolean = false,
        val declaredLength: Long? = null,
        val extraChunk: String = "",
        val terminateChunks: Boolean = true
    )

    /** Small wire-level fixture so truncated/chunked bodies use the actual OkHttp parser. */
    private class LocalServer : Closeable {
        private val socket = ServerSocket(0, 16, InetAddress.getByName("127.0.0.1"))
        val url = "http://127.0.0.1:${socket.localPort}/model"
        val requests = Collections.synchronizedList(mutableListOf<HttpRequest>())
        private val failures = Collections.synchronizedList(mutableListOf<Throwable>())
        @Volatile var reply: (HttpRequest, Int) -> Reply = { _, _ -> Reply() }
        private val worker = Thread({
            while (!socket.isClosed) {
                try {
                    socket.accept().use { connection ->
                        connection.soTimeout = 5000
                        val reader = connection.getInputStream().bufferedReader(Charsets.US_ASCII)
                        val first = reader.readLine() ?: return@use
                        val headers = mutableMapOf<String, String>()
                        while (true) {
                            val line = reader.readLine() ?: break
                            if (line.isEmpty()) break
                            val colon = line.indexOf(':')
                            if (colon > 0) headers[line.substring(0, colon).lowercase(Locale.ROOT)] = line.substring(colon + 1).trim()
                        }
                        val parts = first.split(' ')
                        val request = HttpRequest(parts[0], parts[1], headers)
                        requests.add(request)
                        val response = reply(request, requests.size - 1)
                        val data = response.text.toByteArray(Charsets.US_ASCII)
                        val output = connection.getOutputStream()
                        val responseHeaders = buildString {
                            append("HTTP/1.1 ${response.code} Test\r\nConnection: close\r\n")
                            response.headers.forEach { (key, value) -> append("$key: $value\r\n") }
                            if (response.chunked) append("Transfer-Encoding: chunked\r\n")
                            else append("Content-Length: ${response.declaredLength ?: data.size.toLong()}\r\n")
                            append("\r\n")
                        }
                        output.write(responseHeaders.toByteArray(Charsets.US_ASCII))
                        if (response.chunked) {
                            for (chunk in listOf(response.text, response.extraChunk).filter { it.isNotEmpty() }) {
                                val bytes = chunk.toByteArray(Charsets.US_ASCII)
                                output.write("${bytes.size.toString(16)}\r\n".toByteArray(Charsets.US_ASCII))
                                output.write(bytes)
                                output.write("\r\n".toByteArray(Charsets.US_ASCII))
                                output.flush()
                            }
                            if (response.terminateChunks) output.write("0\r\n\r\n".toByteArray(Charsets.US_ASCII))
                        } else {
                            output.write(data)
                        }
                        output.flush()
                    }
                } catch (e: SocketException) {
                    // Early body rejection/pause can close the peer before the write finishes.
                    if (!socket.isClosed && e.message?.contains("Broken pipe", true) != true &&
                        e.message?.contains("Connection reset", true) != true) failures.add(e)
                } catch (e: Throwable) {
                    failures.add(e)
                }
            }
        }, "downloader-loopback-test").apply { isDaemon = true; start() }

        override fun close() {
            socket.close()
            worker.join(6000)
            assertFalse("HTTP fixture thread did not finish", worker.isAlive)
            assertTrue("HTTP fixture failures: $failures", failures.isEmpty())
        }
    }
}
