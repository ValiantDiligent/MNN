// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.util.Log
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.hf.HfShaVerifier
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.LinkOption
import java.util.concurrent.CancellationException
import java.util.concurrent.TimeUnit

class ModelFileDownloader {
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .followRedirects(false)
        .followSslRedirects(false) //                .addInterceptor(OkHttpUtils.createLoggingInterceptor())
        .build()

    @Throws(FileDownloadException::class, DownloadPausedException::class, IOException::class)
    fun downloadFile(
        fileDownloadTask: FileDownloadTask,
        fileDownloadListener: FileDownloadListener
    ) {
        checkInterrupted()
        Log.d(TAG, "downloadFile inner")
        fileDownloadTask.pointerPath!!.parentFile?.mkdirs()
        fileDownloadTask.blobPath!!.parentFile?.mkdirs()

        if (fileDownloadTask.pointerPath!!.exists()) {
            if (!validate(fileDownloadTask, fileDownloadTask.pointerPath!!)) {
                throw FileDownloadException("Existing pointer failed validation")
            }
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists")
            return
        }

        if (fileDownloadTask.blobPath!!.exists()) {
            if (!validate(fileDownloadTask, fileDownloadTask.blobPath!!)) {
                throw FileDownloadException("Existing destination failed validation")
            }
            createSymlink(
                fileDownloadTask.blobPath.toString(),
                fileDownloadTask.pointerPath.toString()
            )
            Log.d(
                TAG,
                "DownloadFile " + fileDownloadTask.relativePath + " already exists just create symlink"
            )
            return
        }
        synchronized(this) {
            val hfFileMetadata = fileDownloadTask.fileMetadata
            downloadToTmpAndMove(
                fileDownloadTask,
                fileDownloadTask.blobPathIncomplete!!,
                fileDownloadTask.blobPath!!,
                hfFileMetadata!!.location!!,
                hfFileMetadata.size,
                fileDownloadListener
            )
            createSymlink(
                fileDownloadTask.blobPath!!.toPath(),
                fileDownloadTask.pointerPath!!.toPath()
            )
        }
    }

    @Throws(FileDownloadException::class, DownloadPausedException::class, IOException::class)
    private fun downloadToTmpAndMove(
        fileDownloadTask: FileDownloadTask,
        incompletePath: File,
        destinationPath: File,
        urlToDownload: String,
        expectedSize: Long,
        fileDownloadListener: FileDownloadListener
    ) {
        if (expectedSize < 0) {
            throw DownloadProtocolException("Invalid metadata size: $expectedSize")
        }
        if (destinationPath.exists()) {
            if (!validate(fileDownloadTask, destinationPath)) {
                throw FileDownloadException("Existing destination failed validation")
            }
            return
        }
        // Never truncate a destination (or a symlink to one) when restarting a download.
        if (incompletePath.canonicalFile == destinationPath.canonicalFile ||
            Files.isSymbolicLink(incompletePath.toPath())
        ) {
            throw DownloadProtocolException("Temporary download must be a separate regular file")
        }
        syncProgress(fileDownloadTask, incompletePath.length(), fileDownloadListener)
        if (incompletePath.isFile && fileDownloadTask.downloadedSize >= expectedSize) {
            // Length alone cannot prove EOF for a temp left by an earlier pause/crash.
            // A verified hash permits offline completion of nonempty files. Metadata
            // size zero always needs a successful empty HTTP response.
            if (expectedSize > 0 && !fileDownloadTask.etag.isNullOrEmpty() && validate(fileDownloadTask, incompletePath)) {
                moveWithPermissions(incompletePath, destinationPath)
                return
            }
            resetIncomplete(fileDownloadTask, incompletePath, fileDownloadListener)
        }

        val maxRetry = 10
        for (i in 0 until maxRetry) {
            try {
                // A server may return a smaller, but valid, subrange. Each response must
                // finish its declared interval before we request the next one.
                do {
                    downloadChunk(
                        fileDownloadTask, urlToDownload, incompletePath, expectedSize,
                        fileDownloadListener
                    )
                } while (incompletePath.length() < expectedSize)
                if (!validate(fileDownloadTask, incompletePath)) {
                    resetIncomplete(fileDownloadTask, incompletePath, fileDownloadListener)
                    throw DownloadIntegrityException("Downloaded file failed validation")
                }
            } catch (e: DownloadPausedException) {
                throw e
            } catch (e: CancellationException) {
                throw e
            } catch (e: DownloadProtocolException) {
                // Repeating a malformed response cannot repair the protocol.
                throw e
            } catch (e: IOException) {
                // A rollback callback failure (including pause/cancel) must not cause
                // another request or replace the original response error.
                if (e.suppressed.isNotEmpty()) {
                    throw e
                }
                if (Thread.currentThread().isInterrupted) {
                    throw CancellationException("Download interrupted").apply { initCause(e) }
                }
                if (i == maxRetry - 1) {
                    throw e
                }
                Log.e(TAG, "downloadChunk failed sleep and retrying: " + e.message)
                try {
                    Thread.sleep(1000)
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    throw CancellationException("Download interrupted").apply { initCause(e) }
                }
                continue
            }
            // Publish only after length and optional hash have both been verified.
            moveWithPermissions(incompletePath, destinationPath)
            return
        }
    }

    @Throws(FileDownloadException::class, DownloadPausedException::class, IOException::class)
    private fun downloadChunk(
        fileDownloadTask: FileDownloadTask,
        url: String,
        tempFile: File,
        expectedSize: Long,
        fileDownloadListener: FileDownloadListener
    ) {
        checkInterrupted()
        val resumeOffset = tempFile.length()
        syncProgress(fileDownloadTask, resumeOffset, fileDownloadListener)
        val requestBuilder = Request.Builder()
            .url(url)
            .get()
            .header("Accept-Encoding", "identity")
        if (resumeOffset > 0) {
            requestBuilder.header("Range", "bytes=$resumeOffset-")
        }
        var request = requestBuilder.build()
        // Resolve redirects on the real GET rather than discarding a preflight GET body.
        // HttpUrl.resolve handles both relative and absolute Locations without an NPE.
        for (redirect in 0..10) {
            client.newCall(request).execute().use { response ->
                if (response.code in listOf(301, 302, 303, 307, 308)) {
                    val location = response.header("Location")
                        ?: throw DownloadProtocolException("Redirect missing Location")
                    val nextUrl = response.request.url.resolve(location)
                        ?: throw DownloadProtocolException("Invalid redirect Location: $location")
                    if (redirect == 10) {
                        throw DownloadProtocolException("Too many redirects")
                    }
                    request = request.newBuilder().url(nextUrl).build()
                    return@use
                }
                // A 416 body is an error document, never model data. Fail explicitly;
                // complete local files have already passed length/hash validation above.
                if (response.code == 416) {
                    throw DownloadProtocolException("HTTP 416: range not satisfiable")
                }
                if (response.code != 200 && response.code != 206) {
                    if (response.code == 408 || response.code == 429 || response.code in 500..599) {
                        throw IOException("HTTP error: ${response.code}")
                    }
                    throw DownloadProtocolException("Unexpected HTTP status: ${response.code}")
                }
                if (response.header("Content-Encoding")?.let { !it.equals("identity", true) } == true) {
                    throw DownloadProtocolException("Unexpected Content-Encoding")
                }
                val body = response.body ?: throw DownloadProtocolException("Missing response body")
                val start: Long
                val bodySize: Long
                if (response.code == 206) {
                    val range = DownloadContentRange.parse(response.header("Content-Range"))
                    if (range.start != resumeOffset || range.total != expectedSize) {
                        throw DownloadProtocolException("Content-Range does not match request/metadata")
                    }
                    start = range.start
                    bodySize = range.end - range.start + 1
                } else {
                    // Range was ignored: consume this full response from byte zero.
                    start = 0
                    bodySize = expectedSize
                }
                if (body.contentLength() != -1L && body.contentLength() != bodySize) {
                    throw DownloadProtocolException("Content-Length does not match expected body size")
                }
                body.byteStream().use { input ->
                    RandomAccessFile(tempFile, "rw").use { raf ->
                        val buffer = ByteArray(8192)
                        var received = 0L
                        var completed = false
                        var invalidBody = false
                        var failure: Throwable? = null
                        try {
                            if (response.code == 200) {
                                raf.setLength(0)
                            }
                            syncProgress(fileDownloadTask, start, fileDownloadListener)
                            raf.seek(start)
                            while (true) {
                                checkInterrupted()
                                val bytesRead = input.read(buffer)
                                if (bytesRead == -1) break
                                if (bytesRead.toLong() > bodySize - received) {
                                    invalidBody = true
                                    throw DownloadProtocolException("Response body exceeds expected interval")
                                }
                                raf.write(buffer, 0, bytesRead)
                                received += bytesRead
                                syncProgress(fileDownloadTask, start + received, fileDownloadListener)
                            }
                            if (received != bodySize) {
                                throw IOException("Incomplete response body: $received of $bodySize bytes")
                            }
                            checkInterrupted()
                            completed = true
                        } catch (e: Throwable) {
                            failure = e
                            throw e
                        } finally {
                            // All exits before EOF count: read errors, interruption and
                            // listener pause/throw on the final write. Short valid prefixes
                            // remain resumable, but an unconfirmed full interval does not.
                            if (!completed && (received == bodySize || invalidBody)) {
                                try {
                                    raf.setLength(start)
                                    syncProgress(fileDownloadTask, start, fileDownloadListener)
                                } catch (rollback: Throwable) {
                                    // Managers apply += delta before notifying observers. Do
                                    // not call twice, or let an observer hide the root error.
                                    if (failure == null) throw rollback
                                    if (rollback !== failure) failure.addSuppressed(rollback)
                                }
                            }
                        }
                    }
                }
                return
            }
        }
    }

    private fun checkInterrupted() {
        if (Thread.currentThread().isInterrupted) {
            throw CancellationException("Download interrupted")
        }
    }

    private fun resetIncomplete(
        fileDownloadTask: FileDownloadTask,
        file: File,
        listener: FileDownloadListener
    ) {
        RandomAccessFile(file, "rw").use { it.setLength(0) }
        syncProgress(fileDownloadTask, 0, listener)
    }

    private fun syncProgress(fileDownloadTask: FileDownloadTask, retainedBytes: Long, listener: FileDownloadListener) {
        // The task counter is this file's contribution to the managers' aggregate.
        // Reconcile disk length before any reset so we never subtract uncounted bytes.
        val previous = fileDownloadTask.downloadedSize.coerceAtLeast(0)
        val delta = retainedBytes - previous
        fileDownloadTask.downloadedSize = retainedBytes
        // Record delivery before calling out: HF/MS/ML apply the delta even when their
        // downstream observer throws or the manager returns a paused state.
        if (delta != 0L && listener.onDownloadDelta(
                fileDownloadTask.relativePath, retainedBytes, fileDownloadTask.fileMetadata!!.size, delta
            )
        ) {
            throw DownloadPausedException("Download paused")
        }
    }

    private fun validate(fileDownloadTask: FileDownloadTask, src: File): Boolean {
        val expectedSize = fileDownloadTask.fileMetadata?.size ?: return false
        if (expectedSize < 0 || !src.isFile || src.length() != expectedSize) {
            return false
        }
        if (fileDownloadTask.etag.isNullOrEmpty()) {
            return true
        }
        return try {
            HfShaVerifier.verify(fileDownloadTask.etag!!, src.toPath())
        } catch (e: IllegalArgumentException) {
            throw DownloadProtocolException("Invalid checksum metadata: ${e.message}")
        }
    }

    private fun moveWithPermissions(src: File, dest: File) {
        checkInterrupted()
        Log.d(DownloadFileUtils.TAG, "moveWithPermissions ${src.absolutePath} to ${dest.absolutePath}")
        // Do not replace a destination created while this download was in progress.
        if (Files.exists(dest.toPath(), LinkOption.NOFOLLOW_LINKS)) {
            throw FileDownloadException("Destination already exists: $dest")
        }
        Files.move(src.toPath(), dest.toPath())
        dest.setReadable(true, true)
        dest.setWritable(true, true)
        dest.setExecutable(false, false)
    }

    interface FileDownloadListener {
        /**
         * Signed change in retained bytes, including negative reset/rollback corrections.
         * Apply delta before notifying observers or returning the paused state.
         */
        fun onDownloadDelta(
            fileName: String?,
            downloadedBytes: Long,
            totalBytes: Long,
            delta: Long
        ): Boolean
    }

    companion object {
        const val TAG: String = "ModelFileDownloader"
    }
}
