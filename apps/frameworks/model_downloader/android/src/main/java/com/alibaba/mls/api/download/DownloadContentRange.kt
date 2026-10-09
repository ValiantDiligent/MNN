package com.alibaba.mls.api.download

import java.io.IOException

/** A malformed response/metadata is not a transient connection failure. */
internal class DownloadProtocolException(message: String) : IOException(message)

/** A checksum mismatch requires a fresh download, within the existing retry budget. */
internal class DownloadIntegrityException(message: String) : IOException(message)

internal data class DownloadContentRange(val start: Long, val end: Long, val total: Long) {
    companion object {
        fun parse(header: String?): DownloadContentRange {
            val match = Regex("bytes ([0-9]+)-([0-9]+)/([0-9]+)", RegexOption.IGNORE_CASE)
                .matchEntire(header?.trim().orEmpty())
                ?: throw DownloadProtocolException("Invalid Content-Range: $header")
            val (startText, endText, totalText) = match.destructured
            val start = startText.toLongOrNull()
            val end = endText.toLongOrNull()
            val total = totalText.toLongOrNull()
            if (start == null || end == null || total == null || start > end || end >= total) {
                throw DownloadProtocolException("Invalid Content-Range bounds: $header")
            }
            return DownloadContentRange(start, end, total)
        }
    }
}
