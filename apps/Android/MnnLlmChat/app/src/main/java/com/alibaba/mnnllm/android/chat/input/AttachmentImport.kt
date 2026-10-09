package com.alibaba.mnnllm.android.chat.input

import android.content.Intent
import android.net.Uri
import java.io.IOException

/** Imports picker results without changing the current attachment on failure. */
internal fun importAttachmentResult(
    data: Intent?,
    sessionId: String?,
    allowMultiple: Boolean,
    copy: (Uri, String) -> Uri,
    onImported: (List<Uri>) -> Unit,
    onFailure: (Exception) -> Unit
) {
    if (sessionId.isNullOrBlank()) {
        onFailure(IOException("Missing attachment session"))
        return
    }
    val uris = try {
        val clipData = if (allowMultiple) data?.clipData else null
        if (clipData != null) {
            List(clipData.itemCount) { clipData.getItemAt(it).uri }
        } else {
            listOf(data?.data)
        }
    } catch (e: SecurityException) {
        onFailure(e)
        return
    }
    val imported = mutableListOf<Uri>()
    var failure: Exception? = null
    if (uris.isEmpty()) {
        failure = IOException("Empty attachment result")
    }
    for (uri in uris) {
        if (uri == null || uri == Uri.EMPTY) {
            failure = IOException("Missing attachment URI")
            continue
        }
        try {
            imported.add(copy(uri, sessionId))
        } catch (e: IOException) {
            failure = e
        } catch (e: SecurityException) {
            failure = e
        }
    }
    // Publish only successfully copied files; failed items never clear existing previews.
    if (imported.isNotEmpty()) {
        onImported(imported)
    }
    failure?.let(onFailure)
}
