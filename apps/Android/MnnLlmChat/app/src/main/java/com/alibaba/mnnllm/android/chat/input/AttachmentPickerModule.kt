// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

class AttachmentPickerModule(private val activity: ChatActivity) {
    private val takePhotoView: View
    private val chooseImageView: View
    private val chooseVideoView: View

    private val attachmentPreview: ImageView
    private val imagePreviewLayout: View
    private val imagePreviewDelete: ImageView
    private val selectAttachmentLayoutParent: View

    private var imageUri: Uri? = null
    private var photoFile: File? = null
    private var callback: ImagePickCallback? = null

    private val imagePreviewRecycler: androidx.recyclerview.widget.RecyclerView
    private val imagePreviewAdapter: ImagePreviewAdapter

    init {
        takePhotoView = activity.findViewById(R.id.more_item_camera)
        chooseImageView = activity.findViewById(R.id.more_item_photo)
        chooseVideoView = activity.findViewById(R.id.more_item_video)
        if (ModelTypeUtils.isVisualModel(activity.modelId!!)) {
            takePhotoView.setOnClickListener { v: View? -> takePhoto() }
            chooseImageView.setOnClickListener { v: View? -> chooseImageView() }
        } else {
            takePhotoView.visibility = View.GONE
            chooseImageView.visibility = View.GONE
        }
        if (ModelTypeUtils.isVideoModel(activity.modelId!!)) {
            chooseVideoView.setOnClickListener { v: View? -> chooseVideo() }
        } else {
            chooseVideoView.visibility = View.GONE
        }
        val chooseAudioView = activity.findViewById<View>(R.id.more_item_audio)
        if (ModelTypeUtils.isAudioModel(activity.modelId!!)) {
            chooseAudioView.setOnClickListener { v: View? -> chooseAudio() }
        } else {
            chooseAudioView.visibility = View.GONE
        }
        val voiceChatView = activity.findViewById<View>(R.id.more_item_voice_chat)
        //disable temporary
        voiceChatView.visibility = View.GONE
        attachmentPreview = activity.findViewById(R.id.image_preview)
        imagePreviewLayout = activity.findViewById(R.id.image_preview_layout)
        imagePreviewRecycler = activity.findViewById(R.id.image_preview_recycler)
        imagePreviewDelete = activity.findViewById(R.id.image_preview_delete)
        selectAttachmentLayoutParent = activity.findViewById(R.id.layout_more_menu)
        imagePreviewDelete.setOnClickListener { v: View? -> deletePreviewImage() }

        imagePreviewAdapter = ImagePreviewAdapter { uri ->
            imagePreviewAdapter.removeImage(uri)
            if (imagePreviewAdapter.itemCount == 0) {
                hidePreview()
            } else {
                if (callback != null) {
                    callback!!.onAttachmentPicked(imagePreviewAdapter.getImages(), AttachmentType.Image)
                }
            }
        }
        imagePreviewRecycler.adapter = imagePreviewAdapter
    }

    private fun deletePreviewImage() {
        if (imageUri != null) {
            if (photoFile != null) {
                photoFile!!.delete()
                photoFile = null
            }
            imageUri = null
        }
        showAttachmentLayout()
        hidePreview()
    }

    private fun hidePreview() {
        imagePreviewLayout.visibility = View.GONE
        imagePreviewRecycler.visibility = View.GONE
        imagePreviewDelete.visibility = View.GONE
        if (callback != null) {
            callback!!.onAttachmentRemoved()
        }
    }

    private fun chooseAudio() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("audio/x-wav")
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        try {
            activity.startActivityForResult(
                Intent.createChooser(intent, activity.getString(R.string.select_wav_file)),
                REQUEST_CODE_SELECT_WAV
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this.activity, R.string.file_manager_required, Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun startVoiceChat() {
        // Hide the attachment menu
        hideAttachmentLayout()
        
        // Start the voice chat fragment
        activity.startVoiceChat()
    }

    private fun chooseImageView() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
        activity.startActivityForResult(
            Intent.createChooser(intent, activity.getString(R.string.select_picture)),
            REQUEST_CODE_SELECT_IMAGE,
            null
        )
    }

    private fun chooseVideo() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("video/*")
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        try {
            activity.startActivityForResult(
                Intent.createChooser(intent, activity.getString(R.string.select_video)),
                REQUEST_CODE_SELECT_VIDEO
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this.activity, R.string.file_manager_required, Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun takePhoto() {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) 
            != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted, request it
            if (ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)) {
                // Show rationale to user
                Toast.makeText(activity, R.string.camera_permission_rationale, Toast.LENGTH_LONG).show()
            }
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA_PERMISSION
            )
            return
        }
        
        // Permission is granted, proceed with camera
        startCameraIntent()
    }
    
    private fun startCameraIntent() {
        val sessionId = activity.sessionId
        if (sessionId.isNullOrBlank()) {
            showImportFailure(AttachmentType.Image, IOException("Missing attachment session"))
            return
        }
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        try {
            val destination = File(FileUtils.generateDestPhotoFilePath(activity, sessionId))
            val fileProviderUri = FileProvider.getUriForFile(
                activity,
                activity.packageName + ".fileprovider",
                destination
            )
            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProviderUri)
            activity.startActivityForResult(cameraIntent, REQUEST_CODE_CAPTURE_IMAGE)
            photoFile = destination
            imageUri = Uri.fromFile(destination)
        } catch (e: IOException) {
            showImportFailure(AttachmentType.Image, e)
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission denied", e)
            Toast.makeText(activity, R.string.camera_permission_denied, Toast.LENGTH_SHORT).show()
        } catch (e: ActivityNotFoundException) {
            Log.e(TAG, "No camera app found", e)
            Toast.makeText(activity, R.string.no_camera_app_found, Toast.LENGTH_SHORT).show()
        }
    }

    fun setOnImagePickCallback(callback: ImagePickCallback?) {
        this.callback = callback
    }

    fun canHandleResult(requestCode: Int): Boolean {
        return requestCode >= REQUEST_CODE_SELECT_WAV && requestCode <= REQUEST_CODE_CAPTURE_IMAGE ||
               requestCode == REQUEST_CODE_CAMERA_PERMISSION ||
               requestCode == REQUEST_CODE_SELECT_VIDEO
    }

    fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (resultCode != Activity.RESULT_OK) {
            if (requestCode == REQUEST_CODE_CAPTURE_IMAGE) {
                imageUri = null
            }
            return
        }
        when (requestCode) {
            REQUEST_CODE_CAPTURE_IMAGE -> {
                val capturedUri = imageUri
                imageUri = null
                if (capturedUri == null || activity.sessionId.isNullOrBlank()) {
                    showImportFailure(AttachmentType.Image, IOException("Missing captured image or session"))
                    return
                }
                imagePreviewAdapter.addImage(capturedUri)
                showImagePreview()
            }
            REQUEST_CODE_SELECT_IMAGE -> importPickedAttachment(data, AttachmentType.Image)
            REQUEST_CODE_SELECT_VIDEO -> importPickedAttachment(data, AttachmentType.Video)
            REQUEST_CODE_SELECT_WAV -> importPickedAttachment(data, AttachmentType.Audio)
        }
    }

    private fun importPickedAttachment(data: Intent?, type: AttachmentType) {
        importAttachmentResult(
            data = data,
            sessionId = activity.sessionId,
            allowMultiple = type == AttachmentType.Image,
            copy = { uri, sessionId ->
                val path = File(when (type) {
                    AttachmentType.Image -> FileUtils.generateDestImageFilePath(activity, sessionId)
                    AttachmentType.Audio -> FileUtils.generateDestAudioFilePath(activity, sessionId)
                    AttachmentType.Video -> FileUtils.generateDestVideoFilePath(activity, sessionId)
                })
                // Do not overwrite an existing attachment when imports share a timestamp.
                val destination = File.createTempFile(path.nameWithoutExtension + "_", "." + path.extension, path.parentFile)
                Uri.fromFile(FileUtils.copyFileUriToPath(activity, uri, destination.absolutePath))
            },
            onImported = { uris ->
                when (type) {
                    AttachmentType.Image -> {
                        imagePreviewAdapter.addImages(uris)
                        showImagePreview()
                    }
                    AttachmentType.Audio -> showAudioPreview(uris.first())
                    AttachmentType.Video -> showVideoPreview(uris.first())
                }
            },
            onFailure = { showImportFailure(type, it) }
        )
    }

    private fun showImportFailure(type: AttachmentType, error: Exception) {
        Log.e(TAG, "Failed to import $type attachment", error)
        val message = when (type) {
            AttachmentType.Image -> R.string.failed_to_save_image
            AttachmentType.Audio -> R.string.audio_file_failed
            AttachmentType.Video -> R.string.video_file_failed
        }
        Toast.makeText(activity, message, Toast.LENGTH_SHORT).show()
    }
    
    fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Camera permission granted, proceed with camera
                startCameraIntent()
            } else {
                // Camera permission denied
                Toast.makeText(activity, R.string.camera_permission_denied, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showAudioPreview(audioUri: Uri) {
        attachmentPreview.setImageResource(R.drawable.ic_audio_attachment)
        imagePreviewLayout.visibility = View.VISIBLE
        imagePreviewDelete.visibility = View.VISIBLE
        imagePreviewRecycler.visibility = View.GONE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(listOf(audioUri), AttachmentType.Audio)
        }
    }

    private fun showVideoPreview(videoUri: Uri) {
        activity.lifecycleScope.launch(Dispatchers.IO) {
            try {
                val thumbnail = com.alibaba.mnnllm.android.utils.VideoThumbnailUtils.generateVideoThumbnail(
                    activity, videoUri, 200, 200
                )
                withContext(Dispatchers.Main) {
                    if (thumbnail != null) {
                        attachmentPreview.setImageBitmap(thumbnail)
                    } else {
                        attachmentPreview.setImageResource(R.drawable.ic_video)
                    }
                    imagePreviewLayout.visibility = View.VISIBLE
                    imagePreviewDelete.visibility = View.VISIBLE
                    imagePreviewRecycler.visibility = View.GONE
                    hideAttachmentLayout()
                    if (callback != null) {
                        callback!!.onAttachmentPicked(listOf(videoUri), AttachmentType.Video)
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    attachmentPreview.setImageResource(R.drawable.ic_video)
                    imagePreviewLayout.visibility = View.VISIBLE
                    imagePreviewDelete.visibility = View.VISIBLE
                    imagePreviewRecycler.visibility = View.GONE
                    hideAttachmentLayout()
                    if (callback != null) {
                        callback!!.onAttachmentPicked(listOf(videoUri), AttachmentType.Video)
                    }
                }
            }
        }
    }

    private fun showImagePreview() {
        imagePreviewRecycler.visibility = View.VISIBLE
        imagePreviewLayout.visibility = View.GONE
        imagePreviewDelete.visibility = View.GONE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(imagePreviewAdapter.getImages(), AttachmentType.Image)
        }
        imageUri = null
    }

    fun toggleAttachmentVisibility() {
        if (isShowing) {
            hideAttachmentLayout()
        } else {
            showAttachmentLayout()
        }
    }

    fun hideAttachmentLayout() {
        selectAttachmentLayoutParent.visibility = View.GONE
        if (callback != null) {
            callback!!.onAttachmentLayoutHide()
        }
    }

    private fun showAttachmentLayout() {
        selectAttachmentLayoutParent.visibility = View.VISIBLE
        if (callback != null) {
            callback!!.onAttachmentLayoutShow()
        }
    }

    val isShowing: Boolean
        get() = selectAttachmentLayoutParent.visibility == View.VISIBLE

    fun clearInput() {
        photoFile = null
        imageUri = null
        imagePreviewAdapter.clear()
        hidePreview()
    }

    interface ImagePickCallback {
        fun onAttachmentPicked(imageUris: List<Uri>?, audio: AttachmentType?)
        fun onAttachmentRemoved()

        fun onAttachmentLayoutShow()

        fun onAttachmentLayoutHide()
    }

    enum class AttachmentType {
        Image, Audio, Video
    }

    companion object {
        const val TAG: String = "ImagePickerModule"
        var REQUEST_CODE_CAPTURE_IMAGE: Int = 100
        const val REQUEST_CODE_CAMERA_PERMISSION: Int = 101

        var REQUEST_CODE_SELECT_IMAGE: Int = 99
        var REQUEST_CODE_SELECT_VIDEO: Int = 97
        var REQUEST_CODE_SELECT_WAV: Int = 98
    }
}
