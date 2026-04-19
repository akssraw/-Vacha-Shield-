package com.vachashield.mobile

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

data class DetectionResult(
    val syntheticProbability: Double,
    val humanProbability: Double,
    val alert: Boolean,
    val verdict: String
)

object DetectionApiClient {
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .build()

    fun detect(audioFile: File, backendUrl: String): DetectionResult? {
        val normalizedBase = backendUrl.trim().trimEnd('/')
        if (normalizedBase.isEmpty()) {
            return null
        }

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                audioFile.name,
                audioFile.asRequestBody("audio/wav".toMediaType())
            )
            .addFormDataPart("analysis_profile", "strict")
            .build()

        val request = Request.Builder()
            .url("$normalizedBase/detect_voice")
            .post(body)
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return null
                }
                val raw = response.body?.string().orEmpty()
                if (raw.isBlank()) {
                    return null
                }
                val json = JSONObject(raw)
                DetectionResult(
                    syntheticProbability = json.optDouble("synthetic_probability", 0.0),
                    humanProbability = json.optDouble("human_probability", 1.0),
                    alert = json.optBoolean("alert", false),
                    verdict = json.optString("verdict", "unknown")
                )
            }
        } catch (_: Exception) {
            null
        }
    }
}
