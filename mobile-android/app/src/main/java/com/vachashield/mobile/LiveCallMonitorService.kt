package com.vachashield.mobile

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

class LiveCallMonitorService : Service() {
    companion object {
        const val ACTION_START = "com.vachashield.mobile.START_MONITOR"
        const val ACTION_STOP = "com.vachashield.mobile.STOP_MONITOR"
        const val ACTION_DETECTION_UPDATE = "com.vachashield.mobile.DETECTION_UPDATE"

        const val EXTRA_RUNNING = "running"
        const val EXTRA_ALERT = "alert"
        const val EXTRA_AI_PROB = "ai_prob"
        const val EXTRA_HUMAN_PROB = "human_prob"
        const val EXTRA_VERDICT = "verdict"

        private const val CHANNEL_ID = "vacha_shield_monitor"
        private const val SERVICE_NOTIFICATION_ID = 2001
        private const val ALERT_NOTIFICATION_ID = 2002

        private const val SAMPLE_RATE = 16_000
        private const val CHUNK_SECONDS = 3
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var monitorJob: Job? = null
    private var monitoring = false

    private lateinit var notificationManager: NotificationManager

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopMonitoring()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }

            ACTION_START -> {
                if (!monitoring) {
                    startForeground(SERVICE_NOTIFICATION_ID, buildServiceNotification("Initializing monitor..."))
                    startMonitoring()
                }
            }
        }
        return START_STICKY
    }

    private fun startMonitoring() {
        if (!hasRecordAudioPermission()) {
            stopSelf()
            return
        }

        monitoring = true
        broadcastRunningState(true)
        monitorJob = serviceScope.launch {
            while (isActive && monitoring) {
                val backendUrl = MonitorPrefs.getBackendUrl(this@LiveCallMonitorService)
                val pcm = capturePcmChunk()
                if (pcm == null || pcm.isEmpty()) {
                    updateServiceStatus("Waiting for clean audio...")
                    delay(750)
                    continue
                }

                val chunkFile = File(cacheDir, "monitor_chunk_${System.currentTimeMillis()}.wav")
                try {
                    WavWriter.writePcm16MonoWav(chunkFile, pcm, SAMPLE_RATE)
                    val result = DetectionApiClient.detect(chunkFile, backendUrl)
                    if (result == null) {
                        updateServiceStatus("Backend unreachable")
                        delay(1200)
                        continue
                    }

                    val aiPct = (result.syntheticProbability * 100.0).coerceIn(0.0, 100.0)
                    val status = if (result.alert) {
                        "ALERT ${"%.1f".format(aiPct)}% AI"
                    } else {
                        "Monitoring ${"%.1f".format(aiPct)}% AI"
                    }
                    updateServiceStatus(status)
                    broadcastDetection(result)

                    if (result.alert) {
                        showAlertNotification(result)
                    }
                } catch (_: Exception) {
                    updateServiceStatus("Monitoring error")
                } finally {
                    chunkFile.delete()
                }

                delay(450)
            }
        }
    }

    private fun stopMonitoring() {
        monitoring = false
        monitorJob?.cancel()
        monitorJob = null
        broadcastRunningState(false)
    }

    private fun broadcastRunningState(running: Boolean) {
        sendBroadcast(
            Intent(ACTION_DETECTION_UPDATE).apply {
                putExtra(EXTRA_RUNNING, running)
            }
        )
    }

    private fun broadcastDetection(result: DetectionResult) {
        sendBroadcast(
            Intent(ACTION_DETECTION_UPDATE).apply {
                putExtra(EXTRA_RUNNING, true)
                putExtra(EXTRA_ALERT, result.alert)
                putExtra(EXTRA_AI_PROB, result.syntheticProbability)
                putExtra(EXTRA_HUMAN_PROB, result.humanProbability)
                putExtra(EXTRA_VERDICT, result.verdict)
            }
        )
    }

    private fun hasRecordAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun capturePcmChunk(): ByteArray? {
        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuffer <= 0) {
            return null
        }

        val bytesToCapture = SAMPLE_RATE * CHUNK_SECONDS * 2
        val recordBufferSize = maxOf(minBuffer, bytesToCapture)
        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            recordBufferSize
        )

        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            return null
        }

        val data = ByteArray(bytesToCapture)
        var offset = 0

        return try {
            audioRecord.startRecording()
            while (offset < bytesToCapture && monitoring) {
                val read = audioRecord.read(data, offset, bytesToCapture - offset)
                if (read <= 0) {
                    break
                }
                offset += read
            }

            if (offset < SAMPLE_RATE) {
                null
            } else {
                data.copyOf(offset)
            }
        } catch (_: Exception) {
            null
        } finally {
            try {
                audioRecord.stop()
            } catch (_: Exception) {
            }
            audioRecord.release()
        }
    }

    private fun updateServiceStatus(text: String) {
        notificationManager.notify(SERVICE_NOTIFICATION_ID, buildServiceNotification(text))
    }

    private fun showAlertNotification(result: DetectionResult) {
        val aiPct = (result.syntheticProbability * 100.0).coerceIn(0.0, 100.0)
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_error)
            .setContentTitle("Possible AI voice detected")
            .setContentText("AI probability ${"%.1f".format(aiPct)}%")
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText("Possible AI voice detected in call audio. Verdict: ${result.verdict}")
            )
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        notificationManager.notify(ALERT_NOTIFICATION_ID, notification)
    }

    private fun buildServiceNotification(content: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_call_record)
            .setContentTitle("Vacha Shield Call Monitor")
            .setContentText(content)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Call Monitor",
            NotificationManager.IMPORTANCE_DEFAULT
        )
        channel.description = "Live call monitoring alerts"
        notificationManager.createNotificationChannel(channel)
    }

    override fun onDestroy() {
        stopMonitoring()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
