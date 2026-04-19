package com.vachashield.mobile

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Switch
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {
    private lateinit var backendUrlInput: EditText
    private lateinit var autoStartSwitch: Switch
    private lateinit var startStopButton: Button
    private lateinit var statusText: TextView
    private lateinit var lastResultText: TextView

    private var monitorRunning = false
    private var updateReceiverRegistered = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        val granted = requiredPermissions().all {
            grants[it] == true || ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        if (granted) {
            startMonitoring()
        } else {
            statusText.text = "Permissions denied. Cannot start monitoring."
        }
    }

    private val updateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != LiveCallMonitorService.ACTION_DETECTION_UPDATE) {
                return
            }

            if (intent.hasExtra(LiveCallMonitorService.EXTRA_RUNNING)) {
                monitorRunning = intent.getBooleanExtra(LiveCallMonitorService.EXTRA_RUNNING, false)
                updateStartStopButton()
                statusText.text = if (monitorRunning) "Live monitoring active" else "Monitoring stopped"
            }

            if (intent.hasExtra(LiveCallMonitorService.EXTRA_AI_PROB)) {
                val ai = intent.getDoubleExtra(LiveCallMonitorService.EXTRA_AI_PROB, 0.0) * 100.0
                val human = intent.getDoubleExtra(LiveCallMonitorService.EXTRA_HUMAN_PROB, 1.0) * 100.0
                val verdict = intent.getStringExtra(LiveCallMonitorService.EXTRA_VERDICT).orEmpty()
                val alert = intent.getBooleanExtra(LiveCallMonitorService.EXTRA_ALERT, false)

                lastResultText.text = buildString {
                    append("Last result\n")
                    append("Human: ${"%.1f".format(human)}%\n")
                    append("AI: ${"%.1f".format(ai)}%\n")
                    append("Verdict: $verdict\n")
                    append("Alert: ${if (alert) "YES" else "NO"}")
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        backendUrlInput = findViewById(R.id.backendUrlInput)
        autoStartSwitch = findViewById(R.id.autoStartSwitch)
        startStopButton = findViewById(R.id.startStopButton)
        statusText = findViewById(R.id.statusText)
        lastResultText = findViewById(R.id.lastResultText)

        backendUrlInput.setText(MonitorPrefs.getBackendUrl(this))
        autoStartSwitch.isChecked = MonitorPrefs.isAutoStartOnCallEnabled(this)
        autoStartSwitch.setOnCheckedChangeListener { _, isChecked ->
            MonitorPrefs.setAutoStartOnCall(this, isChecked)
        }

        startStopButton.setOnClickListener {
            MonitorPrefs.saveBackendUrl(this, backendUrlInput.text.toString())
            MonitorPrefs.setAutoStartOnCall(this, autoStartSwitch.isChecked)

            if (monitorRunning) {
                stopMonitoring()
            } else {
                if (hasAllPermissions()) {
                    startMonitoring()
                } else {
                    permissionLauncher.launch(requiredPermissions())
                }
            }
        }

        updateStartStopButton()
    }

    override fun onStart() {
        super.onStart()
        registerUpdateReceiver()
    }

    override fun onStop() {
        if (updateReceiverRegistered) {
            unregisterReceiver(updateReceiver)
            updateReceiverRegistered = false
        }
        super.onStop()
    }

    private fun registerUpdateReceiver() {
        if (updateReceiverRegistered) {
            return
        }
        val filter = IntentFilter(LiveCallMonitorService.ACTION_DETECTION_UPDATE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(updateReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(updateReceiver, filter)
        }
        updateReceiverRegistered = true
    }

    private fun startMonitoring() {
        val intent = Intent(this, LiveCallMonitorService::class.java).apply {
            action = LiveCallMonitorService.ACTION_START
        }
        ContextCompat.startForegroundService(this, intent)
        monitorRunning = true
        statusText.text = "Starting live monitor..."
        updateStartStopButton()
    }

    private fun stopMonitoring() {
        val intent = Intent(this, LiveCallMonitorService::class.java).apply {
            action = LiveCallMonitorService.ACTION_STOP
        }
        startService(intent)
        monitorRunning = false
        statusText.text = "Stopping monitor..."
        updateStartStopButton()
    }

    private fun updateStartStopButton() {
        startStopButton.text = if (monitorRunning) "Stop Live Monitor" else "Start Live Monitor"
    }

    private fun hasAllPermissions(): Boolean {
        return requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requiredPermissions(): Array<String> {
        val list = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            list += Manifest.permission.POST_NOTIFICATIONS
        }
        return list.toTypedArray()
    }
}
