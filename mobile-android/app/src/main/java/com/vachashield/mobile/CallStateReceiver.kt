package com.vachashield.mobile

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import androidx.core.content.ContextCompat

class CallStateReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) {
            return
        }
        if (!MonitorPrefs.isAutoStartOnCallEnabled(context)) {
            return
        }

        when (intent.getStringExtra(TelephonyManager.EXTRA_STATE)) {
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                val startIntent = Intent(context, LiveCallMonitorService::class.java).apply {
                    action = LiveCallMonitorService.ACTION_START
                }
                ContextCompat.startForegroundService(context, startIntent)
            }

            TelephonyManager.EXTRA_STATE_IDLE -> {
                val stopIntent = Intent(context, LiveCallMonitorService::class.java).apply {
                    action = LiveCallMonitorService.ACTION_STOP
                }
                context.startService(stopIntent)
            }
        }
    }
}
