package com.vachashield.mobile

import android.content.Context

object MonitorPrefs {
    const val PREFS_NAME = "vacha_shield_mobile_prefs"
    private const val KEY_BACKEND_URL = "backend_url"
    private const val KEY_AUTO_START_ON_CALL = "auto_start_on_call"

    fun saveBackendUrl(context: Context, url: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_BACKEND_URL, url.trim())
            .apply()
    }

    fun getBackendUrl(context: Context): String {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_BACKEND_URL, "http://10.0.2.2:5000")
            .orEmpty()
    }

    fun setAutoStartOnCall(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_AUTO_START_ON_CALL, enabled)
            .apply()
    }

    fun isAutoStartOnCallEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_AUTO_START_ON_CALL, true)
    }
}
