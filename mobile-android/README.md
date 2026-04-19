# Vacha Shield Mobile (Phase 2)

This Android starter app monitors live call-adjacent audio, sends short WAV chunks to the existing backend (`/detect_voice`), and shows real-time AI voice alerts.

## What is implemented

- Foreground background-monitor service (`LiveCallMonitorService`)
- Continuous audio chunk capture (16 kHz, mono, 3-second windows)
- Backend upload with strict detector profile
- Alert notifications on suspicious segments
- Call-state auto start and stop receiver
- Simple control UI for backend URL, auto mode, and live status

## Open in Android Studio

1. Open folder:
   - `C:\Users\Aksshat Singh Rawat\OneDrive\Desktop\Viddhi\Vacha-Shield\mobile-android`
2. Let Gradle sync.
3. Run on a physical Android device for real call tests.

## Backend requirement

Keep backend running on your laptop:

```powershell
cd "C:\Users\Aksshat Singh Rawat\OneDrive\Desktop\Viddhi\Vacha-Shield"
python app.py
```

Then in mobile app set backend URL to your laptop LAN IP, for example:

- `http://192.168.1.23:5000`

If using Android emulator, default `http://10.0.2.2:5000` works.

## Permissions used

- `RECORD_AUDIO`
- `READ_PHONE_STATE`
- `POST_NOTIFICATIONS` (Android 13+)
- Foreground service permissions

## Important platform note

Android call-audio capture has OEM and OS restrictions. This app is designed for practical hackathon demo flow using microphone/call-adjacent monitoring. For production-grade telephony capture, a device-specific or dialer-role implementation is usually required.
