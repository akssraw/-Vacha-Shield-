# Vacha-Shield Localhost Setup

## 1) Install Python dependencies

```powershell
pip install -r requirements.txt
```

If you also want microphone-based local tools such as `call_monitor.py`, install:

```powershell
pip install -r requirements-optional.txt
```

## 2) Build the Lovable UI

```powershell
cd lovable-project-c7943770-3277-4c80-b708-c4aaa8cf13c4-2026-03-09
npm install
npm run build
cd ..
```

## 3) Run the Flask app

```powershell
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

The Flask server now serves the built Lovable UI directly and exposes:
- `POST /detect_voice`
- `POST /feedback`
- `GET /health`

## Optional: Train with internet data (recommended if results are poor)

```powershell
python train_internet_model.py --epochs 10 --target_edge_ai 120 --target_elevenlabs_ai 120 --max_librispeech 120
```

This will:
- Download real human speech from internet datasets.
- Generate cloud TTS deepclone samples from Edge + ElevenLabs voices.
- Retrain `model.pth` and save `model_calibration.json` with an optimized threshold.

For ElevenLabs generation, set `ELEVENLABS_API_KEY` in a local `.env` file at the project root, or point `VACHA_ENV_FILE` to a custom env file.

## Optional: UI Dev Mode (Vite)

If you want hot-reload UI edits:

```powershell
cd lovable-project-c7943770-3277-4c80-b708-c4aaa8cf13c4-2026-03-09
$env:VITE_API_BASE_URL="http://127.0.0.1:5000"
npm run dev
```

Keep `python app.py` running in a separate terminal.

## Background Call Monitoring (mobile flow simulation)

Terminal mode:

```powershell
python call_monitor.py
```

Ambient mode:

```powershell
python ambient_audio_monitor.py
```

Kivy mobile mockup mode:

```powershell
python vacha_mobile_app.py
```
