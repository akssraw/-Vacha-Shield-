---
title: Vacha Shield
sdk: docker
app_port: 7860
---

# Vacha Shield

Vacha Shield is an AI deepfake voice detection system for web and mobile-assisted call monitoring.

## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run backend:
   `python app.py`
3. Open:
   `http://127.0.0.1:5000`

## Core Files

- `app.py` - Flask API + UI serving
- `deepfake_detector.py` - Hybrid inference and scoring logic
- `feature_extraction.py` - PCEN + Delta feature pipeline
- `model.py` - AudioCNN architecture
- `train_knowledge_base.py` - Continuous learning retraining

## Architecture

See `docs/architecture.png`.

## Deployment

See `DEPLOY_FREE.md` for the recommended free-hosting paths.

- Best free link longevity: Hugging Face Spaces with the included `Dockerfile`
- Fastest GitHub-to-public-URL path: Render with the included `render.yaml`
- `railway.json` is kept for credit-based Railway deploys, but Railway is no longer the best fit for a lasting free public demo

## Container Notes

- The container serves a bundled Lovable build from `/app/lovable-dist` when present.
- If no bundled dist exists, the app falls back to the legacy Flask templates in `templates/`.
- Include `model.pth` and `model_calibration.json` in the deployed repo so inference works.

Optional environment variables:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `SEMANTIC_OVERRIDE_ENABLED`
- `DEFAULT_ANALYSIS_PROFILE`
- `ALERT_MIN_THRESHOLD`

## Curated Training

- List approved data sources: `python sync_approved_sources.py --list`
- Sync automatic human corpora into the registry: `python sync_approved_sources.py --sync human_yesno human_librispeech_dev_clean --limit 50`
- Register manually downloaded Hindi human corpora: `python sync_approved_sources.py --register human_common_voice_hi --from-dir "C:\path\to\commonvoice_hi" --limit 300`
- Register manually downloaded AI datasets like WaveFake or ASVspoof: `python sync_approved_sources.py --register ai_wavefake --from-dir "C:\path\to\dataset" --limit 200`
- Train with approved sources plus generated English/Hindi/Hinglish clone audio: `python train_internet_model.py --approved_human_sources all --approved_ai_sources all`

See `docs/curated_sources.md` for the guarded data-ingestion flow.
