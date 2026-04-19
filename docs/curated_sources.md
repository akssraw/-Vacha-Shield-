# Curated Data Pipeline

Vacha-Shield now supports a guarded source registry so the detector can improve from approved human and AI-voice corpora instead of random internet scraping.

## Approved Human Sources

- `human_yesno` - OpenSLR YESNO (`automatic`)
- `human_librispeech_dev_clean` - LibriSpeech dev-clean (`automatic`)
- `human_common_voice` - Mozilla Common Voice (`manual`)
- `human_common_voice_hi` - Mozilla Common Voice Hindi (`manual`)
- `human_ai4bharat_kathbath_hi` - AI4Bharat Kathbath Hindi (`manual`)
- `human_hinglish_field_recordings` - custom Hinglish/code-mixed field recordings (`manual`)
- `human_vctk` - VCTK Corpus (`manual`)

## Approved AI Sources

- `ai_asvspoof_2021_df` - ASVspoof 2021 Deepfake (`manual`)
- `ai_wavefake` - WaveFake (`manual`)
- `ai_generated_edge_tts` - generated during training
- `ai_generated_premium_tts` - optional generated premium-provider voices during training

## Why This Is Safer

- only approved sources enter training
- manual datasets stay auditable and attributable
- generated AI samples are clearly separated from real corpora
- each training run writes a report under `data/training_runs/`

## Commands

List sources:

```bash
python sync_approved_sources.py --list
```

Sync automatic human sources:

```bash
python sync_approved_sources.py --sync human_yesno human_librispeech_dev_clean --limit 50
```

Register a manually downloaded dataset:

```bash
python sync_approved_sources.py --register ai_wavefake --from-dir "C:\path\to\wavefake_subset" --limit 200
```

Train using the registry plus generated English/Hindi/Hinglish clones:

```bash
python train_internet_model.py --approved_human_sources all --approved_ai_sources all
```

Skip registered AI corpora and train only on generated clones:

```bash
python train_internet_model.py --approved_ai_sources none
```

## Hindi And Hinglish

The generator now mixes English, Hindi, and Hinglish prompts by default. Use Hindi human corpora by registering `human_common_voice_hi`, `human_ai4bharat_kathbath_hi`, or your own `human_hinglish_field_recordings` folder.

Example manual registrations:

```bash
python sync_approved_sources.py --register human_common_voice_hi --from-dir "C:\path\to\commonvoice_hi" --limit 300
python sync_approved_sources.py --register human_ai4bharat_kathbath_hi --from-dir "C:\path\to\kathbath_hi" --limit 300
python sync_approved_sources.py --register human_hinglish_field_recordings --from-dir "C:\path\to\hinglish_recordings" --limit 300
```

If you ever want English-only synthetic generation again:

```bash
python train_internet_model.py --disable_multilingual_prompts
```
