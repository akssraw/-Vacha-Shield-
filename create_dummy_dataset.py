import os
import numpy as np
import soundfile as sf

# Dataset Paths
DATASET_DIR = "data/ASVspoof2019_LA_train"
FLAC_DIR = os.path.join(DATASET_DIR, "flac")
PROTOCOL_FILE = os.path.join(DATASET_DIR, "ASVspoof2019.LA.cm.train.trn.txt")

# Create directories
os.makedirs(FLAC_DIR, exist_ok=True)

# Generate synthetic data
print("Generating synthetic Hackathon ASVspoof Dataset...")

protocol_lines = []

# Generate 50 'bonafide' (human) and 50 'spoof' (AI) audio files
for i in range(100):
    speaker_id = f"LA_{i:04d}"
    audio_id = f"LA_T_{1000000 + i}"
    
    # Label
    label = "bonafide" if i < 50 else "spoof"
    
    # Audio Characteristics (To give the CNN something to learn)
    sr = 16000
    duration = 3.0 # seconds
    t = np.linspace(0, duration, int(sr * duration), False)
    
    if label == "bonafide":
        # Create a complex human-like waveform (low frequency harmonics + noise)
        tone = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
        noise = np.random.normal(0, 0.05, tone.shape)
        audio = tone + noise
    else:
        # Create an artificial robotic-like waveform (high frequency strict sine + less noise)
        tone = np.sin(2 * np.pi * 800 * t) + 0.8 * np.sin(2 * np.pi * 1600 * t)
        noise = np.random.normal(0, 0.01, tone.shape)
        audio = tone + noise
        
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save FLAC
    flac_path = os.path.join(FLAC_DIR, f"{audio_id}.flac")
    sf.write(flac_path, audio, sr)
    
    # Protocol Line
    # SPEAKER_ID AUDIO_FILE_ID SYSTEM_ID METHOD KEY
    protocol_line = f"{speaker_id} {audio_id} - - {label}\n"
    protocol_lines.append(protocol_line)

# Write Protocol File
with open(PROTOCOL_FILE, 'w') as f:
    f.writelines(protocol_lines)

print(f"Dataset generated at {DATASET_DIR}")
print(f"Total samples: 100")
