import pyaudio
import wave
import requests
import time
import os

# --- Configuration ---
API_URL = "http://localhost:5000/detect_voice"
TEMP_FILE = "temp_mic.wav"

# Audio Recording Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Use 16kHz directly to match ASVspoof/pipeline expectations
RECORD_SECONDS = 3 # Length of each listening chunk

def record_audio(filename):
    """Records audio from the default microphone and saves it to a WAV file."""
    p = pyaudio.PyAudio()

    print(f"\n[🎙️] Listening for {RECORD_SECONDS} seconds...")
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return True
        
    except OSError as e:
        print(f"\n[!] Microphone Error: {e}")
        print("Please ensure you have a working microphone and PyAudio is installed correctly.")
        p.terminate()
        return False

def analyze_audio(filename):
    """Sends the recorded WAV file to the Vacha-Shield API."""
    print("[⚙️] Analyzing voice signature...")
    try:
        with open(filename, 'rb') as f:
            files = {'file': f}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            syn_prob = result['synthetic_probability'] * 100
            hum_prob = result['human_probability'] * 100
            
            # Print the result cleanly
            print("-" * 40)
            print(f"Human: {hum_prob:>6.2f}% | AI/Deepfake: {syn_prob:>6.2f}%")
            print("-" * 40)
            
            # Trigger the Hackathon Visual Alert
            if result['alert']:
                print("\n🚨🚨🚨 DEEPFAKE DETECTED 🚨🚨🚨")
                print("WARNING: THE VOICE ON THE LINE IS HIGHLY LIKELY TO BE SYNTHETIC!")
                print("🚨🚨🚨 ================= 🚨🚨🚨\n")
            else:
                print("✅ Status: Safe (Human Voice Verified)")
                
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n[!] Connection Error: Cannot reach the Vacha-Shield API.")
        print("Ensure the Flask app is running (python app.py) in a separate terminal.")
        return False
        
    return True

if __name__ == "__main__":
    print("=" * 50)
    print(" Vacha-Shield - Real-Time Prototype Terminal")
    print("=" * 50)
    print("Press Ctrl+C to exit at any time.\n")
    
    try:
        while True:
            # 1. Record
            success = record_audio(TEMP_FILE)
            
            if not success:
                break
                
            # 2. Analyze
            api_up = analyze_audio(TEMP_FILE)
            
            if not api_up:
                break
                
            # 3. Brief pause before next cycle
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Real-Time Client Terminated by User.")
    finally:
        # Cleanup
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
