import requests
import numpy as np
import scipy.io.wavfile as wavfile
import os

# Create a 3 second dummy silent wav file
sr = 16000
audio = np.zeros(int(sr * 3.0), dtype=np.float32)
wavfile.write("dummy_test.wav", sr, audio)

url = "http://127.0.0.1:5000/detect_voice"

with open("dummy_test.wav", "rb") as f:
    files = {"file": ("ambient.webm", f, "audio/webm")}
    data = {"force_alert": "true"}
    print("Sending POST request to /detect_voice with force_alert=true...")
    response = requests.post(url, files=files, data=data)
    
print(f"Status Code: {response.status_code}")
print(f"JSON Response: {response.json()}")

os.remove("dummy_test.wav")
