import os
import random
import asyncio
import edge_tts
from edge_tts import VoicesManager

# =============================================================================
# VACHA-SHIELD PHASE 17: DEEPFAKE SYNTHESIZER
# =============================================================================
# Generates high-fidelity Microsoft Azure TTS voice clones to train the CRNN
# on modern, flawlessly smooth AI temporal artifacts.
# =============================================================================

OUTPUT_DIR = "continuous_learning_dataset/ai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Provide a diverse block of conversational text to clone
TEXT_CORPUS = [
    "Hello everyone, thanks for joining the call today. We have a lot of important updates regarding the Q3 financials that I want to review.",
    "I need you to wire the remaining funds to the offshore account immediately. The compliance team is asking questions and we are running out of time.",
    "Hey mom, I'm in big trouble. I lost my phone and I need you to transfer some money to my friend's Venmo right now so I can buy a ticket home.",
    "The new software update will be deployed at midnight. Please ensure all your systems are backed up and disconnected from the main grid.",
    "I understand your concerns, but the executive board has already made the decision. We will be moving forward with the merger as planned.",
    "Did you see the news this morning? It looks like the market is going to crash again. I'm moving all my assets into liquid cash.",
    "Can you hear me? The connection is really bad. Anyway, the password for the admin server is Omega Alpha Seven Niner. Got that?",
    "We sincerely apologize for the delay. Your account has been temporarily frozen due to suspicious activity. Please verify your identity.",
    "It's great to finally meet you. I've heard a lot of amazing things about your work in the Artificial Intelligence sector.",
    "I am formally requesting a complete audit of the department's expenses. There are several anomalies in the Q2 ledger that don't add up."
]

async def generate_clones(num_clones=100):
    print(f"[*] Initializing Edge-TTS Deepfake Synthesizer...")
    
    # Get all available English voices from Azure
    voices = await VoicesManager.create()
    english_voices = voices.find(Language="en")
    voice_names = [v['Name'] for v in english_voices]
    
    print(f"[*] Found {len(voice_names)} high-fidelity Azure AI Voice Clones.")
    print(f"[*] Generating {num_clones} synthetic audio samples...\n")
    
    for i in range(num_clones):
        # Pick a random conversational sentence and a random AI voice
        sentence = random.choice(TEXT_CORPUS)
        voice = random.choice(voice_names)
        
        # Randomize speech rate to prevent the AI from just memorizing speeds
        rate = f"{random.randint(-15, +15):+d}%"
        # Randomize pitch slightly
        pitch = f"{random.randint(-10, +10):+d}Hz"
        
        # Save exact parameters so we can debug if needed
        filename = f"azure_clone_{i:03d}_{voice.split('-')[-1]}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        communicate = edge_tts.Communicate(text=sentence, voice=voice, rate=rate, pitch=pitch)
        
        # Generate and save the file
        await communicate.save(output_path)
        
        if (i+1) % 10 == 0:
            print(f"  -> Synthesized {i+1}/{num_clones} deepfakes | Last Voice: {voice}")

    print(f"\n[+] Successfully generated {num_clones} Deepfake AI Voice Clones!")
    print(f"    Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Generate 150 clones to heavily outnumber the human dataset.
    # We will use FocalLoss in the training script to balance the weighting.
    asyncio.run(generate_clones(num_clones=150))
