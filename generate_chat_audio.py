import openai
import os
import wave
import json

# Make sure your BOSON_API_KEY is set
BOSON_API_KEY = os.getenv("BOSON_API_KEY")

client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

json_file = "./chats/chat_history.json"
transcript_file = "richa.txt"
ref_audio = "richa.wav"
output_folder = "chat_audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load chat messages
with open(json_file, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# Iterate over messages and generate audio
user_messages = [msg["text"] for msg in chat_data if msg["user"] == 'richa']

for i, text in enumerate(user_messages):
    print(f"Generating audio for richa: {text}")

    response = client.audio.speech.create(
        model="higgs-audio-generation-Hackathon",
        voice=ref_audio, 
        input=text,
        response_format="pcm"
    )

    pcm_data = response.content

    # Save to WAV
    filename = os.path.join(output_folder, f"{i}_{'richa'}.wav")
    with wave.open(filename, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm_data)

    print(f"Saved audio to {filename}")
