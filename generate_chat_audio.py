import os
from examples.generation import main as higgs_main
import json

json_file = "./chats/chat_history.json"
ref_audio = "./sample_voices/richa/richa.wav"
ref_text = './sample_voices/richa/richa.txt'
output_folder = "chat_audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load chat messages
with open(json_file, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# Iterate over messages and generate audio
user_messages = [msg["text"] for msg in chat_data if msg["user"] == 'richa']

for i, text in enumerate(user_messages):
    print(f"Generating audio for richa: {text}")
    output_path = os.path.join(output_folder, f"{i}_richa.wav")

    higgs_main(
        [
        f"--transcript={text}",
        f"--ref_audio={'richa'}",
        f"--out_path={output_path}",
        ]
    )

    print(f"Saved audio to {output_path}")
