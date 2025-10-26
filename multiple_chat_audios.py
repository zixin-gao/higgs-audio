import os
import json
from generate_chat_audio import generate_audio

json_file = "./chats/chat_history.json"
target_user = 'richa'
output_folder = "chat_audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load chat messages
with open(json_file, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# Iterate over messages and generate audio
user_messages = [msg["text"] for msg in chat_data if msg["user"] == f'{target_user}']

for msg in user_messages:
    generate_audio(target_user, msg, output_folder)
