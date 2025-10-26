import os
from generate_chat_audio import generate_audio

# input/output setup
text_file = "./discord_bot/LLM_generate/funny_script.txt"
output_folder = "chat_audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# voice mapping for each username
VOICE_MAP = {
    "Narrator": "narrator",
    "lineee.xt": "lynn",
    "a_licee": "alice",
    "reee.na": "serena",
    "richa._1": "richa", 
}

# --- read the transcript ---
with open(text_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

index = 1

# --- parse and generate audio ---
for line in lines:
    line = line.strip()
    if not line or not line.startswith("["):
        continue  # skip empty lines or commentary

    # extract username and message
    if "]" in line:
        username = line.split("]")[0][1:].strip()
        message = line.split("]", 1)[1].strip().strip('"')
    else:
        continue

    # skip if no message or unknown user
    if not message or username not in VOICE_MAP:
        username = "Narrator"

    voice = VOICE_MAP[username]

    print(f"Generating audio for {username}: {message}")
    generate_audio(voice, message, output_folder, index)
    index += 1
