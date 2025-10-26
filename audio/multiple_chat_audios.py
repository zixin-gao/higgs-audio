import os
from audio.single_chat_audio import generate_audio


def run(text_str, audio_output_folder):
    os.makedirs(audio_output_folder, exist_ok=True)

    # voice mapping for each username
    VOICE_MAP = {
        "Narrator": "narrator",
        "lineee.xt": "lynn",
        "a_licee": "alice",
        "reee.na": "serena",
        "richa._1": "richa", 
    }

    # --- read the transcript ---
    lines = text_str.strip().split("\n")


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
        generate_audio(voice, message, audio_output_folder, index)
        index += 1

