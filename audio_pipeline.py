import os
import torch
from examples.generation import HiggsAudioModelClient
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from examples.generation import main as generate_main, main_new


def generate_audio(voice, text, output_folder, index, model_client, model_path):
    output_path = os.path.join(output_folder, f"{index}_{voice}.wav")
    print(f"Generating audio for {voice}: {text}")

    # build args for the function
    main_new(
        model_client=model_client,
        model_path=model_path,
        audio_tokenizer=model_client._audio_tokenizer,
        max_new_tokens=model_client._max_new_tokens,
        transcript=text,
        scene_prompt=None,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        ras_win_len=None,
        ras_win_max_num_repeat=None,
        ref_audio=voice,
        ref_audio_in_system_message=True,
        chunk_method="none",
        chunk_max_word_num=0,
        chunk_max_num_turns=0,
        generation_chunk_buffer_size=0,
        seed=42,
        device_id=model_client._device_id,
        out_path=output_path,
        use_static_kv_cache=True,
        device=model_client._device,
    )

    print(f"Saved audio to {output_path}")
    return output_path


# input/output setup
text_file = "./chats/chat_history.txt"
output_folder = "chat_audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# voice mapping for each username
VOICE_MAP = {
    "Narrator": "Narrator",
    "lineee.xt": "lynn",
    "a_licee": "alice",
    "reee.na": "serena",
    "richa": "richa", 
}

# --- read the transcript ---
with open(text_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# pre-load the model
device_id = None
audio_tokenizer = "bosonai/higgs-audio-v2-tokenizer"
use_static_kv_cache = 1
model_path = "bosonai/higgs-audio-v2-generation-3B-base"
max_new_tokens = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"

# For MPS, use CPU for audio tokenizer due to embedding operation limitations
audio_tokenizer_device = "cpu" if device == "mps" else device
audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)

# Disable static KV cache on MPS since it relies on CUDA graphs
if device == "mps" and use_static_kv_cache:
    use_static_kv_cache = False
model_client = HiggsAudioModelClient(
    model_path=model_path,
    audio_tokenizer=audio_tokenizer,
    device=device,
    device_id=0 if device == "cuda" else None,
    max_new_tokens=max_new_tokens,
    use_static_kv_cache=use_static_kv_cache,
)



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
    generate_audio(voice, message, output_folder, index, model_client=model_client, model_path=model_path)
    index += 1
