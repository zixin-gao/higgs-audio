import os
from pydub import AudioSegment

input_folder = "chat_audio_outputs"
output_file = os.path.join("audio_output", "top5.wav")

# --- collect and sort wav files by numeric prefix ---
wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
wav_files.sort(key=lambda x: int(x.split("_")[0]))  # sort by prefix number

# --- merge them sequentially ---
combined = AudioSegment.empty()

for wav in wav_files:
    path = os.path.join(input_folder, wav)
    print(f"Adding: {wav}")
    segment = AudioSegment.from_wav(path)
    combined += segment  # concatenate

# --- export final file ---
combined.export(output_file, format="wav")
print(f"Merged")
