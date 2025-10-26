import os
from pydub import AudioSegment

def normalize_audio(segment, target_dBFS=-20.0):
    """normalize the audio segment to target loudness"""
    change_in_dBFS = target_dBFS - segment.dBFS
    return segment.apply_gain(change_in_dBFS)

def run(input_folder, output_file):
    # --- collect and sort wav files by numeric prefix ---
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
    wav_files.sort(key=lambda x: int(x.split("_")[0]))  # sort by prefix number

    if not wav_files:
        print("[merge_audios] no wav files found.")
        return

    combined = AudioSegment.empty()

    for wav in wav_files:
        path = os.path.join(input_folder, wav)
        print(f"[merge_audios] adding: {wav}")
        segment = AudioSegment.from_wav(path)
        normalized = normalize_audio(segment)
        combined += normalized

    # --- export final file ---
    combined.export(output_file, format="wav")
    print(f"[merge_audios] merged and normalized audio saved to: {output_file}")
