import audio.multiple_chat_audios, audio.merge_audios
import os
import subprocess

def voiceover_script(funny_script):
    """
    Generates a wave file from the txt file when called, and puts it into folder.
    Runs multiple_chat_audios.py -> merge_audios.py -> returns True when done.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "../outputs")
    raw_output_dir = os.path.join(output_dir, "raw_output")
    final_audio_dir = os.path.join(output_dir, "final_output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(final_audio_dir, exist_ok=True)

    final_audio_path = os.path.join(final_audio_dir, "top5.wav")

    # multiple_chat_audios.run(funny_script, audio_output_dir)
    # print("[Voiceover] Starting voice generation pipeline...")

    audio.merge_audios.run(raw_output_dir, final_audio_path)
    print("[Voiceover] Top 5 generated")

    return True