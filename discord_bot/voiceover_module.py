import audio.multiple_chat_audios, audio.merge_audios
import os
import subprocess

def voiceover_script(funny_script):
    """
    Generates a wave file from the txt file when called, and puts it into folder.
    Runs multiple_chat_audios.py -> merge_audios.py -> returns True when done.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    audio_output_dir = os.path.join(base_dir, "../audio_output")
    final_merged_audio = os.path.join(base_dir, "../audio_final/top5.wav")

    # multiple_chat_audios.run(funny_script, audio_output_dir)
    # print("[Voiceover] Starting voice generation pipeline...")

    audio.merge_audios.run(audio_output_dir, final_merged_audio)
    print("[Voiceover] Top 5 generated")

    return True