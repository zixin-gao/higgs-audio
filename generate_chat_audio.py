import os
import subprocess
import json

def generate_audio(target_user, text, output_folder, prefix):

    output_path = os.path.join(output_folder, f"{prefix}_{target_user}.wav")
    
    print(f"Generating audio for {target_user}: {text}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    subprocess.run([
        "python", "examples/generation.py",
        f"--transcript={text}",
        f"--ref_audio={target_user}",
        f"--out_path={output_path}",
        ], env=env, check=True)

    print(f"Saved audio to {output_path}")
    return output_folder

