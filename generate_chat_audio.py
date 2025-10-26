import os
import subprocess
import json
from examples.generation import main as generate_main, main_new

# def generate_audio(target_user, text, output_folder, prefix, client):

#     output_path = os.path.join(output_folder, f"{prefix}_{target_user}.wav")
    
#     print(f"Generating audio for {target_user}: {text}")

#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.path.abspath(".")

#     subprocess.run([
#         "python", "examples/generation.py",
#         f"--transcript={text}",
#         f"--ref_audio={target_user}",
#         f"--out_path={output_path}",
#         f"--model_client={client}"
#         ], env=env, check=True)

#     print(f"Saved audio to {output_path}")
#     return output_folder


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
        chunk_method="speaker",
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
