import os
import pickle
import re
import sys


def get_output_root():
    """Dynamically determine OUTPUT_ROOT"""
    # Regex pattern to match folder names
    folder_pattern = re.compile(r"^(20[1-9]|21[0-5])-(\w*)$")
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    # Traverse upwards to find the first matching folder
    while not folder_pattern.match(os.path.basename(script_dir)):
        parent = os.path.dirname(script_dir)
        if parent == script_dir:  # Stop if we reach the root
            print("No matching project folder found. Using current directory.")
            break
        script_dir = parent

    print(f"Using project folder: {script_dir}")
    if folder_pattern.match(os.path.basename(script_dir)):
        return script_dir
    else:
        print(f"{script_dir} ne obstaja. Shranjeno v ROOT.")
        return os.path.abspath("./")


def save_results(vez_sez, eng_sez, states, filename):
    output_root = get_output_root()
    data_dir = os.path.join(output_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump({"vez_sez": vez_sez, "eng_sez": eng_sez, "states": states}, f)
    print(f"Results saved at: {save_path}")


def load_results(filename):
    output_root = get_output_root()
    data_dir = os.path.join(output_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    load_path = os.path.join(data_dir, filename)

    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            return data["vez_sez"], data["eng_sez"], data["states"]
    else:
        print(f"File {load_path} not found.")
        return None, None, None
