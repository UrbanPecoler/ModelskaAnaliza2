import os
import re
import sys

import matplotlib.pyplot as plt


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


def format_plot(ax=None, **kwargs):
    ax = ax or plt.gca()

    # Default values for formatting
    defaults = {
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "xscale": None,
        "yscale": None,
        "grid": None,
        "legend": None,
    }

    # Override defaults with provided kwargs
    defaults.update(kwargs)

    # Apply formatting only if values are not None
    if defaults["title"]:
        ax.set_title(defaults["title"])
    if defaults["xlabel"]:
        ax.set_xlabel(defaults["xlabel"])
    if defaults["ylabel"]:
        ax.set_ylabel(defaults["ylabel"])
    if defaults["xscale"]:
        ax.set_xscale(defaults["xscale"])
    if defaults["yscale"]:
        ax.set_yscale(defaults["yscale"])
    if defaults["grid"] is not None:
        ax.grid(defaults["grid"])
    if defaults["legend"]:
        ax.legend()


def savefigure(name):
    output_root = get_output_root()
    save_path = os.path.join(output_root, "Images", name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved at: {save_path}")
