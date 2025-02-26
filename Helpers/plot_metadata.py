import os

import matplotlib.pyplot as plt

from Helpers.io_functions import get_output_root


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
