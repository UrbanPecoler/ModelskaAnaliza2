import glob
import os
import time

IMPORTS = [
    "pd.set_option('display.max_columns', 50)",
    "mpl.style.use('./porocilo.mplstyle')",
    "# General file settings",
    "   ",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import matplotlib as mpl",
    "import pandas as pd",
    "# Standard imports",
]


def get_recent_python_files(cutoff_minutes=5):
    """Get .py files created in the last `cutoff_minutes` minutes."""
    cutoff = time.time() - (cutoff_minutes * 60)  # Time in seconds
    return [
        f
        for f in glob.glob("**/*.py", recursive=True)  # Find all .py files
        if os.stat(f).st_birthtime > cutoff and "venv/" not in f
        # Compare creation time and ignore venv
    ]


def add_imports_to_file(file_path):
    """Add necessary imports if missing."""
    if not os.path.exists(file_path):
        return

    with open(file_path, "r+") as f:
        content = f.read()
        added = False
        for imp in IMPORTS:
            if imp not in content:
                content = imp + "\n" + content
                added = True

        if added:
            f.seek(0)
            f.write(content)
            f.truncate()
            print(f"✅ Updated: {file_path}")


if __name__ == "__main__":
    recent_files = get_recent_python_files()

    if not recent_files:
        print("No recently created Python files found.")
    else:
        for file in recent_files:
            add_imports_to_file(file)
