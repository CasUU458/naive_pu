import pandas as pd
import os 
from pathlib import Path
import datetime

name = "c"

def load_experiment_results(name,n=0):
    try:
        folder_path = f"EXPERIMENTS/{name}"
        assert os.path.exists(folder_path), f"Folder not found: {folder_path}"
        directory = Path(folder_path)
    except:
        #go up one directory
        directory = Path(os.getcwd()).parent
        directory = os.path.join(directory, f"EXPERIMENTS/{name}")
        directory = Path(directory)

    # Collect files and their modification times
    files = [(f, f.stat().st_mtime) for f in directory.iterdir() if f.is_file()]

    # Sort by modification time (oldest first)
    files_sorted = sorted(files, key=lambda x: x[1])[::-1]  # Newest first
    files = [str(f) for f, _ in files_sorted]
    final_files = []
    for f in files:
        if f.split("/")[-1].startswith("final"):
            final_files.append(f)
    final = final_files[n] if final_files else files[n]

    df = pd.read_pickle(final)
    return df


