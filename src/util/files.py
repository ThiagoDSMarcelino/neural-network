import shutil

def delete_dir(directory: str):
    try:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' successfully removed.")
    except OSError as e:
        print(f"Error: {e}")