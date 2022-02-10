from pathlib import Path


def get_folder_size(d):
    return sum(f.stat().st_size for f in Path(d).glob('**/*') if f.is_file())
