# Downloader for the midi files with cache
import os
from typing import IO, Iterable, List
import requests
import toml
from tqdm import tqdm
import zipfile

DATASETS_CONFIG_PATH = "datasets.toml"
CACHE_DIR = ".cs230_cache"
MIDI_DIR = "midi_data"

def _download(filename: str, url: str) -> str:
    """Download a zip file from a URL if it's not already in the cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(cache_path):
        with requests.get(url, stream=True) as r:
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            chunk_size = 1024
            with open(cache_path, "wb") as f:
                print(f"Downloading {filename} from {url}")
                progress = tqdm(total = total_size_in_bytes, unit = 'iB', unit_scale = True)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    progress.update(len(chunk))
                    f.write(chunk)
                progress.close()
    else:
        print("Using cached: ", filename)
    return cache_path

def download_all() -> List[str]:
    """Download all the zip files and return a list of their paths."""
    config = toml.load(DATASETS_CONFIG_PATH)
    return [_download(f"{filename}.zip", url) for filename, url in config["datasets"].items()]

def unzip_to_midi_files(archive_path: str) -> Iterable[IO[bytes]]:
    """read all MIDI files in the archive recursively."""
    # Use BFS to get all MIDI files in the archive
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            if info.filename.endswith(".mid"):
                yield zip_ref.open(info)

def midi_iterators() -> Iterable[IO[bytes]]:
    """Get an iterator over all MIDI files bytestreams."""
    for archive_path in download_all():
        yield from unzip_to_midi_files(archive_path)

if __name__ == "__main__":
    download_all()




