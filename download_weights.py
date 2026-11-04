"""
download_weights.py
Download and verify YOLO pretrained weights to prevent corrupt checkpoint errors.
"""

import sys
from pathlib import Path
import urllib.request
import torch

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt"
TARGET_PATH = Path("yolov8n-seg.pt")

def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
    mb_down = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent:5.1f}% [{mb_down:.2f} MB / {mb_total:.2f} MB]")
    sys.stdout.flush()

def main():
    if TARGET_PATH.exists():
        print(f"Checking existing file {TARGET_PATH}...")
        try:
            torch.load(TARGET_PATH, map_location="cpu")
            print("Existing file is valid and ready to use!")
            return
        except Exception as e:
            print(f"Existing file is corrupted ({e}), re-downloading...")
            TARGET_PATH.unlink(missing_ok=True)

    print(f"Downloading pretrained weights from:\n  {WEIGHTS_URL}")
    urllib.request.urlretrieve(WEIGHTS_URL, TARGET_PATH, reporthook=progress_bar)
    print("\nDownload complete. Verifying PyTorch checkpoint integrity...")

    try:
        torch.load(TARGET_PATH, map_location="cpu")
        print(f"✓ Weight verification SUCCESSFUL! ({TARGET_PATH.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        TARGET_PATH.unlink(missing_ok=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
