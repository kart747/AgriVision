"""
Download apple stock images from Wikimedia Commons.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from PIL import Image


STOCK_IMAGES = {
    "apple_healthy": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Apple_leaves.jpg/800px-Apple_leaves.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Apple_tree.jpg/800px-Apple_tree.jpg",
    ],
    "apple_scab": [
        "https://upload.wikimedia.org/wikipedia/commons/1/15/Apple_scab_SEM.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Apple_scab.jpg/800px-Apple_scab.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Apple_scab_in_Tashkent_region.jpg/800px-Apple_scab_in_Tashkent_region.jpg",
    ],
    "apple_rust": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Apple_rust_%28Marssonina_carparpa%29_on_apple_leaf.jpg/800px-Apple_rust_%28Marssonina_carparpa%29_on_apple_leaf.jpg",
    ],
}


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "TestData" / "stock_photos"


def download_image(url: str, save_path: Path) -> bool:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        time.sleep(1.5)
        response = requests.get(url, timeout=20, headers=headers)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed: {url[:50]}... - {e}")
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    
    for class_name, urls in STOCK_IMAGES.items():
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"\n{class_name}:")
        for i, url in enumerate(urls):
            filename = f"{class_name}_stock_{i+1}.jpg"
            save_path = class_dir / filename
            
            print(f"  Downloading {url[:60]}...")
            if download_image(url, save_path):
                print(f"    Saved: {filename}")
                downloaded_count += 1
            else:
                print(f"    Failed!")
            time.sleep(1)
    
    print(f"\n\nTotal downloaded: {downloaded_count}")


if __name__ == "__main__":
    main()
