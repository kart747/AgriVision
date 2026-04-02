"""
Merge stock photos into existing training dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STOCK_DIR = ROOT / "TestData" / "stock_photos"
TRAIN_DIR = ROOT / "TestData" / "merged_dataset" / "train"


def main():
    classes = ["apple_healthy", "apple_rust", "apple_scab"]
    
    for class_name in classes:
        stock_class_dir = STOCK_DIR / class_name
        train_class_dir = TRAIN_DIR / class_name
        
        if not stock_class_dir.exists():
            print(f"Stock dir not found: {stock_class_dir}")
            continue
        
        if not train_class_dir.exists():
            print(f"Train dir not found: {train_class_dir}")
            continue
        
        stock_images = list(stock_class_dir.glob("*.*"))
        
        print(f"\n{class_name}:")
        print(f"  Stock images: {len(stock_images)}")
        print(f"  Before merge: {len(list(train_class_dir.glob('*.*')))} images")
        
        for img in stock_images:
            dest = train_class_dir / f"stock_{img.name}"
            shutil.copy2(img, dest)
        
        after_count = len(list(train_class_dir.glob("*.*")))
        print(f"  After merge: {after_count} images")
        print(f"  Added: {len(stock_images)} new images")


if __name__ == "__main__":
    main()
