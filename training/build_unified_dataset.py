"""
Build a unified classification dataset from PlantVillage and PlantDoc object detection data.

This script:
1. Copies mapped PlantVillage classes into a pooled class bucket.
2. Converts PlantDoc detections into cropped classification images.
3. Optionally downsamples PlantVillage per class to preserve PlantDoc representation.
4. Splits the merged pool into train/val/test folders.
5. Writes a JSON summary report.

Example:
  python training/build_unified_dataset.py \
      --plantvillage-dir /path/to/plantvillage/color \
      --plantdoc-images /path/to/plantdoc/images \
      --plantdoc-labels /path/to/plantdoc/labels \
      --plantdoc-classes /path/to/plantdoc/classes.txt \
      --output-dir /path/to/merged_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Mapped overlap classes only. Intentionally excludes classes without clean overlap.
PLANTVILLAGE_TO_UNIFIED: Dict[str, str] = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Cedar_apple_rust": "apple_rust",
    "Apple___healthy": "apple_healthy",
    "Grape___Black_rot": "grape_black_rot",
    "Grape___healthy": "grape_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___healthy": "tomato_healthy",
}


# PlantDoc class names are often inconsistent across mirrors; aliases make mapping resilient.
PLANTDOC_TO_UNIFIED: Dict[str, str] = {
    "Apple Scab Leaf": "apple_scab",
    "Apple rust leaf": "apple_rust",
    "Apple leaf": "apple_healthy",
    "grape leaf black rot": "grape_black_rot",
    "grape leaf": "grape_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato mold leaf": "tomato_leaf_mold",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato leaf yellow virus": "tomato_yellow_leaf_curl_virus",
    "Tomato two spotted spider mites leaf": "tomato_spider_mites",
    "Tomato leaf": "tomato_healthy",
}


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


PLANTDOC_TO_UNIFIED_NORMALIZED: Dict[str, str] = {
    _normalize_label(k): v for k, v in PLANTDOC_TO_UNIFIED.items()
}


@dataclass
class SampleRecord:
    class_name: str
    source: str  # "plantvillage" or "plantdoc"
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged classification dataset from PlantVillage and PlantDoc")
    parser.add_argument("--plantvillage-dir", type=Path, required=True, help="Path to PlantVillage class folders")
    parser.add_argument("--plantdoc-images", type=Path, required=True, help="Path to PlantDoc images directory")
    parser.add_argument("--plantdoc-labels", type=Path, required=True, help="Path to PlantDoc labels directory")
    parser.add_argument(
        "--plantdoc-classes",
        type=Path,
        default=None,
        help="Optional classes.txt (one class per line) for YOLO labels",
    )
    parser.add_argument(
        "--plantdoc-format",
        choices=["auto", "yolo", "voc"],
        default="auto",
        help="Annotation format for PlantDoc labels",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output merged dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pv-per-class",
        type=int,
        default=None,
        help="Optional hard cap for PlantVillage samples per class",
    )
    parser.add_argument(
        "--pv-to-pd-max-ratio",
        type=float,
        default=4.0,
        help="Max PlantVillage:PlantDoc ratio per class (used when PlantDoc exists)",
    )
    parser.add_argument("--bbox-padding", type=float, default=0.05, help="Padding fraction around PlantDoc bbox")
    parser.add_argument("--min-crop-size", type=int, default=32, help="Minimum crop width/height to keep")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing output files")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if any(r <= 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must all be > 0")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def load_yolo_classes(classes_path: Optional[Path]) -> Dict[int, str]:
    if classes_path is None:
        return {}
    lines = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {i: name for i, name in enumerate(lines)}


def detect_label_format(labels_dir: Path) -> str:
    txt_count = len(list(labels_dir.rglob("*.txt")))
    xml_count = len(list(labels_dir.rglob("*.xml")))
    if txt_count > 0 and xml_count == 0:
        return "yolo"
    if xml_count > 0 and txt_count == 0:
        return "voc"
    if txt_count > 0 and xml_count > 0:
        return "yolo"
    raise ValueError("No .txt or .xml annotation files found in PlantDoc labels directory")


def match_image_for_label(label_path: Path, images_dir: Path) -> Optional[Path]:
    stem = label_path.stem
    parent_rel = label_path.parent
    for ext in IMAGE_EXTENSIONS:
        candidate_same_tree = images_dir / parent_rel.relative_to(parent_rel.anchor if parent_rel.is_absolute() else Path(".")) / f"{stem}{ext}"
        if candidate_same_tree.exists():
            return candidate_same_tree
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.rglob(f"{stem}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTENSIONS:
            return match
    return None


def yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    image_w: int,
    image_h: int,
    padding_frac: float,
) -> Tuple[int, int, int, int]:
    x1 = (xc - w / 2.0) * image_w
    y1 = (yc - h / 2.0) * image_h
    x2 = (xc + w / 2.0) * image_w
    y2 = (yc + h / 2.0) * image_h

    pad_x = (x2 - x1) * padding_frac
    pad_y = (y2 - y1) * padding_frac

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_w, int(round(x2 + pad_x)))
    y2 = min(image_h, int(round(y2 + pad_y)))
    return x1, y1, x2, y2


def map_plantdoc_class(raw_name: str) -> Optional[str]:
    return PLANTDOC_TO_UNIFIED_NORMALIZED.get(_normalize_label(raw_name))


def ingest_plantvillage(
    plantvillage_dir: Path,
    pooled_dir: Path,
    dry_run: bool,
) -> Dict[str, List[SampleRecord]]:
    result: Dict[str, List[SampleRecord]] = defaultdict(list)
    for pv_class, unified_class in PLANTVILLAGE_TO_UNIFIED.items():
        class_dir = plantvillage_dir / pv_class
        images = list_images(class_dir)
        for idx, src in enumerate(images):
            dst = pooled_dir / unified_class / f"pv_{idx:07d}{src.suffix.lower()}"
            if not dry_run:
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
            record_path = dst if not dry_run else src
            result[unified_class].append(SampleRecord(unified_class, "plantvillage", record_path))
    return result


def ingest_plantdoc_yolo(
    images_dir: Path,
    labels_dir: Path,
    classes_by_id: Dict[int, str],
    pooled_dir: Path,
    min_crop_size: int,
    bbox_padding: float,
    dry_run: bool,
) -> Tuple[Dict[str, List[SampleRecord]], Counter]:
    result: Dict[str, List[SampleRecord]] = defaultdict(list)
    stats = Counter()
    label_files = list(labels_dir.rglob("*.txt"))

    for label_path in label_files:
        img_path = match_image_for_label(label_path, images_dir)
        if img_path is None:
            stats["missing_image_for_label"] += 1
            continue

        try:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                image_w, image_h = rgb.size
                lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

                for box_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        stats["invalid_yolo_line"] += 1
                        continue

                    class_id = int(float(parts[0]))
                    class_name = classes_by_id.get(class_id)
                    if class_name is None:
                        stats["missing_class_id_name"] += 1
                        continue

                    unified = map_plantdoc_class(class_name)
                    if unified is None:
                        stats["unmapped_plantdoc_class"] += 1
                        continue

                    xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, image_w, image_h, bbox_padding)

                    if x2 <= x1 or y2 <= y1:
                        stats["invalid_bbox"] += 1
                        continue
                    if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                        stats["small_crop_rejected"] += 1
                        continue

                    dst = pooled_dir / unified / f"pd_{img_path.stem}_{box_idx:03d}.jpg"
                    if not dry_run:
                        ensure_dir(dst.parent)
                        crop = rgb.crop((x1, y1, x2, y2))
                        crop.save(dst, quality=95)
                        record_path = dst
                    else:
                        record_path = img_path

                    result[unified].append(SampleRecord(unified, "plantdoc", record_path))
                    stats["plantdoc_crops_kept"] += 1
        except Exception:
            stats["plantdoc_image_read_error"] += 1

    return result, stats


def ingest_plantdoc_voc(
    images_dir: Path,
    labels_dir: Path,
    pooled_dir: Path,
    min_crop_size: int,
    bbox_padding: float,
    dry_run: bool,
) -> Tuple[Dict[str, List[SampleRecord]], Counter]:
    result: Dict[str, List[SampleRecord]] = defaultdict(list)
    stats = Counter()
    label_files = list(labels_dir.rglob("*.xml"))

    for label_path in label_files:
        img_path = match_image_for_label(label_path, images_dir)
        if img_path is None:
            stats["missing_image_for_label"] += 1
            continue

        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                image_w, image_h = rgb.size
                objects = root.findall("object")
                for obj_idx, obj in enumerate(objects):
                    cls = obj.findtext("name", default="").strip()
                    unified = map_plantdoc_class(cls)
                    if unified is None:
                        stats["unmapped_plantdoc_class"] += 1
                        continue

                    bnd = obj.find("bndbox")
                    if bnd is None:
                        stats["invalid_bbox"] += 1
                        continue

                    x1 = int(float(bnd.findtext("xmin", "0")))
                    y1 = int(float(bnd.findtext("ymin", "0")))
                    x2 = int(float(bnd.findtext("xmax", "0")))
                    y2 = int(float(bnd.findtext("ymax", "0")))

                    pad_x = int((x2 - x1) * bbox_padding)
                    pad_y = int((y2 - y1) * bbox_padding)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(image_w, x2 + pad_x)
                    y2 = min(image_h, y2 + pad_y)

                    if x2 <= x1 or y2 <= y1:
                        stats["invalid_bbox"] += 1
                        continue
                    if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                        stats["small_crop_rejected"] += 1
                        continue

                    dst = pooled_dir / unified / f"pd_{img_path.stem}_{obj_idx:03d}.jpg"
                    if not dry_run:
                        ensure_dir(dst.parent)
                        crop = rgb.crop((x1, y1, x2, y2))
                        crop.save(dst, quality=95)
                        record_path = dst
                    else:
                        record_path = img_path

                    result[unified].append(SampleRecord(unified, "plantdoc", record_path))
                    stats["plantdoc_crops_kept"] += 1
        except Exception:
            stats["plantdoc_label_parse_error"] += 1

    return result, stats


def balance_pool(
    per_class: Dict[str, List[SampleRecord]],
    max_pv_per_class: Optional[int],
    pv_to_pd_max_ratio: float,
    seed: int,
) -> Dict[str, List[SampleRecord]]:
    rng = random.Random(seed)
    balanced: Dict[str, List[SampleRecord]] = {}

    for class_name, records in per_class.items():
        pv = [r for r in records if r.source == "plantvillage"]
        pd = [r for r in records if r.source == "plantdoc"]
        rng.shuffle(pv)

        pv_limit = len(pv)
        if max_pv_per_class is not None:
            pv_limit = min(pv_limit, max_pv_per_class)
        if len(pd) > 0:
            ratio_cap = int(len(pd) * pv_to_pd_max_ratio)
            pv_limit = min(pv_limit, max(1, ratio_cap))

        balanced[class_name] = pv[:pv_limit] + pd
        rng.shuffle(balanced[class_name])

    return balanced


def split_records(
    records: Sequence[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def copy_into_split(
    split_name: str,
    records: Iterable[SampleRecord],
    output_dir: Path,
    dry_run: bool,
) -> int:
    written = 0
    for idx, rec in enumerate(records):
        suffix = rec.path.suffix.lower() if rec.path.suffix else ".jpg"
        dst = output_dir / split_name / rec.class_name / f"{rec.source}_{idx:08d}{suffix}"
        if not dry_run:
            ensure_dir(dst.parent)
            if rec.path.exists():
                shutil.copy2(rec.path, dst)
        written += 1
    return written


def build_summary(
    pooled: Dict[str, List[SampleRecord]],
    balanced: Dict[str, List[SampleRecord]],
    split_counts: Dict[str, Dict[str, int]],
    plantdoc_stats: Counter,
    args: argparse.Namespace,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "max_pv_per_class": args.max_pv_per_class,
            "pv_to_pd_max_ratio": args.pv_to_pd_max_ratio,
            "bbox_padding": args.bbox_padding,
            "min_crop_size": args.min_crop_size,
            "dry_run": args.dry_run,
        },
        "per_class": {},
        "plantdoc_stats": dict(plantdoc_stats),
    }

    per_class = {}
    for class_name in sorted(set(list(pooled.keys()) + list(balanced.keys()))):
        pooled_records = pooled.get(class_name, [])
        balanced_records = balanced.get(class_name, [])
        pooled_pv = sum(1 for r in pooled_records if r.source == "plantvillage")
        pooled_pd = sum(1 for r in pooled_records if r.source == "plantdoc")
        final_pv = sum(1 for r in balanced_records if r.source == "plantvillage")
        final_pd = sum(1 for r in balanced_records if r.source == "plantdoc")

        per_class[class_name] = {
            "pooled": {
                "total": len(pooled_records),
                "plantvillage": pooled_pv,
                "plantdoc": pooled_pd,
            },
            "balanced": {
                "total": len(balanced_records),
                "plantvillage": final_pv,
                "plantdoc": final_pd,
            },
            "split": split_counts.get(class_name, {"train": 0, "val": 0, "test": 0}),
        }
    summary["per_class"] = per_class
    return summary


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.plantvillage_dir.exists():
        raise FileNotFoundError(f"PlantVillage dir not found: {args.plantvillage_dir}")
    if not args.plantdoc_images.exists():
        raise FileNotFoundError(f"PlantDoc images dir not found: {args.plantdoc_images}")
    if not args.plantdoc_labels.exists():
        raise FileNotFoundError(f"PlantDoc labels dir not found: {args.plantdoc_labels}")

    output_dir = args.output_dir
    pooled_dir = output_dir / "_pooled"

    if output_dir.exists() and not args.dry_run:
        shutil.rmtree(output_dir)
    if not args.dry_run:
        ensure_dir(output_dir)
        ensure_dir(pooled_dir)

    print("[1/5] Ingesting PlantVillage mapped classes...")
    pv_records = ingest_plantvillage(args.plantvillage_dir, pooled_dir, args.dry_run)

    print("[2/5] Ingesting PlantDoc and generating crops...")
    label_format = args.plantdoc_format if args.plantdoc_format != "auto" else detect_label_format(args.plantdoc_labels)

    if label_format == "yolo":
        class_id_to_name = load_yolo_classes(args.plantdoc_classes)
        if not class_id_to_name:
            raise ValueError("YOLO format requires --plantdoc-classes with class names by index")
        pd_records, pd_stats = ingest_plantdoc_yolo(
            images_dir=args.plantdoc_images,
            labels_dir=args.plantdoc_labels,
            classes_by_id=class_id_to_name,
            pooled_dir=pooled_dir,
            min_crop_size=args.min_crop_size,
            bbox_padding=args.bbox_padding,
            dry_run=args.dry_run,
        )
    else:
        pd_records, pd_stats = ingest_plantdoc_voc(
            images_dir=args.plantdoc_images,
            labels_dir=args.plantdoc_labels,
            pooled_dir=pooled_dir,
            min_crop_size=args.min_crop_size,
            bbox_padding=args.bbox_padding,
            dry_run=args.dry_run,
        )

    pooled_by_class: Dict[str, List[SampleRecord]] = defaultdict(list)
    for cls, recs in pv_records.items():
        pooled_by_class[cls].extend(recs)
    for cls, recs in pd_records.items():
        pooled_by_class[cls].extend(recs)

    # Ensure all expected classes exist in output report, even if empty.
    for cls in sorted(set(PLANTVILLAGE_TO_UNIFIED.values())):
        pooled_by_class.setdefault(cls, [])

    print("[3/5] Applying class-wise balancing...")
    balanced_by_class = balance_pool(
        per_class=pooled_by_class,
        max_pv_per_class=args.max_pv_per_class,
        pv_to_pd_max_ratio=args.pv_to_pd_max_ratio,
        seed=args.seed,
    )

    print("[4/5] Splitting into train/val/test...")
    split_counts: Dict[str, Dict[str, int]] = {}
    for class_name, records in balanced_by_class.items():
        train, val, test = split_records(records, args.train_ratio, args.val_ratio, args.seed)
        split_counts[class_name] = {"train": len(train), "val": len(val), "test": len(test)}

        copy_into_split("train", train, output_dir, args.dry_run)
        copy_into_split("val", val, output_dir, args.dry_run)
        copy_into_split("test", test, output_dir, args.dry_run)

    if not args.dry_run:
        shutil.rmtree(pooled_dir, ignore_errors=True)

    print("[5/5] Writing summary report...")
    summary = build_summary(
        pooled=pooled_by_class,
        balanced=balanced_by_class,
        split_counts=split_counts,
        plantdoc_stats=pd_stats,
        args=args,
    )

    report_path = output_dir / "merge_report.json"
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Label format: {label_format}")
    print(f"Report: {report_path}")
    for class_name in sorted(summary["per_class"].keys()):
        info = summary["per_class"][class_name]
        split = info["split"]
        print(
            f"  - {class_name}: total={info['balanced']['total']} "
            f"(pv={info['balanced']['plantvillage']}, pd={info['balanced']['plantdoc']}) "
            f"-> train={split['train']}, val={split['val']}, test={split['test']}"
        )


if __name__ == "__main__":
    main()
