#!/usr/bin/env python3
"""
Convert an ImageFolder dataset to MXNet RecordIO format (train.rec + train.idx).

Usage:
    python3 convert_to_rec.py /path/to/image_folder /path/to/output_dir [--num-workers 8] [--quality 95]

The image folder must have structure:
    image_folder/
        class_0/
            img1.jpg
            img2.jpg
        class_1/
            ...

Output:
    output_dir/train.rec
    output_dir/train.idx

The header record (index 0) stores:
    header.flag = 1
    header.label = [num_images + 1, num_classes]
so that MXFaceDataset reads indices 1..num_images.
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import mxnet as mx
import numpy as np
from tqdm import tqdm


def scan_dataset(root_dir):
    """Scan the image folder and return (image_paths, labels, num_classes)."""
    root = Path(root_dir)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)

    print(f"Found {num_classes} identity folders. Scanning files...")

    image_paths = []
    labels = []
    for label_id, class_dir in enumerate(tqdm(class_dirs, desc="Scanning folders")):
        files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        ])
        for f in files:
            image_paths.append(str(f))
            labels.append(label_id)

    print(f"Total images: {len(image_paths)}, Total classes: {num_classes}")
    return image_paths, labels, num_classes


def read_image_bytes(path):
    """Read raw file bytes (no decode/re-encode needed if already JPEG)."""
    with open(path, 'rb') as f:
        return f.read()


def build_rec(image_paths, labels, num_classes, output_dir, quality=95):
    """Build train.rec and train.idx with a proper header record at index 0."""
    os.makedirs(output_dir, exist_ok=True)
    idx_path = os.path.join(output_dir, 'train.idx')
    rec_path = os.path.join(output_dir, 'train.rec')

    num_images = len(image_paths)

    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    # Write header at index 0:
    #   header.flag = 1 (or 2), header.label = [num_images+1, num_classes]
    #   This tells MXFaceDataset to read indices 1..num_images
    header0 = mx.recordio.IRHeader(flag=1, label=[num_images + 1, num_classes], id=0, id2=0)
    packed = mx.recordio.pack(header0, b'')
    record.write_idx(0, packed)

    print(f"\nWriting {num_images} images to RecordIO...")
    failed = 0
    for i in tqdm(range(num_images), desc="Building RecordIO"):
        try:
            img_bytes = read_image_bytes(image_paths[i])
            # Header for each image: flag=0, label=class_id
            header = mx.recordio.IRHeader(flag=0, label=float(labels[i]), id=0, id2=0)
            packed = mx.recordio.pack(header, img_bytes)
            record.write_idx(i + 1, packed)  # 1-indexed
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Warning: failed to write {image_paths[i]}: {e}")

    record.close()

    rec_size = os.path.getsize(rec_path) / (1024 ** 3)
    print(f"\nDone! Written {num_images - failed} images ({failed} failed)")
    print(f"  {rec_path} ({rec_size:.2f} GB)")
    print(f"  {idx_path}")
    print(f"  Header: num_images={num_images}, num_classes={num_classes}")


def main():
    parser = argparse.ArgumentParser(description="Convert ImageFolder to MXNet RecordIO")
    parser.add_argument("input_dir", help="Path to image folder (with class subdirectories)")
    parser.add_argument("output_dir", help="Path to output directory for train.rec/train.idx")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    args = parser.parse_args()

    image_paths, labels, num_classes = scan_dataset(args.input_dir)
    if len(image_paths) == 0:
        print("No images found!")
        sys.exit(1)

    build_rec(image_paths, labels, num_classes, args.output_dir, args.quality)


if __name__ == "__main__":
    main()
