#!/usr/bin/env python
# coding=utf-8
import os
from pathlib import Path
from PIL import Image
import argparse

def jpeg_compress_image(image_path, output_path, quality=75):
    """Apply JPEG compression to an image and save it."""
    img = Image.open(image_path)
    img.save(output_path, 'JPEG', quality=quality)

def compress_images_in_directory(input_dir, output_dir, quality=75):
    """Compress all images in the directory using JPEG compression."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in input_path.glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            output_file = output_path / image_file.name
            jpeg_compress_image(image_file, output_file, quality)

def main():
    parser = argparse.ArgumentParser(description="Apply JPEG compression to all images in a directory.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing images for processing.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where processed images will be saved.")
    parser.add_argument("--quality", type=int, default=75, help="JPEG compression quality.")
    args = parser.parse_args()

    compress_images_in_directory(args.input_dir, args.output_dir, args.quality)

if __name__ == "__main__":
    main()
