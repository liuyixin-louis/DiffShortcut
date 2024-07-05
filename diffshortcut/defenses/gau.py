#!/usr/bin/env python
# coding=utf-8
import os
from pathlib import Path
from PIL import Image, ImageFilter
import argparse

def gaussian_smooth_image(image_path, output_path, kernel_size=5):
    """Apply Gaussian smoothing to an image and save it."""
    img = Image.open(image_path)
    gaussian_image = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
    gaussian_image.save(output_path)

def smooth_images_in_directory(input_dir, output_dir, kernel_size=7):
    """Apply Gaussian smoothing to all images in the directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in input_path.glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            output_file = output_path / image_file.name
            gaussian_smooth_image(image_file, output_file, kernel_size)

def main():
    parser = argparse.ArgumentParser(description="Apply Gaussian smoothing to all images in a directory.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing images for processing.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where processed images will be saved.")
    parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size for Gaussian smoothing.")
    args = parser.parse_args()

    smooth_images_in_directory(args.input_dir, args.output_dir, args.kernel_size)

if __name__ == "__main__":
    main()
