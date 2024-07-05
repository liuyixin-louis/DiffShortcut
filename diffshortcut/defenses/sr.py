#!/usr/bin/env python
# coding=utf-8
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionUpscalePipeline

def super_resolution_purification(input_dir, output_dir, sr_scale=4, class_name= 'person'):
    # Load the super resolution model
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    else:
        sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id).to(device)


    file_names = []
    gen_files = []
    # Process each image in the input directory
    for img_file in Path(input_dir).glob('*'):
        if img_file.suffix in ['.jpg', '.png']:
            # Load image
            image = Image.open(img_file).convert("RGB")
            prompt = f"A photo of a {class_name}"  # Adjust prompt as needed
            # resize image to 128x128
            image = image.resize((128, 128))
            # Apply super resolution
            with torch.no_grad():
                sr_image = sr_pipeline(image=image, prompt=prompt).images[0]

            # Save the processed image
            # output_path = Path(output_dir) / img_file.name
            # sr_image.save(output_path)
            file_names.append(img_file.name)
            gen_files.append(sr_image)
    
    # del sr_pipeline
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, file_name in enumerate(file_names):
        output_path = Path(output_dir) / file_name
        gen_files[i].save(output_path)
    
    
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply super resolution purification to images.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing images for processing.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where processed images will be saved.")
    parser.add_argument('--class_name', type=str, default='person', help="Class name for the prompt.")
    args = parser.parse_args()

    super_resolution_purification(args.input_dir, args.output_dir, args.class_name)
