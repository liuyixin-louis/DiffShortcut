import argparse
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from torchvision import transforms as T
from tqdm import tqdm
import os

# Assuming impress and preprocess functions are appropriately defined in the mentioned modules
from impress.utils import preprocess, recover_image, plot
from impress.impress import impress

def denoise_images(image_paths, pipe_img2img, output_path, device='cuda'):
    to_pil = T.ToPILImage()
    # processed_paths = []
    paths = []
    imgs = []
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).to(device).half()  # Ensure preprocess is appropriately defined
        x_purified = impress(x, model=pipe_img2img.vae, clamp_min=-1, clamp_max=1, eps=0.1, iters=3000, lr=0.01, pur_alpha=0.1, noise=0.1)
        x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
        purified_image = to_pil(x_purified[0]).convert("RGB")
        
        # Define where to save the processed image
        to_save = output_path / img_path.name
        os.makedirs(os.path.dirname(to_save), exist_ok=True)
        # purified_image.save(to_save)
        paths.append(to_save)
        imgs.append(purified_image)
        # processed_paths.append(to_save)
    output_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(paths)):
        imgs[i].save(paths[i])
    # return processed_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Process images using IMPRESS purification.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed images.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    

    # Load the model
    model_name = 'stabilityai/stable-diffusion-2-1-base'
    sd_pipe_img2img = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    sd_pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe_img2img.scheduler.config)
    sd_pipe_img2img.to("cuda")

    # Process images
    image_paths = list(input_path.glob('*.[jp][pn]g'))  # Supports both .jpg and .png files
    denoise_images(image_paths, sd_pipe_img2img, output_path, device='cuda')

    # Clean up
    del sd_pipe_img2img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
