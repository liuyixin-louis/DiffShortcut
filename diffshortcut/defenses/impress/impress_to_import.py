import os
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torchvision.transforms as T

from .utils import preprocess, recover_image, plot
from .impress import impress

def denoise_images(image_path, pipe_img2img, output_stamp='impress', device='cuda'):
    # Setup device and load the denoising model
    # device = torch.device(device if torch.cuda.is_available() else "cpu")
    # pipe_img2img = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    # pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    # pipe_img2img.to(device)

    # Gather all image files to be denoised
    image_path = [str(i) for i in image_path]
    image_path_to_return = [ i.replace('final', f"final-{output_stamp}") for i in image_path ] 
    all_ori_img = [i for i in image_path if not os.path.exists(i.replace("final", f"final-{output_stamp}"))]
    
    if not (len(all_ori_img) == 0):
        # Denoising process
        to_pil = T.ToPILImage()
        for img_path in tqdm(all_ori_img):
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device).half()  # You need to define the preprocess function or import it
            x_purified = impress(x, model=pipe_img2img.vae, clamp_min=-1, clamp_max=1, eps=0.1, iters=3000, lr=0.01, pur_alpha=0.1, noise=0.1)
            x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
            purified_image = to_pil(x_purified[0]).convert("RGB")
            to_save = img_path.replace("final", f"final-{output_stamp}")
            os.makedirs(os.path.dirname(to_save), exist_ok=True)
            purified_image.save(to_save)
    return image_path_to_return
    # del pipe_img2img
    # torch.cuda.empty_cache()

def _pre_denoise_images(image_path, pipe_img2img, output_stamp='impress', device='cuda'):
    # Setup device and load the denoising model
    # device = torch.device(device if torch.cuda.is_available() else "cpu")
    # pipe_img2img = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    # pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    # pipe_img2img.to(device)

    # Gather all image files to be denoised
    all_ori_img = []
    instances = os.listdir(image_path)
    for instance in instances:
        this_path = os.path.join(image_path, instance, "noise-ckpt/final")
        all_ori_img += [os.path.join(this_path, i) for i in os.listdir(this_path) if i.endswith(".png")]
    all_ori_img = [i for i in all_ori_img if not os.path.exists(i.replace("final", f"final-{output_stamp}"))]
    
    if not (len(all_ori_img) == 0):
        # Denoising process
        to_pil = T.ToPILImage()
        for img_path in tqdm(all_ori_img):
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device).half()  # You need to define the preprocess function or import it
            x_purified = impress(x, model=pipe_img2img.vae, clamp_min=-1, clamp_max=1, eps=0.1, iters=3000, lr=0.01, pur_alpha=0.1, noise=0.1)
            x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
            purified_image = to_pil(x_purified[0]).convert("RGB")
            to_save = img_path.replace("final", f"final-{output_stamp}")
            os.makedirs(os.path.dirname(to_save), exist_ok=True)
            purified_image.save(to_save)
    # del pipe_img2img
    # torch.cuda.empty_cache()

# Example usage
# image_directory = '/path/to/your/images'
# denoise_images(image_directory, model_name='stabilityai/stable-diffusion-2-1-base', output_stamp='impress', device='cuda')
