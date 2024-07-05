# %%
# !pip install matplotlib -q 
# !pip install jsonlines -q
# !pip install tqdm -q
import torch
import torch.nn as nn
import torchvision.models as models

import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
import sys
import argparse
import jsonlines
from glaze import glaze
import re
import copy
from utils import preprocess, recover_image, plot
from impress import impress
import shutil


# %%

parser = argparse.ArgumentParser(description='diffusion attack')


# model_id = "stabilityai/stable-diffusion-2-1"
parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                    help='stable diffusion weight')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--train', default= 0, type=bool,
                    help='training(True) or testing(False)')

# data
# parser.add_argument('--clean_data_dir', type=str, default='../wikiart/preprocessed_data/claude-monet/clean/train/')
# parser.add_argument('--trans_data_dir', type=str, default='../wikiart/preprocessed_data/claude-monet/trans/train/trans_Cubism_by_Picasso_seed9222')

parser.add_argument('--neg_feed', type=float, default=-1.)
parser.add_argument('--adv_para', type=str, default=None)
parser.add_argument('--pur_para', type=str, default=None)

# ae Hyperparameters
parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
parser.add_argument('--pur_iters', default=3000, type=int, help='ae Hyperparameters')
parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
parser.add_argument('--pur_noise', default=0.1, type=float, help='ae Hyperparameters')



# Miscs
parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

# Device options
parser.add_argument('--device', default='cuda:9', type=str,
                    help='device used for training')

# image_path
parser.add_argument('--image_path', default='', type=str,
                    help='path to the image to purify')

# output_path
parser.add_argument('--output_stamp', default='impress', type=str,
                    help='output path stamp')

args = parser.parse_args()

# %%

np.random.seed(seed = args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
torch.backends.cudnn.deterministic=True
device = torch.device(args.device if torch.cuda.is_available() else "cpu")



# %%

image_path = args.image_path
import os 
all_ori_img = []
instances = os.listdir(image_path)
for ins in instances:
    # noise-ckpt/final
    this_path = os.path.join(image_path, ins, "noise-ckpt/final")
    all_ori_img += [os.path.join(this_path, i) for i in os.listdir(this_path) if i.endswith(".png")]
# print(len(all_ori_img))

# %%
# !pip install transformers

# %%
device=args.device
to_pil = T.ToPILImage()
pipe_img2img = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
pipe_img2img = pipe_img2img.to(device)
for name, param in pipe_img2img.vae.named_parameters():
    param.requires_grad = False

# %%
import os 

for img_to_purify in tqdm(all_ori_img):
    img = Image.open(img_to_purify).convert("RGB")
    x = preprocess(img).to(device).half()
    x_purified = impress(x,
                         model=pipe_img2img.vae,
                         clamp_min=-1,
                         clamp_max=1,
                         eps=0.1,
                         iters=3000,
                         lr=0.01,
                         pur_alpha=0.1,
                         noise=0.1, )
    x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
    purified_image = to_pil(x_purified[0]).convert("RGB")
    to_save = img_to_purify.replace("final", f"final-{args.output_stamp}")
    # to_save = os.path.join(args.output_path, img_to_purify.split("/")[-1])
    os.makedirs(os.path.dirname(to_save), exist_ok=True)
    purified_image.save(to_save)

