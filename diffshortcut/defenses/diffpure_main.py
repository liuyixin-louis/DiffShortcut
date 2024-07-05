from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch

from diffusers import DDPMScheduler
import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
import os 

opt_t = 2
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def diffpure_denoise(all_images_input, all_images_output, opt_t = 300, class_name='person', output_dir=None, empty_condition=False  ):
    all_ori_img = all_images_input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, None)
    from transformers import AutoTokenizer, PretrainedConfig

    tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None
    )

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    # 5. move the models to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device) 

    # 6. set parameters
    prompt = [f"a photo of a {class_name}, clear, high resolution, high quality, noise-free"]
    if empty_condition:
        prompt = ['']    

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion


    noise_scheduler.set_timesteps(timesteps = list(range(opt_t, -1, -1)))

    guidance_scale = 7.5                # Scale for classifier-free guidance

    generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

    batch_size = len(prompt)


    # 7. get the text_embeddings for the passed prompt
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # denoise_prompt='chaotic, intricate, noisy, abstract, pattern, blurry'

    # 8. get the unconditional text embeddings for classifier-free guidance
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [''] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]


    # 9. concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    from PIL import Image
    # image = Image.open(img_path)

    from torchvision import transforms
    size = 512
    center_crop = True
    trans = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),]
                
            )
    images = [Image.open(i).convert("RGB") for i in all_ori_img]
    images_trans = [trans(i) for i in images]
    images_trans_torch = torch.stack(images_trans, dim=0)
    print(images_trans_torch.shape)
    
    all_images = []
    for i in range(images_trans_torch.shape[0]) :
        image_trans = images_trans_torch[i].unsqueeze(0).to(torch_device)
        image_encoded = 0.18215 * vae.encode(image_trans).latent_dist.sample()

        # noise_scheduler.set_timesteps(opt_t)
        noise = torch.randn_like(image_encoded)
        timesteps_torch = torch.full((batch_size,), opt_t, device=torch_device)
        image_encoded = noise_scheduler.add_noise(image_encoded, noise, timesteps_torch)

        latents = image_encoded.to(torch_device)

        # 13. write the denoising loop
        from tqdm.auto import tqdm

        scaling_factor = vae.config.scaling_factor

        # from opt_t to 1 :
        # print(noise_scheduler.timesteps)
        for t in tqdm(noise_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            latent_model_input = latent_model_input
            # t = torch.full((batch_size,), t, device=torch_device)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample


        # 14. use the vae to decode the generated latents back into the image
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
            
        from PIL import Image
        # 15. convert the image to PIL so we can display or save it
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        all_images.append(image)
        
    pil_images = [Image.fromarray(i[0]) for i in all_images]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pathi in enumerate(all_images_output): 
        pil_images[i].save(pathi)
    
    # return pil_images
    del vae
    del noise_scheduler
    del text_encoder
    del unet
    torch.cuda.empty_cache()
    
import os
from pathlib import Path
from PIL import Image

def process_images(input_dir, output_dir, opt_t = 300, class_name='person',empty_condition=False):
    
    # im
    imgs_input = []
    imgs_output = []
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            # if file already exists, skip
            # if os.path.exists(output_path):
            #     continue
            # else:
            imgs_input.append(img_path)
            imgs_output.append(output_path)
    if len(imgs_input) != 0 :
        diffpure_denoise(imgs_input, imgs_output, opt_t = opt_t, class_name=class_name, output_dir=output_dir, empty_condition=empty_condition)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process images using DiffPure purification.")
    
    parser.add_argument('--input_dir', type=str,  help="")
    parser.add_argument('--output_dir', type=str, help="Directory where processed images will be saved.", default='')
    parser.add_argument("--opt_t", type=int, help="the denosing steps.", default=300)
    parser.add_argument("--class_name", type=str, help="class_name.", default='person')
    parser.add_argument("--empty_condition", action='store_true', help="empty_condition.", default=False)
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output_dir, args.opt_t, args.class_name, args.empty_condition)
