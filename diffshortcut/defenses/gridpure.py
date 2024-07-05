import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict, add_dict_to_argparser

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res
from diffshortcut.generic.tools import get_project_root
class DiffPure:
    def __init__(self, model_path=get_project_root()+"/weights/256x256_diffusion_uncond.pt"):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image transformations
        self.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Load model and setup
        self.model, self.diffusion = self.setup_model(model_path)
        self.model.eval()

        # Precomputed alpha values for denoising steps
        # self.alpha_bar = self.precompute_alphas()

    def setup_model(self, model_path):
        parser = self.create_argparser(model_path)
        args = parser.parse_args([])
        dist_util.setup_dist()
        logger.configure()
        
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
        model.to(self.device)
        if args.use_fp16:
            model.convert_to_fp16()
        return model, diffusion

    def create_argparser(self, model_path):
        defaults = dict(
            clip_denoised=True,
            num_samples=1,
            batch_size=1,
            use_ddim=False,
            model_path=model_path,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser


    def denoise(self, images, opt_t):
        img_iter = self.diffusion.q_sample(images, torch.tensor([opt_t]*images.shape[0], device=self.device) )
        indices = list(range(opt_t+1))[::-1]
        # i = indices[0]
        for i in indices:
            if i ==0:
                break 
            t = torch.tensor([i]*images.shape[0], device=self.device)
            with torch.no_grad():
                out = self.diffusion.p_sample(self.model, img_iter, t, clip_denoised=True, denoised_fn=None,
                cond_fn=None,
                model_kwargs={},)
                img_iter = out['sample']
                # return out['pred_xstart'].squeeze(0).cpu()
        return img_iter.squeeze(0).cpu()

    def process_batch(self, batch_images, t):
        # Convert PIL Images to Tensor and preprocess
        images = torch.stack([self.transform(img) for img in batch_images]).to(self.device)
        return self.denoise(images, t)

from tqdm import tqdm
def grid_based_diffpure(image, diff_purifier, window_size=256, stride=128, t=10, gamma=0.1, iterations=10):
    # Convert the input image to a tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)
    # image_tensor = diff_purifier.transform(image).unsqueeze(0) 
    
    # Get image dimensions
    _, _, H, W = image_tensor.shape
    output_tensor = image_tensor
    
    
    for _ in tqdm(range(iterations)):
        # Initialize output tensor with zeros
        output_tensor_this_round = torch.zeros_like(image_tensor)
        weight_tensor = torch.zeros_like(image_tensor)

        # cnt=0
        # Slide over the image to extract patches
        for i in range(0, H - window_size + 1, stride):
            for j in range(0, W - window_size + 1, stride):
                patch = output_tensor[:, :, i:i+window_size, j:j+window_size]
                patch_image = transforms.ToPILImage()(patch.squeeze(0))
                denoised_patch = diff_purifier.process_batch([patch_image], t=t)
                denoised_patch_tensor = denoised_patch.unsqueeze(0)

                # Add the denoised patch to the output tensor
                output_tensor_this_round[:, :, i:i+window_size, j:j+window_size] += denoised_patch_tensor
                weight_tensor[:, :, i:i+window_size, j:j+window_size] += 1
        #         cnt+=1
        # print(cnt)
        # Average the overlapping regions
        output_tensor_this_round /= weight_tensor

        # Apply moving average with the original image
        output_tensor = (1 - gamma) * output_tensor_this_round + gamma * output_tensor

    # Convert the output tensor to a PIL image
    transform_to_pil = transforms.ToPILImage()
    purified_image = transform_to_pil(output_tensor.squeeze(0))

    return purified_image

import os 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-based Image Purifier")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, help="Directory to save purified images")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    diff_purifier = DiffPure(model_path=get_project_root()+"/weights/256x256_diffusion_uncond.pt")
    outs = []
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_dir, image_name)
            image = Image.open(image_path).resize((512, 512))
            purified_image = grid_based_diffpure(image, diff_purifier, t=10, gamma=0.1, iterations=10)
            outs.append((purified_image, image_name))
            
    os.makedirs(output_dir, exist_ok=True)
    for purified_image, image_name in outs:
        purified_image.save(os.path.join(output_dir, image_name))