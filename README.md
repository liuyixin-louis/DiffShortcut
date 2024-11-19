# Investigating and Defending Shortcut Learning in Personalized Diffusion Models
<!-- https://arxiv.org/pdf/2406.18944 -->

> This is an official code implementation of the paper "Investigating and Defending Shortcut Learning in Personalized Diffusion Models" 
> [Paper](https://arxiv.org/pdf/2406.18944); 

## Environment Setup 
- setup the following environment variables 
```shell
# your project root
export ADB_PROJECT_ROOT="/path/to/your/project/root"
# your conda env name
export PYTHONPATH=$PYTHONPATH$:$ADB_PROJECT_ROOT
```

## Software Dependencies
```shell
# python and pytorch, tested under python 3.10 and pytorch 1.13.1
conda create -n diffshortcut python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e . --ignore-installed
pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/TimDettmers/bitsandbytes.git
# install codeformer 
git clone https://github.com/sczhou/CodeFormer.git $ADB_PROJECT_ROOT/diffshortcut/defenses/CodeFormer
cd $ADB_PROJECT_ROOT/diffshortcut/defenses/CodeFormer
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib -y
```

## Simple Purification Code 
```shell
# activate your conda env
export input_dir="./example/"
export output_dir_code="./output/final_results"
export output_dir="./output/final_purified/"
export class_name='person'
python3 diffshortcut/defenses/CodeFormer/inference_codeformer.py -w 0.5 --input_path "$input_dir" -o "$output_dir_code" 
python3 diffshortcut/defenses/sr.py --input_dir $output_dir_code --output_dir $output_dir --class_name $class_name
# you will get the purified images in $output_dir with the same file name as the original one  
```

## Data and Checkpoint Dependencies
```bash
# SD model 
cd ./SD/
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
# dataset
# - We use the open-source dataset, VGGFace2 and CelebA-HQ processed by AntiDreamBooth, which can be found at this [google drive](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV). 
# DPM for Purify
mkdir /weights/
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt -O /weights/256x256_diffusion_uncond.pt
# scoring model checkpoint
mkdir LIQE/checkpoints
wget 'https://huggingface.co/yixin/liqe/resolve/main/LIQE.pt' -O ./LIQE/checkpoints/LIQE.pt
```

## TODO: model training and evulation script 
```bash
# install ip-adapter
# pip install diffusers==0.22.1
# pip install git+https://github.com/tencent-ailab/IP-Adapter.git
# cd IP-Adapter
# git lfs install
# git clone https://huggingface.co/h94/IP-Adapter
# mv IP-Adapter/models models
# mv IP-Adapter/sdxl_models sdxl_models

# check /teamspace/studios/this_studio/diffshortcut/scripts/sub/train/full.sh for training script 

# check /teamspace/studios/this_studio/diffshortcut/scripts/sub/eval/eval.sh for evulating script
```

## Reference 
```bibtex
@article{liu2024investigating,
  title={Investigating and Defending Shortcut Learning in Personalized Diffusion Models},
  author={Liu, Yixin and Chen, Ruoxi and Sun, Lichao},
  journal={arXiv preprint arXiv:2406.18944},
  year={2024}
}
```
