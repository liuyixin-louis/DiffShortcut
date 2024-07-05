from gridpure import DiffPure, Image, argparse, grid_based_diffpure
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-based Image Purifier")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, help="Directory to save purified images")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    diff_purifier = DiffPure()
    outs = []
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_dir, image_name)
            image = Image.open(image_path).resize((512, 512))
            purified_image = grid_based_diffpure(image, diff_purifier, t=100, gamma=0, iterations=1)
            outs.append((purified_image, image_name))
            
    os.makedirs(output_dir, exist_ok=True)
    for purified_image, image_name in outs:
        purified_image.save(os.path.join(output_dir, image_name))