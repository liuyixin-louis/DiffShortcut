import os
import argparse
from PIL import Image
import numpy as np
import cvxpy as cp

def total_variation(Z, shape, p=2):
    h, w, c = shape
    Z_reshaped = cp.reshape(Z, (h, w*c))
    horizontal_diff = Z_reshaped[1:, :] - Z_reshaped[:-1, :]
    horizontal_norm = cp.norm(horizontal_diff, p, axis=1)
    horizontal_sum = cp.sum(horizontal_norm)
    vertical_diff = Z_reshaped[:, 1:] - Z_reshaped[:, :-1]
    vertical_norm = cp.norm(vertical_diff, p, axis=0)
    vertical_sum = cp.sum(vertical_norm)
    return horizontal_sum + vertical_sum

def minimize_total_variation(X, x, lambda_tv=0.01, p=2):
    h, w, c = x.shape
    Z = cp.Variable((h, w*c))
    X_flat = np.reshape(X, (h, w*c))
    x_flat = np.reshape(x, (h, w*c))
    objective = cp.Minimize(cp.norm(cp.multiply((1 - X_flat),(Z - x_flat)), 2) + lambda_tv * total_variation(Z, (h, w, c), p))
    problem = cp.Problem(objective)
    problem.solve(verbose=True, solver=cp.MOSEK)
    return Z.value

def get_tvm_image(img_path, output_path, TVM_WEIGHT=0.01, PIXEL_DROP_RATE=0.02):
    img = Image.open(img_path)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    X = np.random.binomial(1, PIXEL_DROP_RATE, img_array.shape)
    Z_optimal = minimize_total_variation(X, img_array, lambda_tv=TVM_WEIGHT)
    Z_optimal = np.reshape(Z_optimal, img_array.shape)
    img_result = Image.fromarray(np.uint8(Z_optimal*255))
    img_result.save(output_path)

def process_images(input_dir, output_dir):
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            get_tvm_image(img_path, output_path)
            print(f"Processed {img_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TVM purification to images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed images.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_images(args.input_dir, args.output_dir)
