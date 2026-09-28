

from multiprocessing import Pool, cpu_count
import logging
import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import skimage.color
from tiatoolbox import data, logger
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore import wsireader
from PIL import Image
import json

mpl.rcParams["figure.dpi"] = 150  # for high resolution figure

# Clear logger to use tiatoolbox.logger
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

# ✅ Function to check if an item has been processed
def is_processed(folder_name, image_name):
    patches_dir = f"/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/PreProcessing/{folder_name}/tumor/images/{os.path.splitext(image_name)[0]}/Patches"
    return os.path.exists(patches_dir)

# ✅ Stain normalization
def stain_norm(source_image, target_image, image_name, folder_name):
    save_dir = f'/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/PreProcessing/{folder_name}/tumor/images/{os.path.splitext(image_name)[0]}'
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, image_name)
    if os.path.exists(out_path):
        return out_path  # already normalized

    src_image = cv2.imread(source_image)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    trg_image = cv2.imread(target_image)
    method_name = "Macenko"
    stain_normalizer = stainnorm.get_normalizer(method_name)
    stain_normalizer.fit(trg_image)
    normed_sample = stain_normalizer.transform(src_image)
    out = Image.fromarray(normed_sample)
    out.save(out_path)

    return out_path

# ✅ Patch extraction
def extract_patches(image_path, patch_size, image_name):
    img = Image.open(image_path)
    width, height = img.size

    save_dir = os.path.dirname(image_path)
    patches_dir = os.path.join(save_dir, "Patches")
    os.makedirs(patches_dir, exist_ok=True)

    patches_info = []
    patch_count = 0
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch_name = f"patch_{patch_count}.png"
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patch.save(os.path.join(patches_dir, patch_name))
            patches_info.append({"patch_name": patch_name, "x": x, "y": y})
            patch_count += 1

    json_save_path = os.path.join(save_dir, f"patches_info_{os.path.splitext(image_name)[0]}.json")
    with open(json_save_path, "w") as f:
        json.dump(patches_info, f, indent=4)

    return f"Extracted {patch_count} patches for {image_name}"

# ✅ Process one folder
def process_item(item):
    path_all = "/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/TMC_new"
    target_image = '/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/datasets/midog/images/train/001_patch_200_4000.png'

    folder_name = item
    file_path = os.path.join(path_all, folder_name)
    patch_size = 256

    tiles_folder = os.path.join(file_path, "tumor", "images")
    if not os.path.exists(tiles_folder):
        return [f"No tiles folder found for {folder_name}"]

    images = [f for f in os.listdir(tiles_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    results = []
    for img in images:
        source_image = os.path.join(tiles_folder, img)

        try:
            if is_processed(folder_name, img):
                results.append(f"Skipping {img} in {folder_name} (already processed).")
                continue

            image_path = stain_norm(source_image, target_image, img, folder_name)
            results.append(extract_patches(image_path, patch_size, img))
        except Exception as e:
            results.append(f"Error processing {img}: {e}")

    return results

# ✅ Parallel run
def run_parallel_processing():
    path_all = "/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/TMC_new"
    All_folders = [f for f in os.listdir(path_all) if os.path.isdir(os.path.join(path_all, f))]

    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        results = pool.map(process_item, All_folders)

    for folder_results in results:
        for result in folder_results:
            print(result)

if __name__ == "__main__":
    run_parallel_processing()
