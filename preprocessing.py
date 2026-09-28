"""
Stage 1: Stain normalization and patch extraction.

Each image is Macenko stain-normalized to a reference image, then tiled into
non-overlapping square patches. Patch coordinates are saved to a JSON file so
detections can later be mapped back to whole-image coordinates.

Expected input layout:
    <input_root>/<case>/tumor/images/*.png|jpg|jpeg
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
from PIL import Image
from tiatoolbox.tools import stainnorm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def image_output_dir(output_root, case_name, image_name):
    """Folder where the normalized image, its patches and JSON are stored."""
    stem = os.path.splitext(image_name)[0]
    return os.path.join(output_root, case_name, "tumor", "images", stem)


def is_processed(output_root, case_name, image_name):
    """An image counts as processed once its Patches folder exists."""
    return os.path.exists(
        os.path.join(image_output_dir(output_root, case_name, image_name), "Patches")
    )


def stain_norm(source_image, target_image, save_dir, image_name):
    """Macenko-normalize source_image to target_image and save the result."""
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, image_name)
    if os.path.exists(out_path):
        return out_path  # already normalized

    src_image = cv2.cvtColor(cv2.imread(source_image), cv2.COLOR_BGR2RGB)
    trg_image = cv2.cvtColor(cv2.imread(target_image), cv2.COLOR_BGR2RGB)

    stain_normalizer = stainnorm.get_normalizer("Macenko")
    stain_normalizer.fit(trg_image)
    normed_sample = stain_normalizer.transform(src_image)
    Image.fromarray(normed_sample).save(out_path)

    return out_path


def extract_patches(image_path, patch_size, image_name):
    """Tile the image into patch_size x patch_size patches (edges are padded)."""
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

    stem = os.path.splitext(image_name)[0]
    json_save_path = os.path.join(save_dir, f"patches_info_{stem}.json")
    with open(json_save_path, "w") as f:
        json.dump(patches_info, f, indent=4)

    return f"Extracted {patch_count} patches for {image_name}"


def process_case(case_name, input_root, output_root, target_image, patch_size):
    """Normalize and tile every image of one case folder."""
    tiles_folder = os.path.join(input_root, case_name, "tumor", "images")
    if not os.path.exists(tiles_folder):
        return [f"No tiles folder found for {case_name}"]

    images = [f for f in os.listdir(tiles_folder) if f.lower().endswith(IMAGE_EXTENSIONS)]

    results = []
    for img in images:
        try:
            if is_processed(output_root, case_name, img):
                results.append(f"Skipping {img} in {case_name} (already processed).")
                continue

            source_image = os.path.join(tiles_folder, img)
            save_dir = image_output_dir(output_root, case_name, img)
            image_path = stain_norm(source_image, target_image, save_dir, img)
            results.append(extract_patches(image_path, patch_size, img))
        except Exception as e:
            results.append(f"Error processing {img} in {case_name}: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Stain normalization and patch extraction.")
    parser.add_argument("--input-root", default="data/raw",
                        help="Folder containing one subfolder per case (default: data/raw)")
    parser.add_argument("--output-root", default="data/preprocessed",
                        help="Where normalized images and patches are written (default: data/preprocessed)")
    parser.add_argument("--target-image", default="assets/001_patch_200_4000.png",
                        help="Reference image for stain normalization")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels (default: 256)")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel processes (default: all CPU cores)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.target_image):
        raise FileNotFoundError(f"Target image not found: {args.target_image}")
    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input folder not found: {args.input_root}")

    cases = [f for f in os.listdir(args.input_root)
             if os.path.isdir(os.path.join(args.input_root, f))]

    worker = partial(
        process_case,
        input_root=args.input_root,
        output_root=args.output_root,
        target_image=args.target_image,
        patch_size=args.patch_size,
    )

    with Pool(args.workers) as pool:
        results = pool.map(worker, cases)

    for case_results in results:
        for result in case_results:
            print(result)


if __name__ == "__main__":
    main()
