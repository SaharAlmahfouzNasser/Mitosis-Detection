"""
Stage 2: Mitosis detection, coordinate export and image reconstruction.

Runs a trained YOLOv8 model on every patch produced by preprocessing.py,
keeps detections of the mitotic-figure class, maps them to whole-image
coordinates, saves them as CSV and rebuilds the full annotated image.

Expected input layout (output of preprocessing.py):
    <input_root>/<case>/tumor/images/<image>/Patches/*.png
    <input_root>/<case>/tumor/images/<image>/patches_info_<image>.json
"""

import argparse
import csv
import json
import os

from PIL import Image, ImageDraw
from ultralytics import YOLO


def save_bbox_coordinates_to_csv(bbox_list, filename):
    """Write bounding boxes (whole-image coordinates) to a CSV file."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"])
        writer.writerows(bbox_list)


def reconstruct_image(patch_dir, patches_info, output_path):
    """Stitch patches back into a full image using their stored offsets."""
    patch_size = Image.open(os.path.join(patch_dir, patches_info[0]["patch_name"])).size[0]
    width = max(p["x"] + patch_size for p in patches_info)
    height = max(p["y"] + patch_size for p in patches_info)

    reconstructed_img = Image.new("RGB", (width, height))
    for patch_info in patches_info:
        patch_img = Image.open(os.path.join(patch_dir, patch_info["patch_name"]))
        reconstructed_img.paste(patch_img, (patch_info["x"], patch_info["y"]))

    reconstructed_img.save(output_path)
    print(f"Reconstructed image saved to {output_path}")


def process_image(model, image_dir, image_stem, case_name, output_root, class_id, conf):
    """Detect mitoses in all patches of one image and save the outputs."""
    patch_dir = os.path.join(image_dir, "Patches")
    json_path = os.path.join(image_dir, f"patches_info_{image_stem}.json")
    if not os.path.exists(json_path):
        print(f"Missing patch info for {case_name}/{image_stem}, skipping.")
        return

    with open(json_path) as f:
        patches_info = json.load(f)
    offsets = {p["patch_name"]: (p["x"], p["y"]) for p in patches_info}

    detected_patch_dir = os.path.join(image_dir, "Patches_detected")
    os.makedirs(detected_patch_dir, exist_ok=True)

    all_boxes = []
    for patch_name in sorted(os.listdir(patch_dir)):
        if patch_name not in offsets:
            continue

        patch_path = os.path.join(patch_dir, patch_name)
        x_offset, y_offset = offsets[patch_name]
        results = model(patch_path, conf=conf, verbose=False)

        image = Image.open(patch_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) != class_id:
                    continue
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                all_boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset])

        image.save(os.path.join(detected_patch_dir, patch_name))

    print(f"{case_name}/{image_stem}: {len(all_boxes)} mitotic figure(s) detected")

    # Bounding boxes CSV
    bbox_dir = os.path.join(output_root, "bboxes", case_name)
    os.makedirs(bbox_dir, exist_ok=True)
    save_bbox_coordinates_to_csv(all_boxes, os.path.join(bbox_dir, f"{image_stem}.csv"))

    # Reconstructed annotated image
    recon_dir = os.path.join(output_root, "reconstructed", case_name)
    os.makedirs(recon_dir, exist_ok=True)
    reconstruct_image(detected_patch_dir, patches_info,
                      os.path.join(recon_dir, f"{image_stem}_reconstructed.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 mitosis detection on image patches.")
    parser.add_argument("--weights", default="weights/best.pt",
                        help="Path to trained YOLOv8 weights (default: weights/best.pt)")
    parser.add_argument("--input-root", default="data/preprocessed",
                        help="Output folder of preprocessing.py (default: data/preprocessed)")
    parser.add_argument("--output-root", default="outputs",
                        help="Where CSVs and reconstructed images are written (default: outputs)")
    parser.add_argument("--class-id", type=int, default=1,
                        help="Class index of mitotic figures (default: 1)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input folder not found: {args.input_root}")

    model = YOLO(args.weights)

    for case_name in sorted(os.listdir(args.input_root)):
        images_root = os.path.join(args.input_root, case_name, "tumor", "images")
        if not os.path.isdir(images_root):
            continue

        for image_stem in sorted(os.listdir(images_root)):
            image_dir = os.path.join(images_root, image_stem)
            if not os.path.isdir(os.path.join(image_dir, "Patches")):
                continue
            process_image(model, image_dir, image_stem, case_name,
                          args.output_root, args.class_id, args.conf)


if __name__ == "__main__":
    main()
