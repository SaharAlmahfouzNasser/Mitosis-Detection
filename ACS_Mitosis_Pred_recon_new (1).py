import os
import json
import csv
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Save bounding boxes to CSV
def save_bbox_coordinates_to_csv(bbox_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y'])
        for bbox in bbox_list:
            writer.writerow(bbox)

# Reconstruct image from patches
def reconstruct_image(patch_dir, patches_info_path, output_path):
    with open(patches_info_path) as f:
        patches_info = json.load(f)

    patch_size = Image.open(os.path.join(patch_dir, patches_info[0]["patch_name"])).size[0]
    width = max(p["x"] + patch_size for p in patches_info)
    height = max(p["y"] + patch_size for p in patches_info)

    reconstructed_img = Image.new('RGB', (width, height))

    for patch_info in patches_info:
        x, y = patch_info["x"], patch_info["y"]
        patch_name = patch_info["patch_name"]
        patch_img = Image.open(os.path.join(patch_dir, patch_name))
        reconstructed_img.paste(patch_img, (x, y))

    reconstructed_img.save(output_path)
    print(f"Reconstructed image saved to {output_path}")

# Load YOLO model
model = YOLO('/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/runs/detect/train28/weights/best.pt')

# Root path to PreProcessing output
root_dir = "/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/PreProcessing/"
output_base = "/workspace/SuperClassifier/my_MIDOG_dataset/Code/yolov5/Test_all/TMC_new/TMC_new_Final_Outputs"

# Iterate through folders
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name, "tumor", "images")
    if not os.path.exists(folder_path):
        continue

    for img_subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, img_subfolder)
        patch_dir = os.path.join(subfolder_path, "Patches")
        if not os.path.exists(patch_dir):
            continue

        json_path = os.path.join(subfolder_path, f"patches_info_{img_subfolder}.json")
        if not os.path.exists(json_path):
            print(f"Missing JSON: {json_path}")
            continue

        image_list = sorted(os.listdir(patch_dir))
        all_boxes = []

        for patch_name in image_list:
            patch_path = os.path.join(patch_dir, patch_name)
            results = model(patch_path)

            with open(json_path) as f:
                patches_info = json.load(f)

            # Find x, y coordinates of this patch
            patch_info = next((p for p in patches_info if p["patch_name"] == patch_name), None)
            if patch_info is None:
                continue

            x_offset, y_offset = patch_info["x"], patch_info["y"]
            image = Image.open(patch_path)
            draw = ImageDraw.Draw(image)

            for result in results:
                for i, box in enumerate(result.boxes.xyxy):
                    if int(result.boxes.cls[i]) == 1:  # class index = 1
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=(0, 255, 0), width=3)

                        top_left_x = box[0].detach().cpu().item() + x_offset
                        top_left_y = box[1].detach().cpu().item() + y_offset
                        bottom_right_x = box[2].detach().cpu().item() + x_offset
                        bottom_right_y = box[3].detach().cpu().item() + y_offset

                        all_boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
                        print("Added bounding box coordinates:", all_boxes[-1])

            # Save annotated patch
            detected_patch_dir = os.path.join(subfolder_path, "Patches_detected")
            os.makedirs(detected_patch_dir, exist_ok=True)
            image.save(os.path.join(detected_patch_dir, patch_name))

        # Save bounding boxes CSV
        coord_csv_path = os.path.join(output_base, "Final_Output", folder_name)
        os.makedirs(coord_csv_path, exist_ok=True)
        save_bbox_coordinates_to_csv(all_boxes, os.path.join(coord_csv_path, f"{img_subfolder}.csv"))

        # Save reconstructed image
        recon_img_dir = os.path.join(output_base, "Outputs", folder_name)
        os.makedirs(recon_img_dir, exist_ok=True)
        output_path = os.path.join(recon_img_dir, f"img_{img_subfolder}_norm_reconstructed.png")
        reconstruct_image(detected_patch_dir, json_path, output_path)
