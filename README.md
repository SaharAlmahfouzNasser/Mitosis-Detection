# Mitotic Figure Detection in H&E Tumor Images

A two-stage pipeline for detecting **mitotic figures** in H&E-stained histopathology images. Images are stain-normalized to a MIDOG reference, tiled into patches, run through a YOLOv8 detector trained on the [MIDOG](https://midog.deepmicroscopy.org/) dataset, and the detections are mapped back to whole-image coordinates.

<p align="center">
  <img src="assets/001_patch_200_4000.png" width="256" alt="MIDOG reference patch used as the stain-normalization target"/>
  <br/>
  <em>Stain-normalization target (MIDOG training patch <code>001_patch_200_4000.png</code>)</em>
</p>

---

## Pipeline Overview

```mermaid
flowchart LR
    A[Raw tumor image] --> B[Macenko stain normalization<br/>tiatoolbox]
    B --> C[Tile into 256×256 patches<br/>+ patch coordinates JSON]
    C --> D[YOLO inference per patch<br/>Ultralytics]
    D --> E[Keep class 1 = mitotic figure]
    E --> F[Shift boxes to global<br/>image coordinates]
    F --> G[CSV of bounding boxes]
    E --> H[Annotated patches]
    H --> I[Reconstructed annotated<br/>full image]
```

### Stage 1 — Preprocessing (`preprocessing.py`)

1. **Stain normalization** — each source image is normalized with the **Macenko** method (`tiatoolbox.tools.stainnorm`) using a MIDOG training patch as the target. This reduces color variation between scanners and staining protocols so the images resemble the detector's training distribution.
2. **Patch extraction** — the normalized image is tiled into non-overlapping **256 × 256** patches. Edge patches are padded to full size. The top-left `(x, y)` of each patch is stored in a JSON file for later reconstruction.
3. **Parallel & resumable** — case folders are processed in parallel using all CPU cores (`multiprocessing.Pool`), and images that already have a `Patches/` directory are skipped.

### Stage 2 — Detection & Reconstruction (`mitosis_detection.py`)

1. Runs a trained **YOLOv8** model (`ultralytics`) on every patch.
2. Keeps detections of **class index 1** (mitotic figure) and draws them in green on the patch.
3. Converts patch-level boxes to **whole-image coordinates** using the stored patch offsets.
4. Writes one **CSV of bounding boxes** per image.
5. **Reconstructs** the full normalized image from the annotated patches so all detections can be reviewed at once.

---

## Repository Structure

```
.
├── preprocessing.py                  # Stain normalization + patch extraction
├── mitosis_detection.py              # YOLOv8 inference, CSV export, image reconstruction
├── assets/
│   └── 001_patch_200_4000.png        # Stain-normalization target (MIDOG)
├── data/                             # Your images (not tracked by git)
├── weights/                          # Your trained model (not tracked by git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`**

```
tiatoolbox
ultralytics
opencv-python
pillow
```

> `tiatoolbox` may require system libraries such as OpenSlide. See the [tiatoolbox installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html).

### Model weights

The trained detector weights (`best.pt`) are **not included** in this repository. Place them at `weights/best.pt`, or pass another location with `--weights`.

---

## Input Data Layout

The preprocessing script expects one folder per case, each containing tumor image tiles:

```
data/raw/
├── <case_1>/
│   └── tumor/
│       └── images/
│           ├── image_a.png
│           └── image_b.jpg
├── <case_2>/
│   └── tumor/
│       └── images/
│           └── ...
```

Supported formats: `.png`, `.jpg`, `.jpeg`.

---

## Usage

Both scripts run with sensible defaults, so with the layout above you only need:

```bash
# 1. Normalize and tile all images
python preprocessing.py

# 2. Detect mitoses, export coordinates, and reconstruct annotated images
python mitosis_detection.py
```

All paths and settings can be overridden from the command line:

| Script | Argument | Default | Purpose |
|---|---|---|---|
| `preprocessing.py` | `--input-root` | `data/raw` | Folder containing one subfolder per case |
| `preprocessing.py` | `--output-root` | `data/preprocessed` | Normalized images, patches and patch JSONs |
| `preprocessing.py` | `--target-image` | `assets/001_patch_200_4000.png` | Stain-normalization reference |
| `preprocessing.py` | `--patch-size` | `256` | Patch size in pixels |
| `preprocessing.py` | `--workers` | all CPU cores | Number of parallel processes |
| `mitosis_detection.py` | `--weights` | `weights/best.pt` | Trained YOLOv8 weights |
| `mitosis_detection.py` | `--input-root` | `data/preprocessed` | Output folder of `preprocessing.py` |
| `mitosis_detection.py` | `--output-root` | `outputs` | CSVs and reconstructed images |
| `mitosis_detection.py` | `--class-id` | `1` | Class index of mitotic figures |
| `mitosis_detection.py` | `--conf` | `0.25` | Detection confidence threshold |

Example with custom paths:

```bash
python preprocessing.py --input-root /path/to/dataset --output-root /path/to/preprocessed
python mitosis_detection.py --input-root /path/to/preprocessed --weights /path/to/best.pt --conf 0.4
```

Run `python <script>.py --help` for the full list.

---

## Outputs

**After preprocessing**

```
data/preprocessed/
└── <case>/tumor/images/<image_name>/
    ├── <image_name>.png                  # Stain-normalized image
    ├── patches_info_<image_name>.json    # Patch names and (x, y) offsets
    ├── Patches/                          # 256×256 patches
    └── Patches_detected/                 # Annotated patches (created in stage 2)
```

`patches_info_<image_name>.json` example:

```json
[
    {"patch_name": "patch_0.png", "x": 0,   "y": 0},
    {"patch_name": "patch_1.png", "x": 256, "y": 0}
]
```

**After detection**

```
outputs/
├── bboxes/
│   └── <case>/<image_name>.csv                     # Mitosis bounding boxes
└── reconstructed/
    └── <case>/<image_name>_reconstructed.png       # Full annotated image
```

CSV format (pixel coordinates in the normalized full image):

| top_left_x | top_left_y | bottom_right_x | bottom_right_y |
|---|---|---|---|
| 1043.2 | 518.7 | 1081.5 | 560.1 |

> Because edge patches are padded to 256 × 256, the reconstructed image may be slightly larger than the original, with black borders on the right and bottom edges. Box coordinates are unaffected.

---

## Notes & Limitations

- Patches are **non-overlapping**, so a mitotic figure lying on a patch border may be split or missed. Overlapping tiling with box merging (e.g. NMS) would address this.
- Only class `1` is exported by default; use `--class-id` if your model uses a different label mapping.
- Tune the detection threshold with `--conf`.

---

## Acknowledgements

- [MIDOG Challenge](https://midog.deepmicroscopy.org/) — training data and stain reference
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) — Macenko stain normalization
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

In short, you may use, modify, and share this code, but any distributed or network-deployed version, including modified versions, must also be released under AGPL-3.0 with its full source code available.

The detector is built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0 (or a paid Ultralytics Enterprise License). This project follows the same open-source license.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{your_name_2026_mitosis,
  author = {Your Name},
  title  = {Mitotic Figure Detection in H\&E Tumor Images},
  year   = {2026},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```
