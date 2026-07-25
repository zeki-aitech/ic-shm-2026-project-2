import os
import glob
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Semantic Class Mapping & Colors (RGB)
CLASS_COLORS_RGB = {
    "deck": (255, 50, 50),         # Bright Red
    "stay_cable": (0, 230, 255),   # Bright Cyan
    "tower": (50, 220, 50),        # Bright Green
    "foundation": (255, 220, 0)    # Bright Yellow
}

# BGR colors for OpenCV
CLASS_COLORS_BGR = {
    name: (color[2], color[1], color[0])
    for name, color in CLASS_COLORS_RGB.items()
}

DRAW_ORDER = ["deck", "tower", "foundation", "stay_cable"]

def create_image_mask_overlay(image_path, json_path, output_path, alpha=0.4, add_legend=True):
    """
    Overlays Labelme JSON polygon masks onto the given image.
    
    Args:
        image_path (str): Path to input image file.
        json_path (str): Path to input JSON annotation file.
        output_path (str): Path to save the resulting overlapped image.
        alpha (float): Transparency factor for mask overlay (0.0 to 1.0).
        add_legend (bool): Whether to draw a class color legend on the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Warning: Could not read image {image_path}")
        return

    h, w, _ = img.shape
    overlay = img.copy()
    borders = img.copy()

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        sorted_shapes = sorted(
            shapes,
            key=lambda s: DRAW_ORDER.index(s.get('label')) if s.get('label') in DRAW_ORDER else -1
        )

        for shape in sorted_shapes:
            label_name = shape.get('label')
            points = shape.get('points', [])
            if len(points) < 3:
                continue

            pts = np.array(points, dtype=np.int32)
            color_bgr = CLASS_COLORS_BGR.get(label_name, (128, 128, 128))

            # Fill polygon mask on overlay canvas
            cv2.fillPoly(overlay, [pts], color_bgr)
            # Draw crisp polygon border
            cv2.polylines(borders, [pts], isClosed=True, color=color_bgr, thickness=2)

    # Blend original image and colored overlay
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # Combine crisp borders onto blended result
    mask_diff = np.any(borders != img, axis=2)
    blended[mask_diff] = borders[mask_diff]

    if add_legend:
        # Draw legend in top-left corner
        padding = 10
        box_h = 24
        box_w = 40
        start_x, start_y = 15, 15
        
        # Legend background box
        legend_h = len(CLASS_COLORS_RGB) * (box_h + 8) + padding * 2
        legend_w = 170
        sub_img = blended[start_y:start_y+legend_h, start_x:start_x+legend_w]
        white_rect = np.full(sub_img.shape, 30, dtype=np.uint8)
        blended[start_y:start_y+legend_h, start_x:start_x+legend_w] = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 0)
        cv2.rectangle(blended, (start_x, start_y), (start_x+legend_w, start_y+legend_h), (200, 200, 200), 1)

        for idx, (cls_name, color_rgb) in enumerate(CLASS_COLORS_RGB.items()):
            y_offset = start_y + padding + idx * (box_h + 8)
            bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # Color swatch
            cv2.rectangle(blended, (start_x + padding, y_offset), 
                          (start_x + padding + box_w, y_offset + box_h), bgr, -1)
            cv2.rectangle(blended, (start_x + padding, y_offset), 
                          (start_x + padding + box_w, y_offset + box_h), (255, 255, 255), 1)

            # Class label text
            cv2.putText(blended, cls_name, (start_x + padding + box_w + 10, y_offset + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blended)

def process_dataset_overlap(images_dir, json_dir, output_dir, alpha=0.45, add_legend=True):
    """
    Batch processes all matching images and JSON masks in images_dir & json_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.[pP][nN][gG]")) + 
                         glob.glob(os.path.join(images_dir, "*.[jJ][pP][gG]")))

    print(f"🔄 Processing {len(image_files)} images from: {images_dir}")
    print(f"📁 JSON directory: {json_dir}")
    print(f"📁 Target output directory: {output_dir}")

    success_count = 0
    for img_path in tqdm(image_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(json_dir, f"{stem}.json")
        out_path = os.path.join(output_dir, f"{stem}.png")

        create_image_mask_overlay(img_path, json_path, out_path, alpha=alpha, add_legend=add_legend)
        success_count += 1

    print(f"✅ Successfully saved {success_count} overlapped images to: {output_dir}")

if __name__ == "__main__":
    BASE_DIR = "/workspaces/sfm_demo"
    IMAGES_DIR = os.path.join(BASE_DIR, "data/Contest Dataset/images")
    JSON_DIR = os.path.join(BASE_DIR, "data/Contest Dataset/json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data/images_with_masks")

    process_dataset_overlap(IMAGES_DIR, JSON_DIR, OUTPUT_DIR)
