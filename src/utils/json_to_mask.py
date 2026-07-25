import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


CLASS_MAPPING = {
    "background": 0,
    "deck": 1,
    "stay_cable": 2,
    "tower": 3,
    "foundation": 4
}

def convert_json_to_mask(json_path, output_mask_path, image_width=1320, image_height=989):
    """
    Reads a Labelme format JSON file and draws polygon annotations into a uint8 indexed mask.
    Lớp được ưu tiên vẽ theo thứ tự: background (0) -> deck (1) -> tower (3) -> foundation (4) -> stay_cable (2)
    để đảm bảo các dây cáp văng mảnh không bị đè lấp.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    shapes = data.get('shapes', [])
    
    # Sắp xếp theo thứ tự lớp để đảm bảo dải cáp văng mảnh (stay_cable) được vẽ đè lên trên cùng
    draw_order = ["deck", "tower", "foundation", "stay_cable"]
    sorted_shapes = sorted(
        shapes, 
        key=lambda s: draw_order.index(s.get('label')) if s.get('label') in draw_order else -1
    )
    
    for shape in sorted_shapes:
        label_name = shape.get('label')
        points = shape.get('points', [])
        if label_name not in CLASS_MAPPING or len(points) < 3:
            continue
        
        class_id = CLASS_MAPPING[label_name]
        polygon = [(p[0], p[1]) for p in points]
        
        # Vẽ polygon lên mask tạm
        img_poly = Image.new('L', (image_width, image_height), 0)
        ImageDraw.Draw(img_poly).polygon(polygon, outline=1, fill=1)
        poly_arr = np.array(img_poly) > 0
        
        mask[poly_arr] = class_id
        
    # Lưu dưới dạng ảnh PNG 8-bit (P mode hoặc L mode)
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(output_mask_path)

def process_all_jsons(dataset_dir, output_dir):
    json_dir = os.path.join(dataset_dir, "json")
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"🔄 Tìm thấy {len(json_files)} file JSON. Bắt đầu chuyển đổi thành Ground-Truth PNG Masks...")
    
    for jf in tqdm(json_files):
        filename = os.path.splitext(os.path.basename(jf))[0] + ".png"
        out_path = os.path.join(output_dir, filename)
        convert_json_to_mask(jf, out_path)
        
    print(f"✅ Hoàn tất! {len(json_files)} file mask PNG đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    DATASET_DIR = "/home/serene/zeki/sfm_demo/data/Contest Dataset"
    OUTPUT_DIR = "/home/serene/zeki/sfm_demo/outputs/gt_masks"
    process_all_jsons(DATASET_DIR, OUTPUT_DIR)
