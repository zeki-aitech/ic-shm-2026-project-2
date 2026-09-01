# IC-SHM 2026 — Project 2: Structure-Aware 3D Semantic Point Cloud Reconstruction for Cable-Stayed Bridges

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![pycolmap](https://img.shields.io/badge/pycolmap-CUDA12-orange.svg)](https://github.com/colmap/pycolmap)

## 📌 1. Tổng quan Dự án (Project Overview)
Dự án tập trung giải quyết bài toán **Tái tạo Đám mây điểm 3D Ngữ nghĩa (3D Semantic Point Cloud Reconstruction)** cho **Cầu dây văng (Cable-Stayed Bridge)** từ ảnh chụp UAV và tham số Structure-from-Motion (SfM - COLMAP) trong khuôn khổ cuộc thi **IC-SHM 2026**.

Mục tiêu chính là chuyển đổi ảnh chụp đa góc nhìn (Multi-view UAV images) kết hợp nhãn phân vùng 2D (Labelme JSON / Ground-Truth PNG Masks) thành mô hình đám mây điểm 3D (`.ply`) chuẩn xác, áp dụng các bộ lọc hình học kết cấu chuyên biệt để khử nhiễu và định hình cấu kiện cầu.

---

## 🏗️ 2. Hệ thống Phân loại Ngữ nghĩa (Semantic Class Taxonomy)

Mô hình phân loại gồm 5 lớp thành phần kết cấu:

| Class ID | Tên nhãn (`label`) | Mã màu RGB | Mô tả hình học & Ý nghĩa |
| :---: | :--- | :--- | :--- |
| **0** | `background` | Xám `(128, 128, 128)` | Nền trời, mây, mặt nước, đất đá cảnh quan |
| **1** | `deck` | Đỏ `(255, 0, 0)` | Bản mặt cầu / Dầm cầu chính (dạng mặt phẳng nằm ngang) |
| **2** | `stay_cable` | Cyan `(0, 255, 255)` | Hệ thống bó dây cáp văng (nằm trên 2 dải mặt phẳng quạt cáp) |
| **3** | `tower` | Xanh lá `(0, 255, 0)` | Trụ tháp cầu (dạng cột trụ đứng vươn cao) |
| **4** | `foundation` | Vàng `(255, 255, 0)` | Bệ trụ, móng cầu (nằm dưới cùng chân cầu) |

---

## 📂 3. Cấu trúc Thư mục (Directory Structure)

```text
ic-shm-2026-project-2/
├── .devcontainer/                # Môi trường container CUDA 12.1, PyTorch, Pycolmap
│   ├── Dockerfile
│   └── devcontainer.json
├── docs/                         # Tài liệu chi tiết về đề bài & khảo sát kỹ thuật
│   └── CONTEST_SPEC_AND_SURVEY.md
├── notebooks/                    # Notebook trực quan hóa 3D tương tác
│   └── 01_visualize_semantic_3d.ipynb
├── src/
│   ├── reconstruction/           # Lõi tái tạo 3D & Xử lý ngữ nghĩa
│   │   ├── colmap_parser.py           # Parser SfM + DLT Triangulation thủ công
│   │   ├── pycolmap_reconstructor.py  # Triangulation LO-RANSAC qua pycolmap GPU
│   │   ├── gpu_pipeline.py            # Pipeline trích xuất SIFT + Re-matching trên GPU
│   │   ├── semantic_projector.py      # Chiếu ngược 2D->3D với Multi-view Majority Voting
│   │   ├── point_cloud_filter.py      # Bộ lọc hình học kết cấu 3D đa giai đoạn
│   │   └── visualizer.py              # Đọc file PLY và render 3D Plotly tương tác
│   └── utils/                    # Tiền xử lý dữ liệu 2D
│       ├── json_to_mask.py            # Chuyển đổi Labelme JSON thành PNG Mask 8-bit
│       └── create_overlay_dataset.py  # Vẽ đè nhãn và legend lên ảnh gốc
└── tests/                        # Bộ kiểm thử đơn vị tự động (Unit Tests)
    ├── test_colmap_parser.py
    ├── test_point_cloud_filter.py
    ├── test_semantic_projector.py
    └── test_visualizer.py
```

---

## ⚙️ 4. Pipeline Xử lý Kỹ thuật (Technical Pipeline)

1. **Tiền xử lý Nhãn 2D (`json_to_mask.py`)**:
   - Chuyển đổi nhãn polygon từ Labelme JSON sang ảnh mask 8-bit.
   - Thứ tự vẽ ưu tiên: $\text{deck (1)} \to \text{tower (3)} \to \text{foundation (4)} \to \mathbf{\text{stay\_cable (2)}}$ để tránh làm mất các sợi cáp mảnh.

2. **Tam giác hóa 3D & Tối ưu GPU (`pycolmap_reconstructor.py` / `gpu_pipeline.py`)**:
   - Sử dụng `pycolmap-cuda12` trích xuất SIFT và sequential matching trên GPU với camera pose cố định, tạo ra đám mây điểm dày đặc ($>80.000$ điểm).

3. **Chiếu ngược Ngữ nghĩa Đa góc nhìn (`semantic_projector.py`)**:
   - Cơ chế **Multi-view Majority Voting**: Gán nhãn cho từng điểm 3D dựa trên tất cả các frame ảnh quan sát thấy điểm đó.
   - Dây cáp (`stay_cable`) chỉ được gán khi đạt đa số tuyệt đối ($>50\%$), kết hợp bảng ưu tiên `TIE_BREAK_PRIORITY` cho các cấu kiện mảnh.

4. **Bộ lọc Hình học Kết cấu Cầu (`point_cloud_filter.py`)**:
   - **Giai đoạn 1**: Loại bỏ Background (Class 0).
   - **Giai đoạn 2**: Statistical Outlier Removal (SOR) độc lập cho từng class.
   - **Giai đoạn 3**: Deck Plane Filter (Khớp mặt phẳng dầm cầu qua 2-pass PCA + MAD residual cut).
   - **Giai đoạn 4**: Deck Core Density Filter (k-NN density trên mặt phẳng dọc/ngang để khử điểm dầm rải rác ngoài biên).
   - **Giai đoạn 5**: Tower Core Tube Filter (Lọc dạng ống trụ quanh từng thân tháp cầu).
   - **Giai đoạn 6**: Stay-cable Structural Envelope Filter (Khung bao toạ độ dọc, ngang, đứng theo hệ trục trọng lực của camera UAV).
   - **Giai đoạn 7**: Stay-cable Left/Right Fan Planes Filter (Lọc cáp theo 2 dải mặt phẳng neo vào tháp).
   - **Giai đoạn 8 (Tùy chọn)**: `project_cables_to_fan_planes` (Chiếu vuông góc các điểm cáp về đúng 2 mặt phẳng quạt dây văng).

---

## 🚀 5. Hướng dẫn Chạy (Quickstart)

### Cài đặt Môi trường
Dự án được cấu hình sẵn môi trường DevContainer / Dockerfile với CUDA 12.1:
```bash
# Cài đặt các thư viện phụ thuộc
pip install numpy scipy pillow opencv-python open3d pycolmap-cuda12 matplotlib plotly pandas jupyterlab tqdm
```

### Chạy Bộ kiểm thử (Unit Tests)
```bash
python3 -m unittest discover tests
```

### Chạy Pipeline Lọc Đám mây điểm
```bash
python3 -m src.reconstruction.point_cloud_filter \
    --input outputs/point_clouds/semantic_bridge_gpu.ply \
    --output outputs/point_clouds/semantic_bridge_filtered.ply \
    --colmap-model outputs/gpu_pipeline/triangulated
```

### Trực quan hóa tương tác 3D
Mở notebook [`notebooks/01_visualize_semantic_3d.ipynb`](notebooks/01_visualize_semantic_3d.ipynb) để khám phá trực quan đám mây điểm 3D tương tác.
