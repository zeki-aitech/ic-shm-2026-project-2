# IC-SHM 2026 (Project 2) — Đặc Tả Cuộc Thi & Báo Cáo Khảo Sát Kỹ Thuật

**Dự án**: Structure-Aware 3D Semantic Point Cloud Reconstruction for Cable-Stayed Bridges  
**Thời gian khảo sát**: Tháng 09/2026  
**Mục tiêu**: Tái tạo đám mây điểm 3D nhận diện kết cấu và phân loại ngữ nghĩa cho cầu dây văng từ dữ liệu ảnh UAV và tham số SfM.

---

## I. TỔNG QUAN BÀI TOÁN & YÊU CẦU ĐỀ BÀI

### 1. Bối cảnh & Ứng dụng thực tiễn
Cầu dây văng (Cable-Stayed Bridge) là công trình giao thông trọng điểm có quy mô nhịp lớn và cấu trúc không gian phức tạp. Việc kiểm tra định kỳ hiện trường thủ công thường tốn kém chi phí, rủi ro cao và khó tiếp cận các vị trí tháp cao hay bó cáp văng.

Việc ứng dụng thiết bị bay không người lái (UAV) kết hợp thị giác máy tính (Computer Vision) và tái tạo 3D (Structure-from-Motion / Multi-View Stereo) cho phép xây dựng mô hình số 3D đám mây điểm hiện trạng (As-is Digital Twin). Bài toán đặt ra là: **Làm thế nào để đám mây điểm 3D không chỉ chứa tọa độ hình học $(x, y, z)$ mà còn mang thông tin ngữ nghĩa (Semantic labels) phân tách chính xác từng cấu kiện cầu?**

---

### 2. Các thách thức kỹ thuật cốt lõi
1. **Sự không cân bằng và chênh lệch hình học giữa các cấu kiện**:
   - **Bản mặt cầu / Dầm (`deck`)**: Bề mặt diện tích lớn dạng phẳng ngang.
   - **Tháp cầu (`tower`)**: Cột trụ đứng thẳng vươn cao.
   - **Móng cầu (`foundation`)**: Nằm ở phần đáy tiếp giáp mặt đất/nước.
   - **Dây cáp văng (`stay_cable`)**: Cực kỳ mảnh và chiếm tỷ lệ pixel rất nhỏ trong ảnh UAV. Khi tam giác hóa 3D từ các góc nhìn khác nhau, điểm đặc trưng dễ bị nhiễu do điểm ảnh bắt nhầm nền trời hoặc mặt nước phía sau.
2. **Nhiễu tam giác hóa (Triangulation Noise & Drift)**:
   - Dữ liệu SfM ban đầu không có file `points3D.txt` (chỉ có track 2D). Việc tam giác hóa với baseline ngắn từ UAV dễ sinh ra các điểm trôi dạt (outliers) nằm rải rác ngoài không gian cầu.
3. **Dữ liệu nhãn bán giám sát (Semi-supervised)**:
   - Chỉ một phần ảnh UAV được gán nhãn đa giác (Labelme JSON), đòi hỏi thuật toán chiếu ngược và biểu quyết đa góc nhìn (Multi-view Voting) phải có cơ chế lọc và gán nhãn thông minh.

---

## II. ĐẶC TẢ DỮ LIỆU CUỘC THI (CONTEST DATASET)

### 1. Cấu trúc thư mục dữ liệu
```text
Contest Dataset/
├── camera_parameters/          # Tham số SfM / Tư thế camera
│   ├── cameras.txt             # Thông số nội suy camera (Intrinsics)
│   ├── images.txt              # Tư thế từng ảnh (Extrinsics) & vết quan sát 2D
│   ├── rigs.txt                # Khung giàn camera (nếu có)
│   └── frames.txt              # Khung thời gian ảnh
├── images/                     # ~300+ ảnh UAV đã có nhãn tương ứng
├── unlabeled_Images/           # ~100 ảnh UAV chưa có nhãn
└── json/                       # File nhãn đa giác Labelme (.json) cho tập ảnh 'images'
```

### 2. Tham số Camera & SfM
- **Mô hình**: `SIMPLE_RADIAL` (Camera đơn chia sẻ cho toàn bộ ảnh).
- **Kích thước ảnh**: $1320 \times 989$ pixels.
- **Tiêu cự**: $f \approx 925.7016$ px.
- **Tọa độ tâm ảnh (Principal Point)**: $c_x = 660.0, c_y = 494.5$ px.
- **Hệ số méo xuyên tâm**: $k_1$.
- **Tổng số ảnh**: 400 ảnh.
- **Tổng số track đặc trưng**: 86,336 tracks.

### 3. Bảng phân lớp Ngữ nghĩa (Semantic Classes)

| Class ID | Tên cấu kiện | Màu RGB | Màu HEX | Đặc điểm nhận dạng |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `background` | `(128, 128, 128)` | `#808080` | Nền trời, mây, mặt nước sông, cảnh quan |
| **1** | `deck` | `(255, 0, 0)` | `#FF0000` | Bản mặt cầu / Dầm cầu chính |
| **2** | `stay_cable` | `(0, 255, 255)` | `#00FFFF` | Hệ thống bó dây cáp văng |
| **3** | `tower` | `(0, 255, 0)` | `#00FF00` | Trụ tháp cầu |
| **4** | `foundation` | `(255, 255, 0)` | `#FFFF00` | Mố, móng trụ cầu |

---

## III. KIẾN TRÚC VÀ CÁC GIẢI THUẬT NÂNG CAO TRONG DỰ ÁN

### 1. Tiền xử lý Nhãn 2D (`json_to_mask.py`)
Quy tắc vẽ đè (Draw Order) được thiết kế đặc biệt:
$$\text{deck (1)} \longrightarrow \text{tower (3)} \longrightarrow \text{foundation (4)} \longrightarrow \mathbf{\text{stay\_cable (2)}}$$
Việc vẽ `stay_cable` sau cùng đảm bảo các dải cáp mảnh không bị các vùng đa giác dầm hay tháp lớn đè mất nhãn.

### 2. Chiếu ngược 2D sang 3D & Multi-view Voting (`semantic_projector.py`)
- Đối với mỗi điểm 3D $X_i$, tập hợp tất cả các quan sát 2D tương ứng qua các ảnh $I_k$.
- Lấy nhãn 2D $L_{ik} = \text{Mask}_k(u_{ik}, v_{ik})$.
- **Quy tắc bỏ phiếu đặc biệt**:
  - Dây cáp (`stay_cable`) chỉ được công nhận nếu số phiếu đạt **đa số tuyệt đối** ($> 50\%$).
  - Nếu không đạt, loại bỏ phiếu cáp và tiến hành chọn lớp đa số trong các lớp còn lại kết hợp với bảng trọng số `TIE_BREAK_PRIORITY` (ưu tiên `tower` > `foundation` > `deck` > `background`).

### 3. Pipeline Lọc Hình học Kết cấu 3D (`point_cloud_filter.py`)

1. **Chuẩn hóa Hệ tọa độ Cầu (Bridge Frame)**:
   - Ước lượng vector trọng lực $v$ (trục đứng) từ tư thế góc quay camera UAV (do drone bay có roll $\approx 0$).
   - Trục dọc $u$ (longitudinal) và trục ngang $w$ (lateral) được xây dựng trực giao với $v$.
2. **Lọc mặt phẳng Dầm (`filter_deck_plane`)**:
   - Khớp mặt phẳng PCA 2-pass trên tập điểm dầm và cắt bỏ phần dư MAD (Median Absolute Deviation).
3. **Lọc mật độ lõi Dầm (`filter_deck_core_density`)**:
   - Sử dụng k-NN density trên mặt phẳng $(u, w)$ để loại bỏ các điểm dầm phân tán ngoài hành lang cầu.
4. **Lọc ống trụ thân Tháp (`filter_tower_core`)**:
   - Gom cụm K-Means các thân tháp theo trục dọc, sau đó lọc dạng ống trụ (tube) quanh tâm từng thân tháp.
5. **Lọc khung bao Cáp văng (`filter_cable_structural_envelope`)**:
   - Giới hạn cáp nằm trong khoảng cao độ: trên mặt dầm và dưới đỉnh tháp; nằm trong chiều dài nhịp cầu.
6. **Lọc mặt phẳng Quạt cáp (`filter_cable_tower_planes`)**:
   - Xác định 2 mặt phẳng quạt cáp trái/phải dựa trên tọa độ mặt tháp / mép dầm. Loại bỏ cáp có độ lệch ngang vượt ngưỡng dung sai $\tau$.
7. **Chiếu hình học cáp về mặt phẳng quạt (`project_cables_to_fan_planes`)**:
   - Chiếu vuông góc điểm cáp về đúng 2 mặt phẳng quạt cáp gần nhất, giữ nguyên cao độ $z$ phục vụ dựng hình CAD/BIM chuẩn xác.

---

## IV. ĐỊNH HƯỚNG MỞ RỘNG TIẾP THEO

1. **Phân vùng ảnh tự động bằng Deep Learning**: Tích hợp các mô hình Segment Anything (SAM / HQ-SAM) hoặc SegFormer/Mask2Former để tự động gán nhãn cho tập ảnh chưa có nhãn (`unlabeled_Images`).
2. **Cấu hình đường dẫn dữ liệu tập trung**: Sử dụng file cấu hình YAML (`configs/config.yaml`) hoặc `argparse` để chạy độc lập trên mọi môi trường.
3. **Phân tích kết cấu / Đo độ võng và dao động**: Kết hợp đám mây điểm 3D ngữ nghĩa với bài toán trích xuất dao động cáp từ video UAV (Project 1).
