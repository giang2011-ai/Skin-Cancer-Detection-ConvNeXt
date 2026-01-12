# Skin Cancer Classification with Deep Learning (ConvNeXt-Tiny)

## Giới thiệu

Dự án này triển khai một mô hình học sâu cho bài toán **phân loại ảnh da liễu nhị phân** (ung thư da / không ung thư) dựa trên kiến trúc **ConvNeXt-Tiny**.

Mục tiêu chính của bài toán là **phát hiện sớm các ca ung thư da**, do đó ưu tiên các thước đo phản ánh khả năng **không bỏ sót mẫu dương tính**, đặc biệt trong bối cảnh **dữ liệu mất cân bằng nghiêm trọng**.

Pipeline huấn luyện được xây dựng theo hướng thực tiễn, kế thừa các thành phần cốt lõi của CNN truyền thống và tích hợp các cải tiến hiện đại nhằm đạt hiệu năng cao và ổn định.

---

## Bộ dữ liệu

- **Nguồn dữ liệu**: ISIC Challenge 2024 (Kaggle)  
- **Loại dữ liệu**: Ảnh da liễu đã được cắt vùng tổn thương  
- **Số lớp nhãn**: 2 (ung thư, không ung thư)

### Đặc điểm dữ liệu

- Tỉ lệ mẫu dương tính rất thấp (≈ **0.1%**)  
- Dữ liệu đã có nhãn đầy đủ, không có giá trị thiếu  

---

## Chia dữ liệu

Dữ liệu được chia thành:

- **Train**: 75%  
- **Validation**: 15%  
- **Test**: 15%  

Ngoài ra, phân bố nhãn trong các tập dữ liệu được điều chỉnh để **tiệm cận hơn với bối cảnh sàng lọc ung thư thực tế**.

---

## Kiến trúc mô hình

- **Backbone**: ConvNeXt-Tiny (pretrained trên ImageNet)  
- **Đầu ra**: 1 logit (phân loại nhị phân)  
- **Hàm kích hoạt đầu ra**: Sigmoid (trong giai đoạn suy luận)

ConvNeXt là kiến trúc CNN hiện đại, được đề xuất nhằm hiện đại hóa CNN theo triết lý của Vision Transformer, bao gồm:

- Convolution kernel lớn (large kernel convolution)  
- Depthwise convolution  
- Layer Normalization  
- Cấu trúc block đơn giản và hiệu quả  

Phiên bản **ConvNeXt-Tiny** có số lượng tham số vừa phải, phù hợp cho huấn luyện và thử nghiệm trong điều kiện tài nguyên tính toán hạn chế.

---

## Phương pháp huấn luyện

### Xử lý mất cân bằng dữ liệu

- **Weighted Random Sampling** được sử dụng để tăng tần suất xuất hiện của lớp ung thư trong các batch huấn luyện, giúp mô hình học tốt hơn các đặc trưng của lớp hiếm.

### Tăng cường dữ liệu

- Xoay ngẫu nhiên  
- Lật ngang, lật dọc  
- Điều chỉnh màu sắc  
- Chuẩn hóa theo thống kê ImageNet  

### Ổn định mô hình

- **Exponential Moving Average (EMA)** được áp dụng lên trọng số mô hình để giảm dao động trong quá trình huấn luyện.  
- Mô hình tốt nhất được lựa chọn dựa trên **PR-AUC của phiên bản EMA** trên tập validation.

---

## Ensemble & Test-Time Augmentation

- Huấn luyện mô hình với **nhiều seed khác nhau**  
- Kết hợp các mô hình tốt nhất theo phương pháp **ensemble**  
- Áp dụng **Test-Time Augmentation (TTA)** trong giai đoạn suy luận để tăng độ ổn định của dự đoán  

---

## Tham số huấn luyện

| Tham số            | Giá trị                 |
|--------------------|-------------------------|
| Kiến trúc          | ConvNeXt-Tiny           |
| Kích thước ảnh     | 384 × 384               |
| Batch size         | 16                      |
| Epochs             | 20                      |
| Optimizer          | AdamW                   |
| Learning rate      | 5e-5                    |
| Weight decay       | 1e-2                    |
| Scheduler          | OneCycleLR              |
| Loss function      | BCEWithLogitsLoss       |
| Số seed            | 3                       |

---

## Đánh giá mô hình

### Thước đo đánh giá

- **PR-AUC (Precision–Recall AUC)** – thước đo chính  
- **ROC-AUC** – thước đo bổ sung  

PR-AUC được lựa chọn làm thước đo chính vì phản ánh trực tiếp trade-off giữa **Precision** và **Recall**, phù hợp với mục tiêu ưu tiên phát hiện đầy đủ các ca ung thư da, thay vì tối ưu độ chính xác tổng thể.

### Trực quan hóa kết quả

- **Precision–Recall Curve (hình chính)**: đánh giá hiệu năng cuối cùng của mô hình ensemble  
- **PR-AUC theo epoch (hình phụ)**: quan sát xu hướng hội tụ và độ ổn định của quá trình huấn luyện  

---

## Cấu trúc thư mục

```text
.
├── data/
│   ├── train_ref10.csv
│   ├── val_ref10.csv
│   └── test_ref10.csv (nếu có)
├── outputs/
│   ├── best_seed*_convnext_tiny.pt
│   ├── history_seed*_convnext_tiny.csv
│   ├── pr_curve_ensemble.png
│   └── pr_auc_by_epoch.png
├── train.py
└── README.md
