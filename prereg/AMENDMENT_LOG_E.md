# AMENDMENT LOG — hướng E, đợt xác nhận VOC
Đóng băng: `prereg\FREEZE_E.txt` (24-11-2026 18:00:36 +0700). Mọi sửa mã sau đóng băng và mọi phân tích **hậu kiểm** ghi ở đây.

| # | Thời điểm | Loại | Nội dung | Ảnh hưởng tới K1–K4 |
|---|---|---|---|---|
| 1 | 25-11 ~14:40 | **hậu kiểm** | `code\e2_06_posthoc.py` → `results\e2_posthoc.json`: so COCO với VOC từng đại lượng; đường mAP tập đầy đủ trên VOC; K1 vế hai với PCHIP/Akima áp lên đường đã qua isotonic | Không đổi K1–K4 (đã chấm bằng mã đóng băng) |

Không có sửa mã sau đóng băng: `sha256sum -c` trên các file trong `FREEZE_E.txt` đều OK trước khi chạy `e2_05_confirm.py`.

## S2FIX (25-11-2026, prompt `prompts\PROMPT_H44_E_S2FIX_20261125.md`) — tất cả là **hậu kiểm / thăm dò**; định nghĩa ở `S2\E_S2FIX_H44.md` mục 0, ghi trước khi tính
| # | Thời điểm | Loại | Nội dung | Ảnh hưởng tới K1–K4 |
|---|---|---|---|---|
| 2 | 25-11 15:20 | hậu kiểm | F1 độ chệch + độ phủ CI (yolo26n; độ phủ: 2 cặp, mẫu ngoài theo trần thời gian — lệch so với prompt, lý do: ước 26 h) | không chấm lại |
| 3 | 25-11 15:20 | hậu kiểm | F2 K1–K4 theo L4–L8 hai bộ; nửa độ rộng theo n ∈ {50…2.500}; số ảnh cần; n nhỏ nhất | không chấm lại |
| 4 | 25-11 15:20 | hậu kiểm | F3 wAP: đọc Fischer 2020 (định nghĩa trùng về trọng số); BD mAP vs wAP + CI hiệu; tỉ trọng 5 lớp | không chấm lại |
| 5 | 25-11 15:20 | thăm dò | F4 yolo26l (+ Faster R-CNN R50-FPN nếu còn ngân sách) trên tập 1.500 ảnh, hai bộ | không chấm lại |
| 6 | 25-11 15:20 | thăm dò | F6 codec học CompressAI (một lần thử) | không chấm lại |

- 25-11 17:35, F4 (trước khi chạy): đo 20 ảnh, yolo26l 0,60 s/ảnh, ước 6,2 h/tập; Faster R-CNN 3,6 s/ảnh, ước 37 h/tập (results/e3_f4_timing.json). Hai tập > 9 h nên chạy VOC trước rồi COCO, --threads 6, --max-hours 13,5. **Không chạy Faster R-CNN** (vượt ngân sách). Đánh giá: code/e3_05_f4eval.py.

## E-FINISH (26-11-2026, prompts `PROMPT_H44_E_FINISH_20261126.md` + `PROMPT_H44_E_S3_DRAFT_20261126.md` G1)
| # | Thời điểm | Loại | Nội dung | Ảnh hưởng tới K1–K4 |
|---|---|---|---|---|
| 7 | 26-11 10:30 | **xác nhận (prereg riêng)** | C6 (BD-rate so với chọn mức nén): `prereg\PREREG_H44_E_C6.md`, đóng băng `prereg\FREEZE_E_C6.txt` trước khi tính q\* trên VOC. Kết quả: C6a ĐẠT 9/9, C6b ĐẠT (vượt tối đa 0,12; 0 vi phạm chắc) | không chấm lại |
| 8 | 26-11 10:40 | **thăm dò** (cả hai bộ đã nhìn) | G1 detector họ thứ ba: torchvision `fasterrcnn_mobilenet_v3_large_fpn`, trọng số `COCO_V1`, `box_score_thresh = 0.001`, `box_detections_per_img = 300`, transform mặc định; tập 1.500 ảnh rtdetr_l mỗi bộ; ảnh gốc + 3 codec × 8 mức; bỏ detection ngoài 20 lớp trên VOC. Mã `code\e4_01_frcnn.py` (suy luận), `code\e3_05_f4eval.py build` (đánh giá, không sửa), `code\e4_02_g1.py` (luật đọc). Bootstrap ghép cặp: cùng `rtboot\|1500` như rtdetr_l. **Luật đọc (chép prompt, xét J→W và J→A, từng bộ; f = FRCNN):** \|BD(f) − BD(yolo26n)\| ≤ 6 ở cả hai cặp → "RT-DETR-L tách khỏi bốn detector CNN"; \|BD(f) − BD(yolo26n)\| ≥ 8 và CI không chứa 0 ở cả hai cặp → "BD-rate phụ thuộc detector, không riêng RT-DETR"; còn lại → "chưa rõ". Dừng nếu mAP ảnh gốc COCO ngoài 25–40. Ngân sách ≤ 10 h: đo 20 ảnh; ≤ 10 h hai bộ → VOC rồi COCO; > 10 h → chỉ VOC; vẫn > 10 h → chỉ JPEG + WebP trên VOC. q\* (δ = 1, 2) của FRCNN: `e5_c6.py extra frcnn_mv3`, nhãn thăm dò | không chấm lại |
- 26-11 10:45 (trước khi chạy): đo 20 ảnh COCO, frcnn_mv3 0,76 s/ảnh (4 luồng), ước 7,9 h/bộ, hai bộ 15,8 h > 10 h ⇒ **chỉ VOC** (`results/e4_g1_timing.json`); --threads 6 --max-hours 9,5. Kiểm mAP ảnh gốc COCO 25–40 không làm được vì không chạy COCO — thay bằng kiểm 20 ảnh không áp dụng; ghi lệch.
- 26-11 16:23 kết quả #8: chạy xong VOC 150 khối, 21.033 s (5,84 h); kiểm pycocotools max |Δ| = 1,1e-16. mAP ảnh gốc FRCNN-MV3 trên tập con = 57,80. Luật đọc ⇒ **"dep"** (BD-rate phụ thuộc detector, không riêng RT-DETR): FRCNN − YOLO26n = +15,47 [12,92; 18,24] (J→W), +13,10 [10,93; 15,35] (J→A); FRCNN − RT-DETR-L = +0,25, −1,74. q\* (δ = 1) FRCNN = 6/6/6 so với 4–5 ở bốn detector đăng ký; chuyển q\* sang FRCNN vượt tối đa 1,58, 9/12 ô vi phạm chắc; chiều ngược lại 0 vi phạm, bpp × 1,31–1,78. Nhãn **thăm dò**; không chấm lại C6a/C6b. Bài sửa: C3, C6, khuyến nghị, hạn chế, kết luận thu hẹp theo kết quả này.
