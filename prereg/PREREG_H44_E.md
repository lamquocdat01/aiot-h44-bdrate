# PREREG H44 — hướng E, đợt xác nhận trên PASCAL VOC 2007 test (24-11-2026)
Ghi và đóng băng trước khi mở chú thích, mã hoá hay chạy detector trên VOC. Lúc ghi chỉ mới tải và giải nén `VOCtest_06-Nov-2007.tar` từ trang chính thức (451.020.800 byte; SHA-256 `6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892`) và đếm số file: 4.952 ảnh JPEG, 4.952 file chú thích, `ImageSets/Main/test.txt` 4.952 dòng.
Nguồn của các giả thuyết: `prompts\PROMPT_H44_E_S2_CONFIRM_20261124.md` mục B (chép nguyên), Amendment A1 trong `S0\PLAN_H44_v4_ERRBUDGET_20261124.md`. Số thăm dò trên COCO: `S1\E_STEP1_H44.md`, `results\e1_budget.json`, `results\e2_00_wap_coco.json`.

## 1. Thiết kế VOC (chép nguyên prompt, kèm chi tiết cài đặt)
> 4.952 ảnh test (kiểm số ảnh sau giải nén; khác → ghi, vẫn chạy); codec JPEG/WebP/AVIF/HEVC × 8 mức cùng tham số chất lượng như P1 (không dò lại theo bpp trên VOC); yolo26n + yolo26s toàn bộ (HEVC chỉ yolo26n); rtdetr_l trên 1.500 ảnh rút ngẫu nhiên (seed +210), 3 codec. Chú thích VOC → định dạng COCO; ánh xạ 20 lớp VOC sang lớp COCO (ghi bảng ánh xạ; `diningtable→dining table`, `tvmonitor→tv`, `sofa→couch`, `motorbike→motorcycle`, `aeroplane→airplane`, `pottedplant→potted plant`); bỏ detection thuộc 60 lớp còn lại; đối tượng `difficult = 1` đặt `iscrowd = 1`. Thước đo: mAP@[.5:.95] trên 20 lớp. BD: isotonic + PCHIP, miền mAP, lưới L8, ln(bpp). Bootstrap: có hoàn lại, ghép cặp, 1.000 mẫu, CI percentile 95 %.

**Chi tiết cài đặt (ghi trước, mã ở `code\e2_*.py`):**
| Mục | Cách làm |
|---|---|
| Ảnh | Mọi ảnh trong `ImageSets/Main/test.txt` (kể cả ảnh chỉ có đối tượng difficult). image_id = int(tên file) |
| Ánh xạ lớp | 14 lớp cùng tên + 6 lớp đổi tên ở trên; category_id = id COCO của lớp đích (`e2_voc_common.VOC2COCO`) |
| Hộp | VOC (xmin, ymin, xmax, ymax), gốc 1 → COCO [xmin−1, ymin−1, xmax−xmin+1, ymax−ymin+1]; area = w·h |
| Mã hoá | `p1_01_encode.encode` với `results\p1_quality_levels.json` (JPEG/WebP/AVIF/HEVC × 8 mức, không dò lại); ≤ 4 tiến trình |
| Detector | Cài đặt P1: imgsz 640, conf 0,001, max_det 300, CPU FP32; lưu đủ 80 lớp, bỏ 60 lớp ngoài VOC khi đánh giá; 1 tiến trình |
| Đánh giá | COCOeval bbox, area all, maxDets 100, catIds = 20 lớp; mAP nhanh từ cờ khớp lưu (`e2_fastwap`) |
| bpp | byte·8/(W·H) từng ảnh; bpp của một mẫu = trung bình có trọng số (số lần ảnh xuất hiện) |
| Tập rtdetr_l | `default_rng(20261124 + 210).choice(4952, 1500, replace=False)` trên danh sách id đã sắp xếp |
| Bootstrap tập đầy đủ | n ∈ {166, 500, 1.000, 2.500, N}; luồng `default_rng([20261124 + 200, n])`; cùng chỉ số cho mọi (detector, điểm) |
| Bootstrap tập 1.500 | luồng `default_rng([20261124 + 200, 1500, 1])` trên vị trí trong tập con; rtdetr_l và yolo26n dùng cùng chỉ số |
| Nửa độ rộng | (p97,5 − p2,5)/2 của BD isotonic qua 1.000 mẫu (bỏ mẫu NaN, báo tỉ lệ) |
| PCHIP, Akima trên tập đầy đủ | chỉ tính khi cả hai đường đơn điệu tăng chặt theo mAP và khoảng chung > 0; nếu không → "không tính được" |
| wAP | Σ_k n_k·AP_k / Σ_k n_k, n_k = số đối tượng GT không bị bỏ qua của lớp k (trong mẫu, có trọng số), AP_k@[.5:.95] |

## 2. Giả thuyết xác nhận (chép nguyên)
- **K1 (lấy mẫu trội hơn cách khớp đường).** Ở n = 500, với mỗi ô trong 6 ô (3 cặp lõi × yolo26n/s): nửa độ rộng CI ≥ 3,0 điểm % **và** lớn hơn khoảng max − min của BD giữa {PCHIP, Akima, isotonic} trên tập đầy đủ (cách nào không tính được thì bỏ khỏi khoảng; còn < 2 cách thì ô tính là đạt vế hai và ghi chú). ĐẠT nếu ≥ 5/6 ô.
- **K2 (codec sát nhau).** Nửa độ rộng CI của WebP→AVIF ≥ 1,5 × của JPEG→WebP, ở n = 166 và n = 500, cho cả yolo26n và yolo26s. ĐẠT nếu ≥ 3/4.
- **K3 (họ detector).** Trên cùng 1.500 ảnh, isotonic L8: |BD(rtdetr_l) − BD(yolo26n)| ≥ 8 điểm % và CI ghép cặp 95 % không chứa 0, ở cả JPEG→WebP và JPEG→AVIF; **và** trên tập đầy đủ |BD(yolo26n) − BD(yolo26s)| ≤ 6 điểm % ở cả ba cặp lõi. ĐẠT nếu thoả cả hai vế.
- **K4 (wAP).** Kết quả phần A (thăm dò COCO, `results\e2_00_wap_coco.json`): r_COCO(500) = 0,784 (trung vị 6 ô; ô: 0,641 / 0,886 / 0,810 / 0,633 / 0,758 / 0,818) ≤ 0,85. Theo luật của prompt, K4 đăng ký là: **trên VOC, trung vị r(500) ≤ 0,90**, với r = nửa độ rộng CI 95 % của BD (isotonic, L8, n = 500) tính trên wAP chia cho nửa độ rộng tính trên mAP, trung vị qua 6 ô lõi (3 cặp × yolo26n/s). wAP định nghĩa ở mục 1 ("wAP, trung bình theo số đối tượng", theo mô tả của Herglotz et al. TIP 2024; công thức chính xác trong bài gốc của Fischer et al. chưa đọc).

## 3. Kiểm tra vận hành (không phải giả thuyết)
- **DỪNG-V1:** mAP của yolo26n trên ảnh gốc VOC (20 lớp) < 30 hoặc > 80 → dừng, báo (nghi sai ánh xạ lớp).
- **DỪNG-V2:** mAP nhanh lệch COCOeval > 1e-9 trên tập đầy đủ → dừng. Kiểm trên 5 điểm: yolo26n orig, jpeg_3, hevc_2; yolo26s avif_5; rtdetr_l webp_4 (tập 1.500); cho cả mAP và wAP.
- **Hồi quy COCO:** dựng lại yolo26n jpeg_3 (COCO, tập đầy đủ) bằng mã tham số hoá; lệch mAP so với bản lưu ≤ 1e-12.

## 4. Báo kèm, không cổng
Tỉ lệ đường không đơn điệu và tỉ lệ PCHIP tính được theo n; nửa độ rộng CI ở n ∈ {166, 500, 1.000, 2.500} (và N); n nhỏ nhất trong {166, 500, 1.000, 2.500, N} để CI của mỗi cặp không chứa 0; bảng bpp thực đạt trên VOC; các ô HEVC (yolo26n) báo mô tả.

## 5. Thứ tự chạy và ngân sách (ghi trước)
- Đo trước trên 15 ảnh COCO (không phải VOC), 1 tiến trình: yolo26n ≈ 0,11–0,28 s/ảnh, yolo26s ≈ 0,30–0,34, rtdetr_l ≈ 0,9–1,4 tuỳ số luồng. Ước suy luận ≈ 27 giờ, vượt ngân sách 26 giờ máy.
- Thứ tự: mã hoá → yolo26n (33 điểm) → yolo26s (25) → rtdetr_l (25 điểm, 1.500 ảnh) → đánh giá → bootstrap → phân tích. Luồng torch: 4.
- Hết ngân sách trong lúc chạy rtdetr_l → dừng ở checkpoint; K1, K2, K4 và vế 2 của K3 vẫn phân tích trên yolo26n/s; **vế 1 của K3 ghi "chưa kiểm được (thiếu dữ liệu)"**, K3 không được tính là ĐẠT. Không thay rtdetr_l bằng tập nhỏ hơn.

## 6. Luật sau đóng băng
- Không đổi ngưỡng, không đổi mã đã băm. Lỗi mã phát hiện sau đóng băng → sửa, ghi vào `prereg\AMENDMENT_LOG_E.md` kèm diff và lý do, báo cả kết quả trước và sau sửa nếu đã có.
- Mọi phân tích nghĩ ra sau khi thấy số VOC ghi nhãn **hậu kiểm** trong `prereg\AMENDMENT_LOG_E.md`.
- Vượt 26 giờ máy → dừng ở checkpoint và báo.
