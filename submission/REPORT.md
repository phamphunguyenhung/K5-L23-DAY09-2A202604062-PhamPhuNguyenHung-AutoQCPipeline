# Báo cáo Day 09 — Phạm Phú Nguyễn Hưng, 2A202604062

## 1. Kết luận trước
Pipeline kết luận **HOLD** (mã thoát 10). Vào 224 ảnh raw, nhận 202 (loại 22 ở cửa vào: 2 hỏng, 3 quá nhỏ, 8 trùng byte, 5 rò rỉ test, 4 trùng gần đúng). Lỗi bắt buộc còn lại: **0**. Lý do Hold: sau khi người duyệt (mô phỏng, ngân sách 80 ảnh) xem xong, vẫn còn **38/202 ảnh = 18,8%** có cảnh báo chưa ai duyệt, vượt ngưỡng 10% (`08_summary.json`). Dataset xuất ra `submission/export/clean_v1` chỉ là **bản nháp**, chưa được phát hành.

## 2. Lỗi và phân việc
Số liệu từ `02_error_list_v1` (mẫu số 202 ảnh sau cửa vào; riêng các dòng cửa vào tính trên 224).

| Mã lỗi | Số (ảnh) | Ai sửa | Vì sao |
|---|---:|---|---|
| CORRUPT_IMAGE / BAD_SIZE | 2 / 3 | máy (xoá ảnh) | ảnh không dùng được, quy tắc rõ ràng |
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR | 8 / 4 | máy (xoá ảnh) | trùng ảnh; xem mục 3 về nguy cơ xoá nhầm |
| LEAK_TEST | 5 | máy (xoá ảnh) | ảnh trùng kho test/val, giữ lại sẽ làm số đo train bị phồng |
| BAD_CLASS_ID | 6 | máy (xoá dòng) | id lớp ngoài 0..3, không đoán được ý định |
| BAD_FORMAT | 3 | máy (xoá dòng) | dòng thiếu cột / nan |
| BOX_ZERO | 4 | máy (xoá dòng) | box không có diện tích |
| DUP_BOX | 7 | máy (xoá dòng) | cùng lớp, IoU > 0,9 với dòng trước |
| BOX_OUT_OF_IMAGE | 8 | máy (cắt về [0,1]) | tràn nhẹ, cắt giữ nguyên lớp |
| BAD_CLASS_NAME | 1 (data.yaml; 2 dòng) | máy (ghi lại names) | tên `Person`, `Bus` sai chữ hoa |
| MISSING_LABEL_FILE | 6 | **người** | máy không bịa nhãn; `needs_label` đẩy lên đầu hàng đợi |
| POSSIBLE_MISSING / LOOSE_BOX / WRONG_CLASS / FALSE_POS | 52 / 24 / 18 / 27 | **người** | lỗi ngữ nghĩa, máy chỉ cảnh báo |
| BOX_TINY / BOX_HUGE / EMPTY_LABEL | 4 / 3 / 12 | **người** | cảnh báo, cần xem ảnh mới biết đúng hay sai |

Máy sửa tổng cộng 35 thay đổi (`03_fixes.csv`: 20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`), mỗi dòng có trước → sau. Hàng đợi `05_review_queue.csv` có 125 ảnh, xếp theo tổng trọng số của mã lỗi trong `configs/pipeline.yaml`: ảnh `NEEDS_LABEL` (trọng số 10) lên đầu, rồi ảnh vừa bị xoá dòng (`FIXED_DROP_LINE`) kèm `WRONG_CLASS`/`MISSING`, cuối hàng là ảnh chỉ có `POSSIBLE_MISSING` đơn lẻ (~3,2). Ngân sách 80 ảnh nên 45 ảnh cuối hàng không ai xem; đó là nguồn của 38 ảnh còn cảnh báo ở `07_error_list_v3` (còn 57 dòng cảnh báo: 21 FALSE_POS, 20 LOOSE_BOX, 13 MISSING, 3 WRONG_CLASS, 7 TINY, 3 HUGE, 4 EMPTY_LABEL).

## 3. Calibration (2 ngưỡng)
Nguồn: `submission/calibration/calibration_report.md` (39 ảnh calib có lỗi chèn sẵn).

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | DUP_IMAGE_NEAR: 1,00/0,50 → 1,00/0,67 (EXACT giữ 1,00/1,00) | cờ này tự xoá ảnh nên cần precision 1,00. Từ 4 đến 14 precision giữ 1,00 và recall 0,67 (bình nguyên), chọn 12 để còn lề an toàn. Dist 16 nâng recall lên 0,83 nhưng xem raw mới thấy sai (xem dưới) |
| `second_opinion.conf` | 0,25 | **0,6** | WRONG_CLASS 0,56/0,83 → 0,62/0,83; MISSING 0,50/1,00 → 0,57/1,00 | đây là cảnh báo cho người nên giữ recall MISSING = 1,00; 0,6 tăng precision mà không mất recall. 0,8 cho WRONG_CLASS precision 1,00 nhưng recall MISSING rớt 0,75 (bỏ sót vật bị thiếu nhãn) |

**Chỗ calibration có thể đánh lừa:** `phash_dist=16` có F1 tốt nhất trên calib (NEAR 1,00/0,83) vì calib chỉ có 6 ảnh gần trùng và không có ảnh "giống nhưng khác". Chạy thử trên raw (ngoài pipeline chính, thư mục tạm): dist 12 loại 22 ảnh, dist 16 loại 31 ảnh; 3 cặp thêm ở khoảng cách 16 là `img_0011↔img_0008`, `img_0084↔img_0074`, `img_0177↔img_0113`. Mở xem thì **2/3 cặp là ảnh khác hẳn nhau** (cảnh tennis với cảnh trượt tuyết, cảnh lướt sóng với chân dung), chỉ 1 cặp (`img_0084`) là trùng thật. Chọn 16 sẽ xoá nhầm ảnh tốt, nên không chọn dù calib "đẹp". Ngoài ra, precision/recall của từng mã chỉ dựa trên 2 đến 8 lỗi chèn nên mỗi ô nhảy từng bước 12 đến 50%.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn thật.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không người duyệt): sau sửa tự động còn 6 ảnh `MISSING_LABEL_FILE` (máy không bịa nhãn), kết luận **REWORK**, mã thoát **20**, log ghi "KHÔNG xuất: còn lỗi bắt buộc" và thư mục `work/exp_nb` **không được tạo**. Chạy bình thường (có người duyệt mô phỏng gán nhãn cho 6 ảnh đó) thì còn 0 lỗi bắt buộc, kết luận Hold, mã thoát 10. Chạy lại lần hai cho `07_error_list_v3.csv` giống hệt (tái lập).

**(b) Điểm mù đã biết:**
1. *Ảnh trùng biến đổi mạnh:* pHash ở ngưỡng 12 bỏ sót cặp `img_0084↔img_0074` (khoảng cách 16) dù là cùng ảnh; hạ recall xuống 0,67 trên calib. Nâng ngưỡng thì xoá nhầm ảnh khác (mục 3).
2. *Nhãn sai mà model lớn cũng đồng ý:* `second_opinion` chỉ so với một model (yolo26l); nếu cả nhãn nháp và yolo26l cùng nhầm (vd. truck/bus) thì không có cảnh báo. Calib cũng cho thấy LOOSE_BOX chỉ đạt precision 0,20 đến 0,50, nên box lỏng nhẹ hay lọt, hoặc bị cờ nhầm.
3. *Ngưỡng 10% và ngân sách 80 ảnh:* 45 ảnh cuối hàng đợi không ai xem; ảnh nào trong đó thật sự lỗi sẽ lọt vào bản nháp.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Chưa chạy.** Lý do: cần GPU/Colab và lần chạy này giới hạn 30 phút, ưu tiên phần lõi; không có file `submission/results/`, không có số liệu nào được bịa. Notebook `notebooks/day09_train_compare.ipynb` giữ nguyên để chạy sau khi dataset đạt Release (hiện chỉ có bản nháp Hold).

## 6. Quyết định và ghi chú AI
**(a) Quyết định: HOLD.** Điều kiện để thành Release: giảm ảnh còn cảnh báo chưa duyệt xuống dưới 10% (< 21/202 ảnh), tức người duyệt thêm tối thiểu 18 ảnh trong 45 ảnh cuối hàng đợi (ưu tiên ảnh có `POSSIBLE_FALSE_POS` và `POSSIBLE_LOOSE_BOX`), hoặc nâng ngân sách duyệt; đồng thời không được còn lỗi bắt buộc.

**(b) Dùng AI:** Claude Code (Sonnet) viết 4 module `label_format`, `autofix`, `gate`, `second_opinion`.
- Prompt chính: dán đề bài Day 09 kèm yêu cầu tự triển khai, tự test, tự sửa theo docstring của từng stub.
- Cách kiểm: `pytest` 79/79 pass (`artifacts/baseline_tests.log` 55 fail → `artifacts/final_tests.log` 79 pass); `make calibrate` trên lỗi chèn sẵn; chạy `pipeline.run` hai lần so `07_error_list_v3.csv`; xem tay 3 cặp ảnh bị cho là trùng.
- Một lần AI sai: lệnh tạo file đầu tiên dài quá giới hạn shell của môi trường nên bị chặn và phải ghi từng file bằng công cụ sửa file; ngoài ra lựa chọn đầu theo F1 của calib (`phash_dist=16`) là sai khi đối chiếu bằng mắt trên raw (mục 3).
