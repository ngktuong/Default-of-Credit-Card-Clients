# Credit Scoring Model – German Credit Data

Xây dựng mô hình dự báo xác suất khách hàng vỡ nợ (default) từ bộ dữ liệu **German Credit Data**, sử dụng phương pháp **WOE (Weight of Evidence) / IV (Information Value)** kết hợp **Logistic Regression**, sau đó quy đổi thành **Scorecard** theo điểm số và xác định **ngưỡng cutoff** duyệt hồ sơ tín dụng.

## 1. Bộ dữ liệu

- **Nguồn:** [German Credit Data – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), do Prof. Dr. Hans Hofmann (Đại học Hamburg) cung cấp.
- **Số quan sát:** 1000 khách hàng.
- **Số biến:** 20 biến đầu vào (7 biến số, 13 biến phân loại) + 1 biến mục tiêu (`target`).
- **Biến mục tiêu:** nhị phân — `0 = khách hàng tốt`, `1 = khách hàng xấu/default` (đã ánh xạ lại từ mã gốc 1/2 của UCI).
- **Một số biến chính:** tình trạng tài khoản thanh toán (`checking_account`), lịch sử tín dụng (`credit_history`), mục đích vay (`purpose`), số tiền vay (`credit_amount`), tài khoản tiết kiệm (`savings_account`), thời gian làm việc hiện tại (`employment_since`), tuổi (`age`), nghề nghiệp (`job`)...
- **Chi phí phân loại sai (cost matrix):** phân loại nhầm khách xấu thành tốt (approve nhầm) có chi phí gấp 5 lần so với phân loại nhầm khách tốt thành xấu (reject nhầm) — phản ánh đúng bản chất rủi ro tín dụng trong thực tế.
- File mô tả gốc: `german.doc` (là file text mô tả dữ liệu theo định dạng gốc của UCI, không phải văn bản Word).

> Dữ liệu gốc `german.data` cần được tải trực tiếp từ UCI Repository và đặt vào thư mục `data/` (không đính kèm trong repo do bản quyền/kích thước).

## 2. Phương pháp

1. **Tiền xử lý:** đọc dữ liệu, ánh xạ các mã biến phân loại (A11, A12, ...) sang nhãn dễ đọc.
2. **EDA:** khảo sát phân phối biến số/phân loại, kiểm tra outlier, tỷ lệ default theo từng nhóm, ma trận tương quan.
3. **Chia dữ liệu:** train/test theo tỷ lệ 70/30, stratify theo `target`.
4. **WOE / IV:** thử nhiều cách chia bin cho từng biến, tính WOE/IV, gom nhóm các mức có WOE gần nhau để tăng ổn định.
5. **Chọn biến:** dựa trên IV (0.1–0.5), tránh biến có IV quá cao (nghi ngờ leakage) hoặc quá thấp (kém dự báo). WOE được tính **trên tập train** và áp dụng lại cho tập test để tránh rò rỉ dữ liệu.
6. **Kiểm định chéo (Stratified 5-fold CV):** đánh giá độ ổn định của bộ biến đã chọn, WOE được tính lại độc lập trong từng fold.
7. **Kiểm tra đa cộng tuyến (VIF)** giữa các biến WOE trước khi huấn luyện mô hình cuối.
8. **Logistic Regression** trên các biến WOE đã chọn.
9. **Đánh giá mô hình:** AUC, Gini, KS trên tập train/test; ROC curve.
10. **PSI (Population Stability Index):** kiểm tra độ ổn định phân phối xác suất dự báo giữa train và test.
11. **Xây dựng Scorecard:** quy đổi hệ số mô hình + WOE thành điểm số theo chuẩn PDO (Points to Double the Odds).
12. **Xác định ngưỡng cutoff:** dùng chỉ số Youden's J trên tập test, quy đổi sang điểm cutoff để đưa ra quyết định Approve/Reject.

## 3. Kết quả chính

| Chỉ số | Train | Test |
|---|---|---|
| AUC | 0.7774 | 0.7852 |
| Gini | 0.5548 | 0.5704 |
| KS | 0.4599 | 0.4952 |

- **Cross-validation (5-fold):** AUC trung bình = 0.7586 ± 0.0275, KS trung bình = 0.4551 ± 0.0314.
- **VIF:** tất cả các biến WOE đều < 1.2 → không có dấu hiệu đa cộng tuyến.
- **PSI (train vs test):** 0.0209 (< 0.1) → mô hình ổn định, không lệch phân phối đáng kể.
- **Biến được chọn vào mô hình cuối:** `duration`, `credit_history`, `checking_account`, `savings_account` (dạng WOE).
- **Ngưỡng cutoff:** xác định bằng Youden's J, tương ứng khoảng 505 điểm scorecard (Base Score = 600, PDO = 20).

## 4. Cấu trúc thư mục

```
├── notebook_updated.ipynb   # Notebook chính: toàn bộ pipeline
├── german.doc                # Mô tả gốc bộ dữ liệu (UCI)
├── data/
│   └── german.data           # Dữ liệu gốc (tự tải từ UCI, không kèm trong repo)
├── requirements.txt
└── README.md
```

## 5. Cách chạy

```bash
pip install -r requirements.txt
jupyter notebook notebook_updated.ipynb
```

## 6. Hạn chế và hướng cải thiện

- Việc chia bin (binning) hiện thực hiện thủ công dựa trên phân phối dữ liệu và giá trị WOE; có thể thử các phương pháp binning tối ưu tự động (ví dụ OptBinning) để so sánh.
- Mô hình cuối chỉ sử dụng 4 biến; có thể thử nghiệm thêm các tổ hợp biến khác dựa trên IV/VIF hoặc các phương pháp lựa chọn đặc trưng khác.
- Ngưỡng cutoff hiện chọn theo Youden's J (tối ưu thống kê); khi triển khai thực tế nên kết hợp thêm mục tiêu kinh doanh (tỷ lệ phê duyệt mong muốn, khẩu vị rủi ro, chi phí sai phân loại theo cost matrix ở trên).

## 7. Công cụ sử dụng

Python, pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn.
