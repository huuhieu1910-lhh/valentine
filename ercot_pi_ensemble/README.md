# Physics-informed interpretable ensemble (CNN + Transformer), PyTorch

Bản PyTorch (Python 3.11) của **Fig. 1: Overall architecture of the proposed physics-informed
interpretable ensemble framework**, dùng tập dữ liệu ở **Table I: ERCOT + ASOS (2018–2025)**.

## Mỗi khối trong sơ đồ nằm ở đâu trong code

| Khối trong Fig. 1 | Code |
|---|---|
| **Input Tensor X (24 × 13)**: Load, Weather, Calendar, ERCOT + ASOS 2018–2025 | `data.py`: `FEATURES` (13 kênh), `load_csv`, `prepare_data` (cửa sổ 24 giờ × 13 kênh) |
| **Branch 1, CNN**: Conv1D(k=3, c=64) → BN + ReLU → MaxPool1D(s=2) → Conv1D(k=3, c=64) → BN + ReLU → MaxPool1D(s=2) → GAP → Dense (z_CNN) → ŷ_CNN | `models.CNNBranch` |
| **Branch 2, Transformer**: Linear Projection (d=64) → Positional Encoding → 2 encoder block [Layer Norm → Multi-Head Self-Attention (h=4, d_k=16) → Layer Norm → Feed-Forward (d_ff=256)] → GAP → Dense (z_T) → ŷ_T | `models.TransformerBranch`, `EncoderBlock`, `MultiHeadSelfAttention`, `FeedForward`, `PositionalEncoding` |
| **Physics-Informed Loss**: L_θ = L_MSE + λ1·L_parabolic + λ2·L_ramp, hiệu chỉnh theo ERCOT Piecewise Parabolic Envelope, T0 = 18.5 °C | `physics.PhysicsCalibration`, `physics.PhysicsInformedLoss` |
| **Weighted Ensemble**: ŷ = Σ w_i·ŷ_i, i ∈ {CNN, T}, và mũi tên **Feedback** từ loss | `models.WeightedEnsemble`, `train.fit_ensemble_weights` |
| **SHAP DeepExplainer**: φ_i^ens = w'_CNN·φ_i^CNN + w'_T·φ_i^T | `explain.explain_ensemble`, `explain.plot_heatmap` |
| **Operator Forecast + Attribution + Regime-Flag** | `operator_report.operator_report` |

Toàn bộ siêu tham số đọc từ hình nằm trong `pi_ensemble/config.py`.

## Cài đặt

```bash
cd ercot_pi_ensemble
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy

Demo trên dữ liệu tổng hợp (có cùng cột và cùng khoảng giá trị như Table I, 70,128 giờ):

```bash
python -m pi_ensemble.run --synthetic --epochs 20
```

Chạy với dữ liệu ERCOT + ASOS thật:

```bash
python -m pi_ensemble.run --data ercot_asos_2018_2025.csv
```

Các tùy chọn hay dùng: `--horizon 24` (dự báo 24 giờ tới thay vì 1 giờ), `--lambda1`, `--lambda2`,
`--epochs`, `--max-train-windows` (lấy mẫu con để chạy nhanh), `--explain-hours` (số giờ cuối của
tập test đưa qua SHAP), `--device cuda`.

Chạy tests:

```bash
pytest -q
```

## Định dạng file CSV

Mỗi dòng là một giờ, gồm các cột:

| Cột | Ý nghĩa (Table I) |
|---|---|
| `timestamp` | thời điểm (giờ) |
| `load_mw` | Demand (MW), tải hệ thống ERCOT theo giờ |
| `temp_c` | Air temp. (°C), trung bình 3 trạm |
| `feels_like_c` | Feels-like (°C), heat-index / wind-chill |
| `humidity` | Relative humidity (%) |
| `wind_speed` | Surface wind (m/s) |
| `precip_mm` | Precipitation (mm), lượng mưa theo giờ |
| `weather_code` | W-data: số hoặc chuỗi `wxcodes` ASOS gốc (sẽ tự label-encode) |

Các cột lịch (`hour`, `weekday`, `month`, `day_of_year`, `is_weekend`, `is_holiday` theo lịch
liên bang Mỹ) được tạo tự động từ `timestamp`. Giờ trùng (DST) bị loại, khoảng trống ≤ 6 giờ được nội suy.
Chia tập theo năm: Train 2018–2022 / Val 2023 / Test 2024–2025. Chuẩn hóa z-score chỉ dùng
thống kê của tập train.

## Những chỗ sơ đồ không ghi rõ và cách đã chọn

1. **13 kênh đầu vào.** Table I chỉ liệt kê 12 biến (1 load + 6 thời tiết + hour, weekday, month,
   weekend, holiday) nhưng Fig. 1 ghi X (24 × 13). Kênh thứ 13 được chọn là `day_of_year`. Nếu dữ liệu
   của bạn dùng kênh khác thì chỉ cần sửa `FEATURES` trong `config.py`; `n_features` tự cập nhật theo.
2. **Mục tiêu dự báo.** Cửa sổ là 24 giờ t−23…t, mục tiêu là tải ở `horizon` giờ tiếp theo
   (mặc định 1).
3. **CNN.** Conv dùng padding "same" nên chiều thời gian đi 24 → 12 → 6. MaxPool dùng kernel 2, stride 2.
   Dense layer có 64 unit + ReLU (tạo ra z), sau đó là một lớp Linear cho ŷ.
4. **Encoder block.** Đúng thứ tự trong hình: LayerNorm → MHSA → LayerNorm → FFN (kiểu pre-LN), có
   residual connection như Transformer chuẩn. FFN dùng ReLU vì SHAP DeepExplainer có luật DeepLIFT cho ReLU.
   Dropout 0.1. Positional encoding dạng sinusoidal.
5. **L_parabolic.** Bao parabol từng đoạn có đỉnh cố định T0 = 18.5 °C:
   P(T) = a0 + a_heat·(T0 − T)₊² + a_cool·(T − T0)₊², hai nhánh độ cong riêng (≥ 0). Biên dưới và
   biên trên được fit trên các quantile 0.5 % / 99.5 % của tải trong từng bin 1 °C của tập train, nới
   thêm 5 %, rồi nhân với xu hướng tăng trưởng tải hằng năm (log-linear) để không phạt nhầm các năm
   2024–2025 có tải cao hơn. Loss chỉ phạt phần dự báo nằm ngoài bao:
   mean(ReLU(lower − ŷ)² + ReLU(ŷ − upper)²).
6. **L_ramp.** Độ dốc ŷ_{t+1} − y_t (và giữa các bước liên tiếp khi horizon > 1) không được vượt giới hạn
   ramp của giờ trong ngày đó. Giới hạn là quantile 99.9 % của |ΔP| theo từng giờ trên tập train, nới
   thêm 10 %. Loss: mean(ReLU(|Δŷ| − R_max(hour))²).
7. **Huấn luyện và Feedback.** Bước 1: mỗi nhánh tối ưu L_θ riêng (AdamW, gradient clipping,
   early stopping theo L_θ trên Val 2023). Bước 2 (mũi tên Feedback): đóng băng hai nhánh, học
   w = softmax(α) bằng cách tối ưu L_θ của ŷ ensemble trên Val, nên w_i ≥ 0 và Σw_i = 1. Đây chính
   là w' trong công thức SHAP.
8. **SHAP.** `shap.DeepExplainer` chạy riêng cho từng nhánh với cùng background (mẫu từ tập train);
   vì ensemble tuyến tính nên φ^ens = w'_CNN·φ^CNN + w'_T·φ^T. Với nhánh CNN kết quả chính xác
   (tổng φ bằng f(x) − E[f]). Với nhánh Transformer, LayerNorm, softmax và các phép nhân trong attention
   không có luật DeepLIFT trong SHAP nên dùng gradient thay thế; kết quả là xấp xỉ. Độ lệch
   max|f(x) − E[f] − Σφ| của từng mô hình được ghi vào `summary.json` (`shap_completeness_gap_mw`).
9. **Regime-Flag.** `regime` = heating / neutral / cooling tùy nhiệt độ giờ mục tiêu nằm dưới, trong
   hay trên T0 ± 2 °C. `flags` = `ENVELOPE` (dự báo ra ngoài bao parabol), `RAMP` (ramp vượt giới hạn),
   `EXTREME_TEMP` (nhiệt độ ngoài khoảng 1–99 % của tập train), hoặc `OK`. Khi vận hành thật, nhiệt độ
   giờ mục tiêu lấy từ dự báo thời tiết.

## Kết quả đầu ra (`--out`, mặc định `outputs/`)

| File | Nội dung |
|---|---|
| `model.pt` | trọng số hai nhánh và ensemble, cấu hình, hệ số hiệu chỉnh vật lý, thống kê chuẩn hóa (`run.load_checkpoint` để nạp lại) |
| `summary.json` | trọng số ensemble, MAE / RMSE / MAPE và tỉ lệ vi phạm vật lý trên Test của từng nhánh và ensemble, độ lệch SHAP, lịch sử huấn luyện |
| `operator_report.csv` | mỗi giờ: dự báo ensemble / CNN / Transformer, thực tế, biên dưới/trên của bao, regime, flags, baseline SHAP và 3 yếu tố đóng góp lớn nhất (MW) |
| `shap_importance_mw.csv` | mean \|φ\| của từng kênh (MW) cho CNN, Transformer, ensemble |
| `shap_heatmap.png` | heatmap thời gian × kênh của mean \|φ^ens\| (ô SHAP trong Fig. 1) |

Dữ liệu tổng hợp (`--synthetic`) chỉ để chạy thử pipeline; số liệu từ nó không phản ánh ERCOT thật.
