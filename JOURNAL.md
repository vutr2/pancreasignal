# Journal

## 2026-11-10: Phase 2, `features.py`

**Bài học chính: test pass chưa chắc code đúng.** 4 test cũ đều pass, nhưng code có 3 bug. Lý do là test cũ
chỉ kiểm tra *tên cột, số hàng, không có NaN*, còn **giá trị** bên trong thì không kiểm tra.

### Bug 1: `build_features` không làm gì cả (nặng nhất)
```python
if cfg.get('normalize_by_creatinine'):     # sai
if features['normalize_by_creatinine']:    # đúng
```
- Config có dạng `cfg = {"features": {"normalize_by_creatinine": True, ...}, "model": ...}`. Hai cờ này
  nằm **bên trong** `cfg["features"]`, không nằm ở cấp ngoài cùng.
- `cfg.get("normalize_by_creatinine")` không tìm thấy key nên trả về `None`, và `if None:` là False. Hai bước
  bị bỏ qua mà **không có lỗi nào hiện ra**, nên model sẽ học trên giá trị thô.
- **Rút ra:** với key bắt buộc phải có, dùng `d["key"]` thay vì `d.get("key")`. Nếu thiếu key hoặc gõ sai
  tên, `d["key"]` báo `KeyError` ngay. `.get()` chỉ hợp với key *tuỳ chọn* có giá trị mặc định,
  ví dụ `m.get("calibrate")`.

### Bug 2: chọn cột quá sớm nên mất cột creatinine
```python
result = df[biomarkers + demographics]                   # creatinine bị bỏ mất
result = normalize_by_creatinine(result, biomarkers)     # KeyError: 'creatinine'
```
- Bug này bị bug 1 che: vì bước chia không bao giờ chạy nên chưa ai thấy lỗi.
- Cách sửa: giữ thêm `creatinine` lúc chọn cột, rồi ở dòng `return` cuối hàm chỉ trả về cột feature.
- **Thứ tự quan trọng:** chia trước, log sau, tức là `log1p(LYVE1 / creatinine)`. Nếu làm ngược lại sẽ
  ra một feature khác hẳn.

### Bug 3: `log_transform` bỏ qua tham số `columns`
```python
col_num = df.select_dtypes(include=['number']).columns    # sai: lấy MỌI cột số
result[columns] = np.log1p(result[columns])              # đúng: chỉ cột được yêu cầu
```
- Bản cũ log cả `age` (76 thành 4.34) và `creatinine`. Test cũ chỉ kiểm tra cột `TFF1` nên không phát hiện.
- **Rút ra:** hàm phải làm đúng những gì tham số yêu cầu. Nếu một tham số không được dùng tới trong
  thân hàm thì gần như chắc chắn đó là bug.

### Sửa thêm: `encode_sex` báo lỗi khi gặp giá trị lạ
- `.map({"M": 1, "F": 0})` biến `"m"`, `"Male"` hay NaN thành NaN **mà không báo gì**. Sang Phase 6, API sẽ
  nhận dữ liệu người dùng nhập tay, nên giờ hàm báo `ValueError` ngay khi gặp giá trị lạ.

### Test mới thêm vào `tests/test_features.py`
| Test | Bắt bug |
|---|---|
| `test_build_features_values`: so **giá trị** với `log1p(x / creatinine)` | 1, 2 |
| `test_log_transform_only_given_columns`: `age`, `creatinine` không bị đổi | 3 |
| `test_encode_sex_rejects_unknown`: `"m"` phải báo lỗi | encode_sex |
| `test_build_features_flags_off`: tắt cờ thì giữ nguyên giá trị thô | cờ config |

Đã kiểm tra: chạy các test mới trên code cũ thì 3 test fail, trên code mới thì pass hết.
