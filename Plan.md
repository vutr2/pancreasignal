# PancreaSignal: Kế hoạch từng bước

> **Nguyên tắc số 1:** bạn tự viết mọi dòng code có đánh dấu `⚠️ YOUR CODE`. Phần khung (config, load data, runner, tests) đã chạy được. Mỗi phase kết thúc khi **test của phase đó pass**.
>
> **Môi trường:** mọi lệnh trong file này giả định đã `source .venv/bin/activate`. Cài thư viện mới: `pip install <tên>` **và** thêm tên đó vào `dependencies` trong `pyproject.toml`.
>
> **Nguyên tắc số 2:** chỉ ghi lên CV những con số có trong `reports/*_metrics.json` do bạn tự chạy ra.

Phiên bản v1 không cần MIMIC. Phần MIMIC/CORAL để sang v2, sau khi có reference và được PhysioNet duyệt.

---

## Tổng quan

| Phase | Nội dung | Thời gian | Done khi |
|---|---|---|---|
| 0 | Setup môi trường + repo | 0.5 ngày | `pytest` chạy, 6 test pass |
| 1 | Tải data + EDA | 2–3 ngày | Notebook 01 trả lời đủ 6 câu hỏi |
| 2 | Features + Logistic Regression | 2–3 ngày | `test_features.py` + `test_logreg` pass |
| 3 | Evaluation đúng chuẩn (LOCO, CI, sens@spec) | 3–4 ngày | **Toàn bộ test pass** + chạy thật lần đầu |
| 4 | XGBoost, calibration, SHAP, model card | 1 tuần | Bảng so sánh 3 model × 3 task trong README |
| 5 | Model miRNA (GEO) | 1 tuần | Có model thứ 2, cùng evaluation |
| 6 | FastAPI `/predict` | 3–4 ngày | `curl` trả về risk + giải thích |
| 7 | Intake agent (đọc phiếu xét nghiệm) | 1 tuần | F1 trích xuất ≥ mục tiêu, 0 lỗi đổi đơn vị |
| 8 | Guideline RAG | 1 tuần | Faithfulness đo được trên bộ câu hỏi |
| 9 | LangGraph orchestration + critic + human review | 1 tuần | Chạy demo end-to-end |
| 10 | Eval CI, UI, write-up | 1 tuần | Repo public, README có kết quả + demo |

**Phase 0–4 là "model hoàn chỉnh"** (khoảng 3 tuần). Phase 5–10 biến dự án từ model thành hệ thống AI engineering.

---

## Phase 0: Setup (0.5 ngày)

- [ ] Cài công cụ: `brew install python@3.12 libomp git`
- [ ] Giải nén skeleton, `cd pancreasignal`
- [ ] Tạo môi trường: `python3.12 -m venv .venv`
- [ ] Kích hoạt: `source .venv/bin/activate` (**mỗi lần mở Terminal mới đều phải chạy lại**; thấy `(.venv)` ở đầu dòng là đúng)
- [ ] Cài thư viện: `pip install --upgrade pip && pip install -e ".[dev,notebook,tracking]"`
- [ ] Chốt phiên bản: `pip freeze --exclude-editable > requirements-lock.txt`, rồi commit file này (để tái lập kết quả)
- [ ] `git init && git add -A && git commit -m "Initial skeleton"`
- [ ] Tạo repo **private** trên GitHub `vutr2/pancreasignal`, rồi push. Public sau khi xong Phase 4.
- [ ] `nbstripout --install` (xoá output notebook mỗi lần commit)
- [ ] `pytest`: kỳ vọng **6 passed, 12 failed** (12 test fail là các phần bạn sẽ viết)

**Done:** 6 test trong `test_data.py` pass.

---

## Phase 1: Data + EDA (2–3 ngày)

- [ ] Lấy Kaggle API token (Kaggle → Settings → API → Create New Token), rồi chạy:
      `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
- [ ] `python scripts/download_data.py --only urine`
- [ ] Mở `data/raw/urine/`, kiểm tra **tên file CSV thật**, sửa `data.path` trong `configs/urine.yaml` nếu khác
- [ ] Chạy thử: `python -c "from pancreasignal.data.urine import load_urine; print(load_urine().shape)"`
      Nếu `validate()` báo lỗi, đọc kỹ thông báo. Đó là dữ liệu thật đang khác giả định; sửa giả định cho đúng, đừng xoá phần kiểm tra.
- [ ] Làm `notebooks/01_eda.ipynb`: trả lời Q1–Q6 bằng biểu đồ + 1–2 câu nhận xét mỗi câu
- [ ] Commit: `"EDA: urine dataset"`

**Done:** trả lời được không cần nhìn: có bao nhiêu PDAC, phân bố theo center ra sao, CA19-9 thiếu bao nhiêu, có bao nhiêu ca stage I–II.

**Câu phỏng vấn cần trả lời được:** *"Dataset này có bias gì?"* Gợi ý: đây là dữ liệu case-control (chọn bệnh nhân đã biết chẩn đoán), không phải dân số sàng lọc; tuổi các nhóm khác nhau; CA19-9 bị thiếu không ngẫu nhiên.

---

## Phase 2: Features + Logistic Regression (2–3 ngày)

File: `src/pancreasignal/features.py`, `src/pancreasignal/models/__init__.py`

- [ ] `normalize_by_creatinine`: chạy `pytest tests/test_features.py::test_normalize_by_creatinine`
- [ ] `log_transform`, `encode_sex`
- [ ] `build_features`: chú ý thứ tự các bước và **không dùng thống kê của toàn bộ dataset**
- [ ] `build_logreg`: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])`
- [ ] `pytest tests/test_features.py tests/test_models_and_pipeline.py::test_logreg`
- [ ] Commit sau mỗi hàm pass test

**Done:** 4 test features + `test_logreg` pass.

**Học:** vì sao `StandardScaler` phải nằm **trong** Pipeline. Nếu fit scaler trên toàn bộ data rồi mới chia train/test, thông tin từ tập test sẽ rò sang lúc train (data leakage).

---

## Phase 3: Evaluation đúng chuẩn (3–4 ngày): quan trọng nhất

File: `src/pancreasignal/evaluation/splits.py`, `src/pancreasignal/evaluation/metrics.py`

- [ ] `leave_one_center_out` (`tests/test_splits.py`)
- [ ] `sensitivity_at_specificity` (2 test)
- [ ] `bootstrap_auroc_ci`
- [ ] `pytest`: **tất cả pass trừ `test_xgboost` và `test_ece`** (thuộc Phase 4)
- [ ] **Chạy thật lần đầu:**
  ```bash
  python -m pancreasignal.train --task pdac_vs_control
  python -m pancreasignal.train --task pdac_vs_benign
  python -m pancreasignal.train --task pdac_vs_rest
  ```
- [ ] Ghi kết quả vào `RESULTS.md`: AUROC (CI), sens@90/95, từng fold, so sánh với CA19-9 trên cùng các hàng
- [ ] Commit: `"LOCO evaluation + first real results"`

**Done:** có 3 file `reports/*_metrics.json` thật.

**Tự kiểm tra:**
- AUROC của `pdac_vs_control` **phải cao hơn** `pdac_vs_benign`. Nếu ngược lại, có bug.
- AUROC > 0.97 → nghi ngờ leakage trước khi vui mừng.
- Fold nào AUROC thấp hẳn? Ghi lại lý do có thể (center khác, ít mẫu).

**Câu phỏng vấn:** *"Vì sao không dùng random K-fold?"* và *"Sensitivity ở 95% specificity nghĩa là gì với bệnh nhân?"*

---

## Phase 4: Model tốt hơn + giải thích (1 tuần)

- [ ] `build_xgboost`, rồi `python -m pancreasignal.train --model xgboost --task pdac_vs_benign`
- [ ] Thêm **baseline chỉ dùng tuổi**: nếu nó mạnh gần bằng model biomarker thì model đang dựa vào tuổi để "ăn gian"
- [ ] `expected_calibration_error` + calibration (`calibrate: true` → `CalibratedClassifierCV(method="isotonic")`)
- [ ] Vẽ calibration curve + ROC curve từ `_oof.csv` (`notebooks/02_results.ipynb`)
- [ ] **Sensitivity theo stage** (I–II vs III–IV) từ cột `stage` trong `_oof.csv`
- [ ] SHAP cho XGBoost: biomarker nào quan trọng nhất? Có khớp với paper gốc không?
- [ ] Viết `MODEL_CARD.md`: mục đích, dữ liệu, kết quả, giới hạn, không dùng cho lâm sàng
- [ ] Bảng 3 model × 3 task trong README
- [ ] (Tuỳ chọn) MLflow UI: `mlflow ui` để so sánh các lần chạy
- [ ] **Chuyển repo sang public**

**Done:** `pytest` → **18 passed**. README có bảng kết quả + 2 biểu đồ + model card.

🎯 **Mốc CV đầu tiên:** đến đây đã đủ một dự án ML hoàn chỉnh để đưa lên CV.

---

## Phase 5: Model miRNA (1 tuần)

- [ ] `python scripts/download_data.py --only mirna_gse59856`
- [ ] Viết `src/pancreasignal/data/geo.py`: parse series matrix (`GEOparse` hoặc tự đọc), lấy nhãn PDAC / biliary / control
- [ ] Feature selection **bên trong** CV (elastic-net logistic regression); dữ liệu có hàng nghìn miRNA nhưng chỉ vài trăm mẫu
- [ ] Dùng lại `compute_metrics`, so sánh với urine model
- [ ] Viết `configs/mirna.yaml`

**Câu phỏng vấn:** *"p >> n thì làm sao tránh overfitting?"* (p là số feature, n là số mẫu)

---

## Phase 6: FastAPI (3–4 ngày)

- [ ] Lưu model đã train: `joblib.dump` → `models/urine_v1.joblib` (không commit, ghi lại SHA)
- [ ] `src/pancreasignal/api/app.py`: `POST /predict`
  - input (Pydantic): giá trị biomarker **kèm đơn vị**, tuổi, giới
  - đổi đơn vị về chuẩn của dataset (mg/mmol creatinine…) và có **unit test riêng cho đổi đơn vị**
  - output: xác suất, top-3 lý do (SHAP), cờ "ngoài phân bố" nếu giá trị nằm ngoài khoảng của dữ liệu train
- [ ] `uvicorn pancreasignal.api.app:app --reload`, test bằng `curl`
- [ ] `Dockerfile`

---

## Phase 7: Intake agent: đọc phiếu xét nghiệm (1 tuần): ⏭️ BỎ QUA (2026-11-11)

> **Lý do bỏ qua:** không bắt buộc cho các phase sau. MedGemma không gọi được qua Hugging Face Inference Providers,
> và gọi model đọc ảnh qua API thì tốn credit (tài khoản trả về 402). Các marker LYVE1/REG1B/TFF1 cũng không có
> trên phiếu xét nghiệm thật, nên phiếu nào cũng phải tự tạo.
> **Ảnh hưởng:** Phase 9 nhận input **có cấu trúc (JSON kèm đơn vị)**, đúng như API Phase 6 đã nhận. Phần
> "0 lỗi đổi đơn vị" đã được đảm bảo ở mức API bằng `tests/test_units.py`.
> Có thể quay lại làm sau (danh sách gốc ở dưới).

- [ ] `brew install ollama` → pull MedGemma 1.5 4B bản GGUF
- [ ] Tạo **tập test 50–100 phiếu xét nghiệm**: dùng mẫu phiếu thật (đã xoá thông tin cá nhân) điền giá trị từ dataset. Ghi rõ trong README là ảnh tổng hợp, chỉ dùng để test
- [ ] Prompt + Pydantic schema → JSON `{test_name, value, unit, reference_range}`
- [ ] Map tên xét nghiệm → LOINC, chuẩn hoá đơn vị
- [ ] Đo: F1 từng trường, **số lỗi đổi đơn vị (mục tiêu 0)**, latency

---

## Phase 8: Guideline RAG (1 tuần)

- [x] Corpus: **20 bài có license mở** (`configs/rag_corpus.yaml`): CA19-9, sàng lọc nhóm nguy cơ cao, IPMN. **CAPS 2020 và Kyoto 2024 bị loại**: CAPS không có license, Kyoto không có trên PMC.
- [x] ~~Postgres + pgvector (Docker)~~ **LanceDB** (không cần Docker hay server): vector + BM25 + hybrid (RRF). ~~MedCPT~~ → embedding **`nvidia/nemotron-3-embed-1b`** (không phải cài PyTorch). ~~Reranker~~ → các endpoint reranker của NVIDIA đã ngừng hoạt động (410), nên dùng RRF của hybrid thay thế
- [x] LLM qua **NVIDIA API**: trả lời bằng `nemotron-3.5-lightning` (tắt chế độ suy nghĩ), chấm điểm bằng `glm-5.3-flash` (họ model khác). Mọi lần gọi đều có cache. Đã gọi thử từng model: phần lớn model trong danh mục trả 404 hoặc treo
- [x] Bắt buộc trích dẫn (trả lời bằng JSON, mỗi câu kèm nhãn nguồn) + **bước kiểm chứng** từng câu với nguồn của chính nó; từ chối khi không đủ bằng chứng. Kiểm tra trong code, không chỉ dựa vào prompt
- [x] Bộ 50 câu hỏi (`evals/rag_questions.yaml`: 40 câu trả lời được, 10 câu ngoài corpus) → hit@k, MRR, faithfulness, tỉ lệ từ chối. Kết quả: RESULTS.md mục 9

---

## Phase 9: LangGraph orchestration (1 tuần)

- [ ] Graph: `intake → risk_model (gọi API Phase 6) → critic → rag → human_review (interrupt) → report`
- [ ] Critic kiểm tra: vàng da/bilirubin cao (CA19-9 dương tính giả), creatinine bất thường, thiếu biomarker, giá trị ngoài phân bố → **abstain** (từ chối kết luận)
- [ ] Langfuse (self-host bằng Docker) để trace từng bước, đo cost và latency
- [ ] Báo cáo song ngữ VI/EN

---

## Phase 10: Eval CI + hoàn thiện (1 tuần)

- [ ] `evals/`: tập test cho từng agent (extraction, RAG, critic, adversarial)
- [ ] GitHub Actions: chạy `pytest` + eval; **chặn merge nếu metric tụt**
- [ ] UI đơn giản (Next.js hoặc Streamlit) + video demo 2 phút
- [ ] Blog kỹ thuật: vấn đề, dữ liệu, cách đánh giá, kết quả, giới hạn
- [ ] Cập nhật CV bằng số thật

---

## v2 (khi có quyền PhysioNet)

- [ ] MIMIC-IV `hosp`: model xét nghiệm thường quy (tiểu đường mới khởi phát, sụt cân, bilirubin…)
- [ ] CORAL: tập test chuẩn cho trích xuất ghi chú ung thư tụy
- [ ] Chỉ xử lý trên Mac, chỉ dùng LLM local

---

## Quy tắc làm việc hằng ngày

1. Mở `Plan.md`, chọn **1 checkbox**
2. Chạy test của nó → thấy fail → viết code → test pass
3. `git commit` với message rõ ràng
4. Ghi 1–2 dòng vào `JOURNAL.md`: hôm nay học được gì, bug gì
5. Kẹt quá 1 giờ → hỏi (mang theo thông báo lỗi + đoạn code)
