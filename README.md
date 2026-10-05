# Lab 18: Production RAG Pipeline

**K4-Track3A · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ⚠️ Tùy chọn * | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` | ✅ Có | RAGAS eval (M4), Enrichment LLM (M5), sinh câu trả lời |
| LM Studio (OpenAI-compatible server) | ⚠️ Tùy chọn * | Dùng thay OpenAI API khi chạy offline |

`*` — hai dòng này được học viên sửa lại; bản gốc của đề bài ghi Docker là **bắt buộc**
(`✅`) và `OPENAI_API_KEY` là **không bắt buộc** (`⚠️`). Xem ghi chú bên dưới.

### Ghi chú về môi trường chạy thực tế

> **Phần này do học viên bổ sung, KHÔNG phải nội dung gốc của đề bài.**
> README gốc yêu cầu chạy Docker cho Qdrant và dùng API OpenAI trả phí (`OPENAI_API_KEY`).
> Bài của tôi chạy hoàn toàn offline trên máy cá nhân với LM Studio, nên tôi ghi lại
> những điểm khác biệt ở đây để người chấm không bất ngờ. Toàn bộ 5 module vẫn giữ nguyên
> logic và cấu trúc của đề bài; các khác biệt chỉ nằm ở *nguồn cung cấp dịch vụ*.
>
> — Nguyễn Trọng Minh (02496), K4-Track3A, 05/10/2026

Bài này được chạy với **LM Studio** thay cho API OpenAI có trả phí, và **Qdrant in-memory**
thay cho Docker. Cả hai đều không làm thay đổi logic của từng module:

- **LLM qua LM Studio.** Đặt `OPENAI_API_KEY` (key của LM Studio) và `OPENAI_API_BASE`
  trong `.env`. Cần **cả hai** biến: `openai` SDK đọc `OPENAI_BASE_URL`, còn
  `langchain-openai` 0.1.x (mà RAGAS dùng bên trong) **chỉ** đọc `OPENAI_API_BASE`.
  Thiếu biến thứ hai, RAGAS sẽ gọi nhầm API OpenAI thật và fail với
  `OpenAIAuthenticationError`. RAGAS cũng cần override LLM + embedding vì mặc định của
  nó là `gpt-4o-mini` / `text-embedding-ada-002` (không có trong LM Studio).
- **RAGAS chạy tuần tự.** `RunConfig(max_workers=1)` — LM Studio phục vụ lần lượt từng
  request trên 1 GPU, còn mặc định của RAGAS là 16 worker → mọi call đều timeout và trả
  về `NaN`.
- **Embedding qua LM Studio.** M2 dense search gọi `/v1/embeddings` thay vì tải
  `BAAI/bge-m3` qua `sentence_transformers` (model đã nạp sẵn trên GPU, tiết kiệm ~2.3 GB).
- **Qdrant in-memory.** `DenseSearch` tự fallback sang `QdrantClient(":memory:")` khi
  không kết nối được Docker. Corpus chỉ ~26 KB nên không cần lưu lâu dài.

**Pre-download models** (chỉ cần model reranker, để tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3A-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3A-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3A-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **23h59 ngày diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
