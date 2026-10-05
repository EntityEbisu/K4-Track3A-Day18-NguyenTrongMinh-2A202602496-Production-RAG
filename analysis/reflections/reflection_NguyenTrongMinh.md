# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Trọng Minh (02496)
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 05/10/2026
**Mô hình sinh câu trả lời:** `ternary-bonsai-8b` (Q2_K, qua LM Studio — xem README để biết lý do dùng server local thay API trả phí)

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Threshold 0.85 trên corpus 26 tài liệu cho **208 chunks** (avg 99 ký tự) so với basic **51 chunks** (avg 410 ký tự). Chunk nhiều hơn 4× và nhỏ hơn 4× vì MiniLM đặt ngưỡng similarity cao nên hầu như mọi câu đều bị tách. **Nhận xét:** threshold 0.85 quá cao cho corpus tiếng Việt ngắn — chunk semantic ở đây *tệ hơn* basic về mặt ngữ nghĩa. Có chunk chỉ 6 ký tự (min_len=6) → gần như vô dụng. Bài học: threshold phải calibrate theo corpus, không copy từ slide. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | **97 children** từ **11 parents** (parent 2048, child 256 ký tự). Đây là chiến lược được dùng thật trong `src/pipeline.py`. Ưu điểm: truy xuất theo child (nhỏ, chính xác) nhưng trả về parent cho LLM (đủ ngữ cảnh). Nhược điểm lộ ra ở failure #3: child 256 ký tự **cắt đứt bảng** phê duyệt theo khoảng ngày, mất mốc "16-30 ngày: CEO". |
| **Structure-aware chunking** | M1 | `chunk_structure_aware()` | **106 chunks** (avg 197 ký tự), giữ nguyên header trong `metadata["section"]`. Ít nhất theo file (`nghi_phep_khong_luong.md`) thì hợp hơn hierarchical vì không cắt giữa bảng — đây chính là fix được đề xuất cho failure #3. |
| **BM25 + Dense fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | `score(d) = Σ 1/(k + rank + 1)`, k=60. Kết quả đo được: **Context Recall 0.7833 → 0.8333 (+0.05)** so với baseline dense-only. RRF giải quyết đúng mâu thuẫn điểm số giữa BM25 (lexical, bắt đúng "120 ngày") và dense (semantic, hiểu "nghỉ phép"). Điểm yếu thấy rõ ở câu hỏi multi-hop: BM25 + dense cùng trả về *cả hai* phiên bản v2023 và v2024 mà không phân biệt cái nào còn hiệu lực. |
| **Vietnamese segmentation** | M2 | `segment_vietnamese()` | Bắt buộc `.replace("_", " ")`. Không có bước này, underthesea tạo token `nghỉ_phép` trong corpus nhưng query "nghỉ phép" tách thành 2 token → **BM25 không khớp**, điểm BM25 về 0. Đây là chi tiết nhỏ nhưng quyết định toàn bộ hiệu quả của M2 trên tiếng Việt. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3, 20 candidates → top 3. **Latency 1574 ms** (avg, 3 lần đo). Chunk "nghỉ phép năm" được chấm **+0.9981** (retrieval score chỉ 0.81), chunk "nghỉ phép không lương" +0.8964 — reranker phân biệt được mức độ liên quan mà dense search gộp chung. Chi phí ~1.6 s/câu là chấp nhận được vì nó chạy local, không tốn API. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | faithfulness 0.5083 · answer_relevancy 0.7107 · context_precision 0.9125 · context_recall 0.8333. **3/4 metric ≥ 0.70.** Metric thấp nhất là Faithfulness — và phần lớn nguyên nhân **không nằm ở RAG** mà ở prompt + giới hạn model (xem Phần 2). Bài học lớn nhất của lab: **metric thấp ≠ pipeline sai**, phải lần từng bước Error Tree trước khi kết luận. |
| **Contextual prepend** | M5 | `contextual_prepend()` / `_enrich_single_call()` | Chọn **combined mode**: 1 LLM call/chunk thay vì 4 → **96 chunks = 96 calls** (thay vì 384). Kiểm chứng an toàn: raw chunk được giữ **nguyên văn** trong `enriched_text` (0/96 chunk bị LLM viết lại) — enrichment chỉ *tiền* một câu mô tả vị trí, không sửa số liệu. Anthropic benchmark ghi giảm 49% retrieval failure; ở lab này chưa đo được cải thiện rõ vì bottleneck nằm ở model sinh. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 2.1 RAGAS trả về toàn `NaN` — nguyên nhân kép

**Exact error:**
```
Exception raised in Job[1]: BadRequestError(Error code: 400 - {'error': "'input' field must be a string or an array of strings"})
→ aggregate: {'faithfulness': nan, 'answer_relevancy': nan, 'context_precision': nan, 'context_recall': nan}
```

**Nguyên nhân gốc & quá trình debug — có 2 lỗi độc lập:**

1. **`OpenAIEmbeddings` mặc định tokenize trước khi gửi.** Với `check_embedding_ctx_length=True` (mặc định), `langchain_openai` gọi `_get_len_safe_embeddings()` rồi gửi **mảng int token** lên `/v1/embeddings`. LM Studio chỉ nhận string → HTTP 400. Đọc source `langchain_openai/embeddings/base.py:491` để xác nhận. Fix: `check_embedding_ctx_length=False`.

2. **`RunConfig.max_workers` mặc định = 16.** LM Studio phục vụ **tuần tự** trên 1 GPU, nên 16 request đồng thời → mọi call đều timeout. Fix: `RunConfig(max_workers=1, timeout=300)`.

3. **Bug ẩn thứ ba (mới phát hiện khi đọc lại code):** `float(x) or 0.0` **không** biến `NaN` thành `0.0`, vì `NaN` là truthy trong Python:
   ```python
   >>> float('nan') or 0.0
   nan
   ```
   Nên chỉ **một** câu hỏi hỏng cũng kéo toàn bộ aggregate về `NaN`. Fix: thêm `_finite()` dùng `math.isfinite()`.
   → Đây là bài học: fix lỗi tầng dưới chưa chắc lỗi tầng trên đã hết, phải verify lại output thực tế.

### 2.2 Ba lỗi qdrant-client chỉ lộ ra ở chế độ in-memory

Docker không chạy nên `QdrantClient(":memory:")` được dùng, và bản local expose 3 lỗi mà bản remote không có:

| Lỗi | Message | Fix |
|---|---|---|
| `collection_name` phải positional | `TypeError: query_points() missing 1 required positional argument: 'collection_name'` | `query_points(collection, query=...)` |
| list bị hiểu là mảng 2D | `ValueError: Multivector  is not found in the collection` | bọc `models.NearestQuery(nearest=...)` |
| client không thật sự là local | chỉ `UserWarning: Failed to obtain server version` | verify `get_collections()` chạy được + `check_compatibility=False` |

Ngoài ra đổi `Distance.COSINE` → `Distance.DOT` vì vector đã L2-normalize (tương đương cosine nhưng nhanh hơn).

**Bài học:** `tests/test_m2.py` **không hề** khởi tạo `DenseSearch` — 5/5 test pass mà dense search chưa chạy bao giờ. Test pass ≠ code path đã được kiểm chứng. Phải tự gọi tay.

### 2.3 Faithfulness giảm so với baseline — nghi vấn sai, kết luận đúng

Ban đầu tôi nghi ngờ enrichment làm hỏng corpus. Đã **loại trừ bằng dữ liệu**: raw chunk giữ nguyên văn 0/96, số duy nhất LLM thêm (`2024.`) đến từ tên file. Nghi vấn thứ hai là context window 8k — cũng **loại trừ**: prompt thực tế chỉ ~485 token, dùng 6% budget.

Kết luận thật, có A/B:
| System prompt | Bị từ chối | Đáp án đúng |
|---|---|---|
| Scaffold gốc | **3/5** | 2/5 |
| Bỏ lối thoát "Nếu không có" | **0/5** | 4/5 |

Cùng model, cùng context, cùng temperature → lỗi do **prompt**, không phải RAG. Failure #4 là ví dụ điển hình: context chứa nguyên văn *"chưa được hưởng gói bảo hiểm sức khỏe PVI"* nhưng model vẫn trả "Không tìm thấy."

### 2.4 Giới hạn của mô hình 2-bit

`ternary-bonsai-8b` (Q2_0, 2.3 GB) **không thực sự làm phép tính**: 4/10 câu hỏi cần nhân/chia (2% × 15.000.000; 85% × 20.000.000; 100% × 25.000.000). Model đọc đúng tỉ lệ rồi bịa con số (`1,5%`, `50.000.000`) hoặc từ chối trả lời. Vì 3 metric RAGAS dùng LLM làm judge, judge yếu kéo điểm xuống theo **cả hai chiều** — Faithfulness 0.5083 phản ánh giới hạn của judge chứ không chỉ của người trả lời.

### 2.5 Lỗi công cụ trong quá trình làm bài

- `patch` tool **tự thêm indentation của vùng match vào mọi dòng mới** → code Python bị indent kép, `IndentationError`. Mất thời gian sửa 3 lần trên cùng một file. Sau đó `git checkout -- src/m2_search.py` để khôi phục sạch — nhưng hàm `segment_vietnamese()` sửa trước đó **bị mất luôn** vì chưa commit. Phải làm lại từ đầu.
- `response_format={"type":"json_object"}` bị LM Studio **từ chối** (HTTP 400: `'response_format.type' must be 'json_schema' or 'text'`), trong khi scaffold gợi ý `json.loads()` thẳng. Model còn hay bọc JSON trong markdown fence → phải parse thủ công có regex fallback.

### Kiến thức còn thiếu & cách bổ sung

| Thiếu | Bổ sung bằng cách |
|---|---|
| Hiểu vì sao RRF chỉ cần *rank*, không cần *score* | Đọc paper Cormack et al. 2009; hiểu RRF chống lại chênh lệch thang điểm giữa BM25 (không chuẩn hoá) và cosine (0-1) |
| Biết LM Studio chỉ chấp nhận `json_schema`/`text` | Đọc response lỗi 400 và chấp nhận giới hạn của OpenAI-compatible server, không phải OpenAI thật |
| Phân biệt lỗi retrieval với lỗi generation | Học cách dựng Error Tree và **kiểm chứng giả thuyết bằng A/B** thay vì suy đoán |

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý tra cứu quy định nội bộ (HR/Policy) cho VinUni

#### 1. Hiện trạng
- **Pipeline hiện tại:** Ở lab này đã dựng xong: chunking phân cấp → hybrid BM25+Dense (RRF) → rerank 20→3 → sinh câu trả lời → đánh giá RAGAS. Chạy được end-to-end offline.
- **Bottlenecks đã đo được:**
  - Faithfulness 0.5083 — yếu do prompt có lối thoát + model 2-bit không tính được.
  - 3 câu hỏi "nghỉ phép năm / mật khẩu" trả lời sai vì corpus có **hai phiên bản chính sách** (v2023 vs v2024) mà không có trường `superseded_by` nào đánh dấu cái nào còn hiệu lực.
  - Chunk 256 ký tự cắt đứt bảng phê duyệt theo khoảng (failure #3).
  - Toàn bộ pipeline mất **60.6 phút** cho 20 câu, trong đó ~2/3 là thời gian chờ RAGAS gọi LLM tuần tự.

#### 2. Kế hoạch cải thiện
1. **Chunking:** chuyển sang `chunk_structure_aware()` cho tài liệu dạng bảng/quy trình, giữ `chunk_hierarchical()` cho văn bản dài. Thêm metadata `effective_date` + `superseded_by` để giải quyết mâu thuẫn phiên bản.
2. **Search:** giữ **Hybrid + RRF** (đã chứng minh Context Recall +0.05). Bổ sung **metadata filter** lọc theo `effective_date <= today` để loại chính sách hết hiệu lực ngay từ đầu, thay vì để LLM tự suy luận.
3. **Reranking:** giữ cross-encoder (đã cho +0.9981 vs 0.81 retrieval score). Nếu cần tốc độ, dùng `FlashrankReranker` cho tier-1 và cross-encoder chỉ khi rerank top-10.
4. **Evaluation:** giữ **RAGAS 4 metrics** làm chỉ số chính, nhưng thêm **deterministic checks** để không phụ thuộc LLM judge: so khớp con số trong câu trả lời với con số trong ground truth. Bổ sung test-set riêng cho câu hỏi đòi hỏi phép tính.
5. **Enrichment:** giữ **combined mode (1 call/chunk)**. Bỏ phần sinh `questions`/`summary` nếu thấy retrieval không dùng đến — chỉ cần `contextual_prepend`, cắt ~60% chi phí enrichment.

#### 3. Timeline triển khai
- **Tuần 1:** Sửa prompt sinh câu trả lời (bỏ lối thoát "Nếu không có"), thêm metadata `effective_date`/`superseded_by` cho corpus. Đo lại RAGAS.
- **Tuần 2:** Chuyển các tài liệu dạng bảng sang `chunk_structure_aware()`. Thêm bộ test 10 câu hỏi phép tính để đo riêng khả năng số học.
- **Tuần 3:** Thêm calculator tool — chuyển mọi phép tính (%, phạt, hoàn trả) sang code Python thay vì để LLM tự tính.
- **Tuần 4:** Cache kết quả RAGAS theo (model, prompt version, corpus hash) để A/B prompt không tốn 30 phút mỗi lần. Đóng gói thành API nội bộ.

#### 4. Rủi ro cần lưu ý
- **Đổi mô hình làm đổi baseline.** Chuyển từ `ternary-bonsai-8b` (2-bit) sang model mạnh hơn sẽ làm Faithfulness tăng rõ — nhưng mọi so sánh A/B sau đó phải ghi rõ dùng model nào, nếu không số liệu trở nên vô nghĩa.
- **Không được "sửa" số liệu RAGAS cho đẹp.** Điểm thấp là thông tin thật về giới hạn hệ thống; che giấu nó thì loss analysis mất tác dụng.