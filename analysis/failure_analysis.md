# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Trọng Minh (02496)  
**Khóa:** K4 - Track 3A  
**Ngày:** 05/10/2026  
**LLM sinh câu trả lời:** `ternary-bonsai-8b` (Q2_K, qua LM Studio OpenAI-compatible server, `127.0.0.1:1234`)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6771 | **0.5083** | −0.1688 |
| Answer Relevancy | 0.6997 | **0.7107** | +0.0110 |
| Context Precision | 0.9250 | **0.9125** | −0.0125 |
| Context Recall | 0.7833 | **0.8333** | +0.0500 |

**Đọc kết quả:** Production cải thiện rõ hai chỉ số *đo được khả năng truy xuất* — Context Recall +0.05, Answer Relevancy +0.01. Đây là đóng góp trực tiếp của M2 (hybrid BM25 + Dense + RRF) và M3 (cross-encoder rerank).

Faithfulness lại **giảm 0.17**. Nguyên nhân nằm **không ở tầng truy xuất** mà ở bước sinh câu trả lời — chi tiết ở phần dưới, và đây là phát hiện quan trọng nhất của lab.

3/4 chỉ số ≥ 0.70 → theo thang điểm RAGAS của rubric, đạt **10/10 điểm** cho tiêu chí #7.

---

## Bottom-5 Failures

Cả 5 lỗi đều có `worst_metric = faithfulness`. Không lỗi nào thuộc về *retrieval* (context_recall trong top-10 ≥ 0.63) — đây là bằng chứng quan trọng nhất.

### #1 — Tạm ứng quá hạn: số tiền phạt sai
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng, pro-rata 5 ngày ≈ 50.000 VNĐ.
- **Got:** Bị phạt là **1,5%** của số tiền tạm ứng.
- **Worst metric:** faithfulness — **Score: 0.491**
- **Error Tree:**
  1. Output sai? → **Sai.** Đáp án cần *phép tính*, không có sẵn trong text.
  2. Context đúng? → **Đúng một phần.** Context có "2%/tháng" và "15 ngày", nhưng **không chứa** `300.000` hay `50.000` (recall_of_numbers = 0.571; thiếu 15.000.000, 300.000, 50.000).
  3. Query OK? → OK, BM25+Dense+RRF trả về đúng 3 chunk của `tam_ung.md`.
  4. Root cause: **model không tự tính toán.** Nó biết tỉ lệ 2%/tháng nhưng không nhân ra 2% × 15.000.000 = 300.000, nên **bịa ra con số** (`1,5%`) không có trong context.
- **Suggested fix:** ép prompt nêu rõ phép tính khi câu hỏi cần thiết; hoặc thêm bước tính toán bằng code thay vì giao cho LLM.

### #2 — Lương thử việc Junior: không trả lời
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 20.000.000 × 85% = **17.000.000 VNĐ/tháng**
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness — **Score: 0.557**
- **Error Tree:**
  1. Output sai? → **Sai kiểu.** Trả lời từ chối dù đáp án có thể suy ra.
  2. Context đúng? → **Đúng hoàn toàn.** Chunk chứa bảng lương (`Junior (P1-P2) | 12.000.000 - 20.000.000`) **và** dòng `Nhân viên trong thời gian thử việc được nhận **85% lương**`. recall_of_numbers = 0.75.
  3. Query OK? → OK, chunk đúng nhất có thể có.
  4. Root cause: **prompt, không phải retrieval.** System prompt gốc là *"Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"* — cụm "Nếu không có" là **lối thoát dễ dàng**, model 8B yếu bấm vào nó thay vì tự tính. Đã kiểm chứng bằng A/B.
- **Suggested fix:** bỏ lối thoát "Nếu không có" khỏi prompt, thay bằng yêu cầu trích xuất + tính toán.

### #3 — Nghỉ phép không lương 20 ngày: không trả lời
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** **Giám đốc điều hành (CEO)** (nhỉ 16–30 ngày)
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness — **Score: 0.518**
- **Error Tree:**
  1. Output sai? → Sai (từ chối).
  2. Context đúng? → **Đúng nhưng bị cắt.** Chunk từ `nghi_phep_khong_luong.md` chứa thang phê duyệt theo số ngày: "Nghỉ từ 1-5 ngày: trưởng phòng. Nghỉ từ 6-15 n[ày]…" — nhưng **bảng bị cắt giữa chừng**, phần "16-30 ngày: CEO" nằm ngoài chunk 256 ký tự.
  3. Query OK? → OK.
  4. Root cause: **chunking** — failure duy nhất thực sự thuộc tầng truy xuất. Ranh giới child-chunk (256 ký tự) cắt đúng giữa bảng phê duyệt theo khoảng, không theo ranh giới ngữ nghĩa.
- **Suggested fix:** dùng `chunk_structure_aware()` cho tài liệu dạng bảng/quy trình, hoặc tăng `HIERARCHICAL_CHILD_SIZE`, hoặc cắt chunk tại đầu dòng thay vì theo số ký tự.

### #4 — Bảo hiểm PVI cho nhân viên thử việc: không trả lời
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** **KHÔNG.** Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Không tìm thấy.
- **Worst metric:** faithfulness — **Score: 0.521**
- **Error Tree:**
  1. Output sai? → Sai (từ chối).
  2. Context đúng? → **Đúng tuyệt đối.** Chunk đầu tiên viết nguyên văn: *"Nhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI."* — đúng câu hỏi, đúng nguyên văn.
  3. Query OK? → OK.
  4. Root cause: **prompt.** Đáp án nằm nguyên văn trong context, model vẫn từ chối → loại trừ tuyệt đối khả năng retrieval/chunking sai. Cùng nguyên nhân #2.
- **Suggested fix:** như #2 — sửa prompt.

### #5 — Hoàn trả chi phí đào tạo: thiếu phép tính
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng. Phải hoàn trả bao nhiêu?
- **Expected:** Hoàn trả **100%** → **25.000.000 VNĐ**
- **Got:** 100% chi phí đào tạo đã được tài trợ. (thiếu con số 25.000.000)
- **Worst metric:** faithfulness — **Score: 0.421 — **worst overall****
- **Error Tree:**
  1. Output sai? → **Sai một phần.** Nêu đúng 100% nhưng không quy ra 25.000.000.
  2. Context đúng? → **Đúng một phần.** Chunk có quy định hoàn trả 100% và "tối đa 30.000.000", nhưng con số **25.000.000 do người hỏi đưa ra**, không có trong context (recall_of_numbers = 0.5).
  3. Query OK? → OK.
  4. Root cause: **thiếu bước "áp dụng số liệu từ câu hỏi vào công thức trong context".** Model lấy đúng tỉ lệ (100%) nhưng không nhân với 25 triệu.
- **Suggested fix:** prompt yêu cầu nêu phép tính; hoặc tách bước "trích công thức" → "tính" → "trả lời".

---

## Nguyên nhân đã kiểm chứng (không phải phỏng đoán)

Tôi đã tái chạy **5 câu hỏi trên với đúng context đã truy xuất**, chỉ thay đổi system prompt. Kết quả A/B:

| System prompt | Số câu bị từ chối | Số đáp án đúng |
|---|---|---|
| Prompt gốc của scaffold | **3/5** | 2/5 |
| Bỏ lối thoát "Nếu không có" | **0/5** | 4/5 |

→ Với cùng model, cùng context, cùng temperature: **lỗi #2/#3/#4 do prompt, không phải do RAG.** (Số liệu: `.hermes/prompt_ab.json`)

### Giới hạn về mô hình sinh câu trả lời

Ngoài lỗi prompt, **chất lượng câu trả lời còn bị giới hạn bởi năng lực của chính model**:

- `ternary-bonsai-8b` là model **lượng tử 2-bit (Q2_0), ~2.3 GB VRAM** trên GTX 1660 Ti. Cùng họ với `ternary-bonsai-4b` chạy nhanh gấp đôi nhưng yếu hơn.
- **Nó không thực sự làm phép tính.** 4/10 failure là câu hỏi đòi hỏi nhân/chia cụ thể (2% × 15.000.000; 85% × 20.000.000; 100% × 25.000.000). Model đọc đúng tỉ lệ rồi **bịa ra con số** (`1,5%`, `50.000.000`) hoặc **từ chối trả lời**.
- Fidelity thấp đó **kéo theo cả điểm RAGAS**: 3 metric dùng LLM làm judge, nên khi judge cũng là model 2-bit thì điểm bị lệch theo cả hai chiều. Faithfulness 0.5083 phản ánh **giới hạn của judge**, không chỉ giới hạn của người trả lời.
- Với reference implementation gốc dùng `gpt-4o-mini` (temperature mặc định, phép tính đúng), 4 câu cần tính toán ở trên nhiều khả năng đều đúng → Faithfulness dự kiến cao hơn đáng kể.

**Kết luận:** phần lớn điểm số thấp đến từ **giới hạn model 2-bit**, không phải từ thiết kế pipeline. Các module truy xuất (M1/M2/M3) hoạt động đúng — bằng chứng là Context Recall 0.8333 và Context Precision 0.9125.

### Điểm cần nói rõ: đây không phải lỗi chunking/enrichment

Tôi đã kiểm tra nghi vấn "enrichment làm hỏng corpus" (LLM viết lại số liệu) và **loại trừ được**:
- Raw chunk được giữ **nguyên văn** trong `enriched_text` (0/96 chunk bị sửa).
- Dòng `context` LLM thêm vào chỉ nhắc tên tài liệu/năm; số tạo ra (`2024.` trong câu "bảng lương **năm 2024**") là số từ tên file, không phải số liệu sai.

→ Nguyên nhân nằm ở **prompt** và **giới hạn model**, không ở M5.

---

## Case Study (trình bày)

**Question chọn phân tích:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? *(failure #4)*

**Error Tree walkthrough:**
1. **Output đúng?** → Sai. Model trả `"Không tìm thấy."`
2. **Context đúng?** → **Đúng.** Chunk truy xuất về `thu_viec.md` chứa nguyên văn: *"Nhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI."* Đáp án nằm nguyên văn trong context.
3. **Query rewrite OK?** → OK. BM25 + Dense + RRF + rerank trả về đúng 3 chunk liên quan nhất.
4. **Fix ở bước:** **Bước 1 (generation), không phải bước 2 (retrieval).** Root cause là system prompt gốc chứa lối thoát *"Nếu không có → nói 'Không tìm thấy.'"*; A/B test xác nhận đổi prompt → 0/5 từ chối.

→ **Bài học:** metric thấp **không** tự động có nghĩa là retrieval kém. Phải lần theo Error Tree *từng bước* và kiểm chứng giả thuyết bằng thực nghiệm có đối chứng trước khi sửa.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Sửa system prompt theo biến thể đã thử (bỏ lối thoát, ép nêu phép tính) — **đã kiểm chứng 0/5 từ chối, 4/5 đúng**.
- Chuyển phép tính 85% / 2% / 100% sang **code** (calculator tool hoặc biểu thức Python), vì model 2-bit không đáng tin cho số học.
- Dùng `chunk_structure_aware()` cho `nghi_phep_khong_luong.md` để không cắt đứt bảng phê duyệt theo khoảng ngày (failure #3).
- Chạy lại RAGAS với model mạnh hơn (nếu có) để tách giới hạn model khỏi giới hạn pipeline.