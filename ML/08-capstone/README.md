# Giai đoạn 8 — Dự án cuối khóa

**Thời lượng:** 3–4 tuần
**Mục tiêu:** Kết hợp mọi thứ đã học thành **một chatbot AI hoàn chỉnh của riêng bạn** — một "ChatGPT thu nhỏ" mà bạn hiểu từng thành phần.

```text
                                                                                                                     Bạn đang ở đây
                                                                                                                           ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm] → [Capstone]
```

## Mục lục

1. [Ôn tập: toàn bộ pipeline tạo ra một AI](#1-ôn-tập-toàn-bộ-pipeline-tạo-ra-một-ai)
2. [Kiến trúc dự án đề xuất](#2-kiến-trúc-dự-án-đề-xuất)
3. [Chọn hướng dự án](#3-chọn-hướng-dự-án)
4. [Các mốc thực hiện](#4-các-mốc-thực-hiện)
5. [Tiêu chí đánh giá](#5-tiêu-chí-đánh-giá)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

---

## 1. Ôn tập: toàn bộ pipeline tạo ra một AI

Đây là bức tranh bạn phải **tự vẽ lại và giải thích được** mà không nhìn tài liệu:

```text
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. DỮ LIỆU         Crawl web → trích text → lọc → khử trùng lặp → trộn code/sách/toán        │  GĐ 5
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 2. TOKENIZER       BPE: văn bản ↔ chuỗi token ID                                             │  GĐ 4
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 3. KIẾN TRÚC       Embedding → N × [Attention + MLP + residual + norm] → LM head → softmax    │  GĐ 2–4
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 4. PRETRAINING     Đoán token tiếp theo, cross-entropy, AdamW, hàng nghìn GPU → BASE MODEL   │  GĐ 0–5
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 5. SFT             Hội thoại mẫu + chat template + loss masking → biết trò chuyện            │  GĐ 6
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 6. PREFERENCE      RLHF / Constitutional AI / DPO → hữu ích, trung thực, vô hại              │  GĐ 6
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 7. REASONING RL    GRPO / RL với phần thưởng kiểm chứng được → biết suy nghĩ                 │  GĐ 6
 ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 8. SẢN PHẨM        Inference tối ưu + system prompt + RAG + tools + an toàn + đánh giá       │  GĐ 7
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Câu hỏi tự kiểm tra

Trả lời được hết → bạn đã hiểu cách tạo ra ChatGPT/Claude/Gemini:

- [ ] Vì sao "đoán token tiếp theo" lại đủ để mô hình học được kiến thức và suy luận?
- [ ] Attention làm gì mà MLP không làm được, và ngược lại?
- [ ] Vì sao cần residual connection và LayerNorm để huấn luyện mô hình sâu?
- [ ] Base model và chatbot khác nhau thế nào? Điều gì biến cái này thành cái kia?
- [ ] Vì sao LLM bịa (hallucinate)? Có những cách nào để giảm?
- [ ] RLHF, Constitutional AI, DPO khác nhau ở đâu?
- [ ] Mô hình suy luận (o1, R1) được huấn luyện thế nào?
- [ ] Vì sao LLM kém đếm chữ cái nhưng giỏi viết code?
- [ ] Chi phí huấn luyện một mô hình 7B trên 2T token là bao nhiêu FLOPs, cần bao nhiêu bộ nhớ?
- [ ] Khi bạn gõ tin nhắn vào ChatGPT, chuyện gì xảy ra từ lúc bấm Enter đến khi chữ hiện ra?

---

## 2. Kiến trúc dự án đề xuất

```text
   ┌─────────────────────────────────────────────────────────────┐
   │                    GIAO DIỆN WEB CHAT                        │
   │         (Gradio / Streamlit / HTML + FastAPI)                │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    ORCHESTRATOR (Python)                     │
   │                                                             │
   │   system prompt ─┐                                          │
   │   lịch sử chat ──┼──▶ ghép prompt ──▶ gọi LLM ──▶ có gọi    │
   │   tài liệu RAG ──┘                                  tool?   │
   │                                                 ┌────┴───┐  │
   │                                               có│        │không
   │                                     ┌───────────▼──┐     ▼  │
   │                                     │ Thực thi tool│  trả lời│
   │                                     │ (máy tính,   │        │
   │                                     │  tìm kiếm)   │        │
   │                                     └──────┬───────┘        │
   │                                            └─▶ quay lại LLM │
   └──────────┬──────────────────────────────────┬───────────────┘
              ▼                                  ▼
   ┌─────────────────────┐          ┌─────────────────────────────┐
   │   VECTOR DATABASE   │          │   INFERENCE SERVER          │
   │ (Chroma / FAISS)    │          │ (vLLM / Ollama)             │
   │ tài liệu của bạn    │          │ mô hình BẠN TỰ FINE-TUNE    │
   └─────────────────────┘          │ (SFT + DPO/GRPO + LoRA)     │
                                    └─────────────────────────────┘

   + Bộ đánh giá: ~50–100 câu hỏi có đáp án + LLM-as-judge
```

---

## 3. Chọn hướng dự án

| Hướng | Mô tả | Phù hợp nếu |
|---|---|---|
| **A. Trợ lý học tập tiếng Việt** | Fine-tune mô hình mở trên dữ liệu hỏi đáp tiếng Việt + RAG trên tài liệu học của bạn | Muốn sản phẩm dùng được ngay |
| **B. nanochat đầy đủ** | Chạy toàn bộ pipeline nanochat từ pretrain đến chat, thử thay dữ liệu/siêu tham số, viết báo cáo phân tích | Muốn hiểu sâu nhất toàn bộ quy trình |
| **C. Mô hình suy luận nhỏ** | Dùng GRPO dạy mô hình ~0.5–1.5B giải toán tiểu học tiếng Việt, đo độ chính xác trước/sau | Hứng thú với RL và reasoning |
| **D. Trợ lý chuyên ngành** | Chatbot cho một lĩnh vực (luật, y tế tham khảo, lập trình) với RAG + tool + bộ đánh giá kỹ | Muốn dự án cho CV/portfolio |

---

## 4. Các mốc thực hiện

```text
   Tuần 1                 Tuần 2                  Tuần 3                  Tuần 4
   ┌──────────────┐       ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ • Chọn hướng │       │ • Fine-tune  │        │ • RAG + tools│        │ • Đánh giá   │
   │ • Thu thập & │ ────▶ │   SFT (LoRA) │ ─────▶ │ • Orchestrator│ ─────▶ │ • Sửa lỗi    │
   │   làm sạch   │       │ • DPO/GRPO   │        │ • Giao diện  │        │ • Viết báo   │
   │   dữ liệu    │       │ • Đo trước/  │        │   web        │        │   cáo + demo │
   │ • Bộ đánh giá│       │   sau        │        │              │        │              │
   └──────────────┘       └──────────────┘        └──────────────┘        └──────────────┘
```

> 💡 **Viết bộ đánh giá TRƯỚC khi huấn luyện** (tuần 1) — để đo được mỗi thay đổi có thực sự cải thiện hay không.

### Checklist

- [ ] Chọn hướng dự án, viết mô tả 1 trang: mục tiêu, dữ liệu, cách đánh giá
- [ ] Thu thập và làm sạch dữ liệu
- [ ] Xây bộ đánh giá (50–100 câu), đo mô hình gốc làm mốc
- [ ] Fine-tune SFT (LoRA/QLoRA)
- [ ] Preference tuning (DPO) hoặc RL (GRPO)
- [ ] Serve mô hình bằng vLLM/Ollama
- [ ] RAG trên dữ liệu riêng
- [ ] Tool use (ít nhất 2 công cụ)
- [ ] Giao diện web chat
- [ ] Đánh giá cuối: bảng so sánh base → SFT → DPO/GRPO → + RAG
- [ ] Viết README/blog giải thích toàn bộ quá trình và bài học rút ra
- [ ] Quay video demo

---

## 5. Tiêu chí đánh giá

| Tiêu chí | Đạt | Tốt | Xuất sắc |
|---|---|---|---|
| **Mô hình** | Dùng mô hình mở có sẵn | Tự fine-tune SFT | SFT + DPO/GRPO, có số đo cải thiện |
| **Hệ thống** | Chat cơ bản | + RAG | + RAG + tools + streaming |
| **Đánh giá** | Thử tay vài câu | Bộ đánh giá tự động | + phân tích lỗi, so sánh các phiên bản |
| **Hiểu biết** | Chạy được | Giải thích được từng thành phần | Viết blog đủ để người khác học theo |

### Tiêu chí hoàn thành

✅ Demo chạy được **và** giải thích được mọi thành phần (từ tokenizer đến giao diện) cho một người khác.

---

## 6. Tài liệu tham khảo

- 💻 ⭐ [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) — pipeline ChatGPT thu nhỏ hoàn chỉnh
- 📘 [Hugging Face — LLM Course](https://huggingface.co/learn/llm-course) · [TRL docs](https://huggingface.co/docs/trl) · [PEFT docs](https://huggingface.co/docs/peft)
- 📝 [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- 💻 [vLLM docs](https://docs.vllm.ai) · [Ollama](https://ollama.com) · [Gradio](https://www.gradio.app)
- 🎥 Xem lại: [Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — sau khi hoàn thành lộ trình, bạn sẽ hiểu **mọi chi tiết** trong video này

---

## Nhật ký dự án

| Ngày | Việc đã làm | Kết quả / số đo | Vấn đề gặp phải |
|------|-------------|-----------------|-----------------|
|      |             |                 |                 |
