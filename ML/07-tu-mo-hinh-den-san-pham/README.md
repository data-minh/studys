# Giai đoạn 7 — Từ mô hình đến sản phẩm

**Thời lượng:** 3–4 tuần
**Mục tiêu:** Hiểu các lớp hệ thống **xung quanh** LLM khiến ChatGPT, Claude, Gemini dùng được trong thực tế: inference nhanh, RAG, công cụ, đa phương thức, đánh giá, an toàn.

```text
                                                                                                         Bạn đang ở đây
                                                                                                               ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Kiến trúc một sản phẩm chatbot AI](#1-kiến-trúc-một-sản-phẩm-chatbot-ai)
2. [Tối ưu inference](#2-tối-ưu-inference)
3. [Context window và system prompt](#3-context-window-và-system-prompt)
4. [RAG — cho mô hình tra cứu tài liệu](#4-rag--cho-mô-hình-tra-cứu-tài-liệu)
5. [Tool use và agents](#5-tool-use-và-agents)
6. [Đa phương thức: ảnh, âm thanh, video](#6-đa-phương-thức-ảnh-âm-thanh-video)
7. [Ngữ cảnh dài](#7-ngữ-cảnh-dài)
8. [Đánh giá LLM](#8-đánh-giá-llm)
9. [An toàn và interpretability](#9-an-toàn-và-interpretability)
10. [Video bài giảng & tài liệu](#10-video-bài-giảng--tài-liệu)
11. [Checklist & thực hành](#11-checklist--thực-hành)

---

## 1. Kiến trúc một sản phẩm chatbot AI

Mô hình chỉ là **một phần** của sản phẩm:

```text
   ┌──────────────┐
   │ Người dùng   │  web / app / API
   └──────┬───────┘
          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ ỨNG DỤNG / ORCHESTRATOR                                          │
   │                                                                  │
   │  • Ghép system prompt + lịch sử hội thoại + tin nhắn mới         │
   │  • Bộ lọc an toàn đầu vào / đầu ra (classifier)                   │
   │  • Quản lý công cụ: tìm kiếm web, chạy code, đọc file...          │
   │  • RAG: truy xuất tài liệu liên quan                              │
   │  • Bộ nhớ (memory), quản lý context                               │
   └──────────────────────────┬──────────────────────────────────────┘
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ INFERENCE SERVER (cụm GPU)                                        │
   │  • Batching nhiều người dùng  • KV cache  • Lượng tử hóa          │
   │  • Speculative decoding       • Phân tán mô hình trên nhiều GPU    │
   │                        ┌──────────────┐                           │
   │                        │   LLM        │                           │
   │                        └──────────────┘                           │
   └─────────────────────────────────────────────────────────────────┘
                              ▼
                    Trả từng token về (streaming)
```

---

## 2. Tối ưu inference

Phục vụ hàng trăm triệu người dùng → chi phí inference **vượt xa** chi phí huấn luyện theo thời gian.

### Lượng tử hóa (quantization)

Lưu trọng số bằng ít bit hơn:

```text
   BF16:  0.7231  ─┐
   INT8:  92       ├─ cùng một trọng số, ít bit hơn
   INT4:  11      ─┘   (kèm hệ số scale để khôi phục gần đúng)

   Mô hình 70B:   BF16 = 140 GB  │ INT8 = 70 GB │ INT4 ≈ 35–40 GB
                  (2+ GPU 80GB)  │ (1 GPU 80GB) │ (chạy trên máy cá nhân mạnh)

   Chất lượng giảm rất ít với INT8, giảm chút ít với INT4 (tùy phương pháp)
```

Định dạng phổ biến: **GGUF** (llama.cpp, chạy trên CPU/Mac), **GPTQ**, **AWQ**, **FP8** (GPU mới).

### Continuous batching

```text
   Static batching                      Continuous batching ⭐
   (chờ cả batch xong mới nhận mới)     (yêu cầu xong thì thay ngay yêu cầu mới)

   Req A ████████████                   Req A ████████████
   Req B ████░░░░░░░░ ← chờ không        Req B ████ Req D ████████
   Req C ██████░░░░░░    làm gì          Req C ██████ Req E ██████
                                        → GPU luôn bận, throughput tăng nhiều lần
```

**PagedAttention (vLLM):** quản lý KV cache như **bộ nhớ ảo** của hệ điều hành — chia thành các trang nhỏ, cấp phát khi cần → không lãng phí bộ nhớ, phục vụ nhiều người dùng hơn.

### Speculative decoding

Dùng mô hình **nhỏ, nhanh** đoán trước vài token, mô hình **lớn** kiểm tra tất cả trong **một** lần forward:

```text
   Mô hình nháp (nhỏ):   "Hà Nội là thủ đô"  ─▶ đoán nhanh 4 token: [của] [Việt] [Nam] [.]
                                                     │
   Mô hình chính (lớn):  kiểm tra cả 4 cùng lúc (1 forward pass, song song)
                                                     ▼
                         chấp nhận: [của] ✓ [Việt] ✓ [Nam] ✓ [.] ✗ → tự sinh [,]

   → 4 token chỉ với ~1 lần chạy mô hình lớn. Kết quả GIỐNG HỆT chạy mô hình lớn một mình.
```

### Framework phục vụ

| Tool | Dùng khi |
|---|---|
| **vLLM**, SGLang | server GPU, throughput cao |
| **llama.cpp**, Ollama | máy cá nhân, CPU, Mac |
| TensorRT-LLM | tối ưu tối đa trên GPU NVIDIA |

---

## 3. Context window và system prompt

**Context window** = số token tối đa mô hình nhìn thấy cùng lúc — là **"bộ nhớ làm việc"** của mô hình.

```text
   ┌──────────────────────────────── context window ────────────────────────────────┐
   │ [System prompt] [Tài liệu RAG] [Lịch sử hội thoại...] [Tin nhắn mới] [Trả lời]  │
   └─────────────────────────────────────────────────────────────────────────────────┘
     "Bạn là Claude,    kết quả       mọi lượt trước đó         câu hỏi      mô hình
      do Anthropic      tìm kiếm      được gửi LẠI mỗi lần      hiện tại     sinh ra
      tạo ra..."
```

**Hai loại "trí nhớ" của LLM:**

| | Trong trọng số (pretraining) | Trong context window |
|---|---|---|
| Ví như | trí nhớ dài hạn, mơ hồ | trí nhớ làm việc, rõ ràng |
| Cập nhật | phải huấn luyện lại | chỉ cần đưa vào prompt |
| Độ chính xác | dễ bịa với thông tin hiếm | chính xác hơn nhiều |

→ **Mẹo:** muốn LLM trả lời chính xác về tài liệu → **dán tài liệu vào context** thay vì hỏi nó "có nhớ không". Đây là nền tảng của RAG.

---

## 4. RAG — cho mô hình tra cứu tài liệu

**Retrieval-Augmented Generation:** tìm tài liệu liên quan, đưa vào context, rồi mới trả lời.

```text
   CHUẨN BỊ (một lần)
   ┌───────────┐   chia nhỏ    ┌───────────┐  embedding   ┌──────────────────┐
   │ Tài liệu  │ ────────────▶ │  Chunks   │ ───────────▶ │ Vector database  │
   │ PDF, web, │  (~500 token  │ đoạn 1    │  model       │ [0.12, -0.4,...] │
   │ wiki nội  │   mỗi đoạn)   │ đoạn 2    │              │ [0.88,  0.1,...] │
   │ bộ...     │               │ ...       │              │ ...              │
   └───────────┘               └───────────┘              └────────┬─────────┘
                                                                   │
   KHI HỎI                                                         │
   ┌───────────────┐  embedding  ┌──────────┐  tìm k vector gần    │
   │ Câu hỏi user  │ ──────────▶ │ vector q │ ─────────────────────┘
   └───────────────┘             └──────────┘  nhất (cosine similarity)
                                                     │
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │ Prompt = hướng dẫn          │
                                      │        + top-k đoạn liên quan│ ──▶ LLM ──▶ Trả lời
                                      │        + câu hỏi             │          (kèm trích dẫn)
                                      └─────────────────────────────┘
```

**Độ tương đồng cosine:**

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}\cdot\mathbf{d}}{|\mathbf{q}|\,|\mathbf{d}|}$$

— chính là tích vô hướng ở Giai đoạn 0!

**Cải tiến thường dùng:** tìm kiếm lai (vector + từ khóa BM25), **reranker** sắp xếp lại kết quả, viết lại câu hỏi trước khi tìm.

---

## 5. Tool use và agents

### Function calling

Mô hình được post-train để sinh ra **lời gọi hàm có cấu trúc** khi cần; hệ thống thực thi và trả kết quả về context:

```text
   User: "Thời tiết Hà Nội hôm nay thế nào?"
      │
      ▼
   LLM sinh:  {"tool": "get_weather", "args": {"city": "Hà Nội"}}   ← không trả lời ngay
      │
      ▼
   Hệ thống gọi API thời tiết thật ──▶ {"temp": 31, "condition": "nắng"}
      │
      ▼  (kết quả được thêm vào context)
   LLM sinh: "Hôm nay Hà Nội nắng, khoảng 31°C."
```

### Agent — vòng lặp suy nghĩ → hành động → quan sát

```text
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
   ┌─────────┐     ┌──────────────┐     ┌──────────┴──┐
   │ SUY NGHĨ│ ──▶ │  HÀNH ĐỘNG   │ ──▶ │  QUAN SÁT   │
   │ (LLM)   │     │ gọi tool:    │     │ kết quả tool│
   │         │     │ search, code,│     │             │
   └────┬────┘     │ đọc file...  │     └─────────────┘
        │          └──────────────┘
        ▼  khi đủ thông tin
   ┌─────────┐
   │ TRẢ LỜI │
   └─────────┘
```

Ví dụ: Claude Code, ChatGPT agent, Gemini Deep Research. **MCP (Model Context Protocol)** là chuẩn mở để kết nối LLM với công cụ và nguồn dữ liệu.

---

## 6. Đa phương thức: ảnh, âm thanh, video

### Vision Transformer (ViT) — ảnh cũng là chuỗi token

```text
   Ảnh 224×224                  Chia thành 196 ô 16×16         Mỗi ô → 1 vector → như token
   ┌────────────────┐           ┌──┬──┬──┬──┬──┐
   │                │           │1 │2 │3 │..│14│              [v1] [v2] [v3] ... [v196]
   │    🐱          │   ────▶   ├──┼──┼──┼──┼──┤    ────▶          │
   │                │           │15│..│  │  │  │                   ▼
   │                │           ├──┼──┼──┼──┼──┤              Transformer
   └────────────────┘           │..│  │  │  │196              (giống hệt GPT,
                                └──┴──┴──┴──┴──┘               không có causal mask)
```

### CLIP — học chung không gian ảnh & văn bản

```text
   Ảnh ──▶ [Image encoder] ──▶ vector ảnh  ─┐
                                            ├─ kéo GẦN nhau nếu khớp,
   "một con mèo" ──▶ [Text encoder] ──▶ vector văn bản ─┘  đẩy XA nếu không khớp

   Huấn luyện trên 400 triệu cặp (ảnh, chú thích) từ internet
```

### Đưa ảnh vào LLM (kiểu LLaVA)

```text
   Ảnh ──▶ [Vision encoder] ──▶ [Projector] ──▶ "token ảnh" ─┐
                                 (MLP nhỏ)                    ├──▶ [ LLM ] ──▶ "Đây là con mèo
   "Ảnh này có gì?" ──▶ [Tokenizer] ──▶ token văn bản ───────┘               đang ngủ trên ghế"
```

**Gemini** được thiết kế **đa phương thức ngay từ pretraining** (văn bản, ảnh, âm thanh, video trộn lẫn trong dữ liệu), thay vì ghép bộ mã hóa ảnh vào LLM có sẵn. Âm thanh cũng được mã hóa thành chuỗi token (từ spectrogram hoặc audio codec).

---

## 7. Ngữ cảnh dài

Attention có chi phí $O(n^2)$ theo độ dài chuỗi: context 1M token → ma trận attention $10^{12}$ phần tử mỗi head mỗi lớp.

**Các kỹ thuật:**
- **FlashAttention:** không lưu ma trận $n \times n$ (Giai đoạn 5)
- **Mở rộng RoPE** (position interpolation, YaRN): huấn luyện ngắn, dùng dài
- **Sliding window attention:** mỗi token chỉ nhìn $w$ token gần nhất ở một số lớp
- **GQA/MQA:** giảm KV cache
- **Ring attention / context parallelism:** chia chuỗi dài ra nhiều GPU
- Huấn luyện thêm một giai đoạn trên dữ liệu dài

**Kiểm tra:** "needle in a haystack" — giấu một câu trong tài liệu rất dài, hỏi mô hình có tìm ra không.

---

## 8. Đánh giá LLM

| Benchmark | Đo gì |
|---|---|
| **MMLU** | kiến thức 57 môn học (trắc nghiệm) |
| **GSM8K**, **MATH** | toán tiểu học → thi đấu |
| **HumanEval**, **SWE-bench** | viết code, sửa lỗi repo thật |
| **GPQA** | câu hỏi khoa học cấp tiến sĩ |
| **HellaSwag** | suy luận thường thức (dùng trong video GPT-2) |
| **LMArena** (Chatbot Arena) | người dùng thật bình chọn giữa 2 mô hình ẩn danh → xếp hạng Elo |

**LLM-as-judge:** dùng LLM mạnh để chấm điểm câu trả lời theo tiêu chí — rẻ, nhanh, nhưng có thiên kiến (thích câu dài, thích câu của chính nó).

> ⚠️ **Ô nhiễm dữ liệu (contamination):** đề benchmark có thể đã nằm trong dữ liệu pretraining → điểm cao giả tạo. Luôn tự xây bộ đánh giá riêng cho bài toán của bạn.

---

## 9. An toàn và interpretability

### An toàn

- **Red-teaming:** đội chuyên tấn công mô hình để tìm lỗ hổng trước khi phát hành
- **Jailbreak:** prompt lách quy tắc (nhập vai, mã hóa, chia nhỏ yêu cầu...) → phòng thủ bằng huấn luyện + bộ phân loại
- **Prompt injection:** văn bản độc hại trong tài liệu/web mà agent đọc được, cố điều khiển agent
- **Đánh giá năng lực nguy hiểm:** kiểm tra mô hình có hỗ trợ vũ khí sinh học, tấn công mạng... không

### Mechanistic interpretability — "mổ xẻ" bên trong mô hình

**Câu hỏi:** Hàng tỷ con số bên trong mô hình thực sự **biểu diễn** điều gì?

```text
   Vấn đề: một nơ-ron phản ứng với NHIỀU khái niệm không liên quan (polysemantic)
           → mô hình nén nhiều khái niệm hơn số nơ-ron (superposition)

   Giải pháp: Sparse Autoencoder (SAE) tách activation thành các "feature" dễ hiểu

   activation ──▶ [SAE] ──▶ hàng triệu feature, mỗi lúc chỉ vài feature bật
   (d chiều)               ┌────────────────────────────────────┐
                           │ feature #3121: "Cầu Cổng Vàng"     │
                           │ feature #8804: "lỗi bảo mật trong  │
                           │                  code"             │
                           │ feature #...:  "sự nịnh bợ"        │
                           └────────────────────────────────────┘
   Tăng cường một feature → thay đổi hành vi mô hình tương ứng
```

Anthropic đã tìm ra hàng triệu feature dễ hiểu bên trong Claude và các "mạch" (circuits) thực hiện suy luận nhiều bước — lĩnh vực nghiên cứu rất sôi động.

---

## 10. Video bài giảng & tài liệu

### Video

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [Karpathy — Intro to Large Language Models (1 giờ)](https://www.youtube.com/watch?v=zjkBMFhNj_g) | LLM như hệ điều hành, tool use, đa phương thức, an toàn, jailbreak, prompt injection |
| 2 | ⭐ [Karpathy — How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw) | Góc nhìn người dùng: context, tool, tìm kiếm, mô hình suy luận, đa phương thức |
| 3 | [3Blue1Brown — Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) | Ôn tập toàn cảnh |
| 4 | [Stanford CS231n (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) | Nền tảng thị giác máy tính cho phần đa phương thức |
| 5 | [Stanford CS336 (2025)](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) | Các bài về inference, evaluation |

### Đọc thêm

- 📝 ⭐ [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- 📝 [Lilian Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- 📝 [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- 💻 [vLLM docs](https://docs.vllm.ai) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Ollama](https://ollama.com)
- 🎮 [LMArena](https://lmarena.ai) — bảng xếp hạng mô hình do người dùng bình chọn
- 📝 ⭐ [Anthropic — Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — feature bên trong Claude
- 📝 [Transformer Circuits Thread](https://transformer-circuits.pub) — nghiên cứu interpretability của Anthropic
- 📝 [Anthropic Research](https://www.anthropic.com/research)
- 📄 Paper:
  - Inference: [vLLM / PagedAttention (2023)](https://arxiv.org/abs/2309.06180) · [Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192)
  - RAG & tool: [RAG (2020)](https://arxiv.org/abs/2005.11401) · [ReAct (2022)](https://arxiv.org/abs/2210.03629) · [Toolformer (2023)](https://arxiv.org/abs/2302.04761)
  - Đa phương thức: [ViT (2020)](https://arxiv.org/abs/2010.11929) · [CLIP (2021)](https://arxiv.org/abs/2103.00020) · [LLaVA (2023)](https://arxiv.org/abs/2304.08485) · [Gemini (2023)](https://arxiv.org/abs/2312.11805)

---

## 11. Checklist & thực hành

### Lý thuyết

- [ ] Vẽ kiến trúc end-to-end của một sản phẩm chatbot
- [ ] Giải thích quantization, continuous batching, PagedAttention, speculative decoding
- [ ] Phân biệt "trí nhớ trong trọng số" và "trí nhớ trong context"
- [ ] Vẽ pipeline RAG
- [ ] Giải thích function calling và vòng lặp agent
- [ ] Giải thích ViT, CLIP và cách đưa ảnh vào LLM
- [ ] Nêu các kỹ thuật cho ngữ cảnh dài
- [ ] Kể các benchmark chính và vấn đề contamination
- [ ] Giải thích superposition và sparse autoencoder ở mức trực giác

### Thực hành (lưu trong `code/`)

- [ ] Serve mô hình của bạn (từ Giai đoạn 6) bằng vLLM hoặc llama.cpp/Ollama, đo tokens/giây
- [ ] Thử lượng tử hóa 4-bit, so sánh chất lượng và tốc độ
- [ ] Xây RAG trên tài liệu học tập của chính bạn (các file `notes/` trong repo này!)
- [ ] Cho chatbot dùng tool: máy tính + tìm kiếm, tự viết vòng lặp agent
- [ ] Viết bộ đánh giá nhỏ (~50 câu hỏi có đáp án) + LLM-as-judge

### Tiêu chí hoàn thành

✅ Vẽ được và giải thích từng thành phần trong kiến trúc end-to-end của một sản phẩm chatbot AI.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
