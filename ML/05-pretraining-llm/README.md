# Giai đoạn 5 — Pretraining LLM (tạo base model)

**Thời lượng:** 4–6 tuần
**Mục tiêu:** Hiểu cách OpenAI, Anthropic, Google huấn luyện **base model** trên hàng nghìn tỷ token — và tự tái hiện GPT-2 (124M) ở quy mô nhỏ.

```text
                                                                              Bạn đang ở đây
                                                                                    ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Pretraining tạo ra cái gì?](#1-pretraining-tạo-ra-cái-gì)
2. [Dữ liệu: nguyên liệu quan trọng nhất](#2-dữ-liệu-nguyên-liệu-quan-trọng-nhất)
3. [Scaling laws](#3-scaling-laws--quy-luật-mở-rộng)
4. [Công thức huấn luyện](#4-công-thức-huấn-luyện)
5. [Bộ nhớ và mixed precision](#5-bộ-nhớ-và-mixed-precision)
6. [Huấn luyện phân tán](#6-huấn-luyện-phân-tán)
7. [Tăng tốc: FLOPs, MFU, FlashAttention](#7-tăng-tốc-flops-mfu-flashattention)
8. [Inference: sinh văn bản và KV cache](#8-inference-sinh-văn-bản-và-kv-cache)
9. [Kiến trúc LLM hiện đại](#9-kiến-trúc-llm-hiện-đại)
10. [Base model hành xử thế nào?](#10-base-model-hành-xử-thế-nào)
11. [Video bài giảng & tài liệu](#11-video-bài-giảng--tài-liệu)
12. [Checklist & thực hành](#12-checklist--thực-hành)

---

## 1. Pretraining tạo ra cái gì?

```text
   ┌─────────────────────┐        ┌────────────────────┐        ┌─────────────────────┐
   │  ~15 nghìn tỷ token │        │  Hàng nghìn GPU    │        │    BASE MODEL       │
   │  văn bản internet,  │ ─────▶ │  × vài tháng       │ ─────▶ │  (ví dụ: nén        │
   │  sách, code, ...    │        │  × hàng chục-trăm  │        │   internet vào      │
   │  (~vài chục TB)     │        │    triệu USD       │        │   ~vài trăm GB      │
   └─────────────────────┘        └────────────────────┘        │   trọng số)         │
                                                                └─────────────────────┘
```

Base model là một **"bộ mô phỏng tài liệu internet"**: đưa vào đầu một văn bản, nó viết tiếp theo phong cách của internet. Nó **chưa phải trợ lý** — chưa biết trả lời câu hỏi, chưa biết từ chối, chưa có "tính cách". Giai đoạn 6 sẽ biến nó thành ChatGPT/Claude.

> 💡 Pretraining chiếm **~95–99% compute** của toàn bộ quá trình tạo ra một LLM. Kiến thức của mô hình hầu hết đến từ đây.

---

## 2. Dữ liệu: nguyên liệu quan trọng nhất

### Pipeline xử lý dữ liệu

```text
   Common Crawl (hàng tỷ trang web, petabyte HTML)
        │
        ▼
   ┌──────────────────┐  bỏ HTML, menu, quảng cáo → giữ nội dung chính
   │ Trích xuất text  │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  bỏ web độc hại, spam, người lớn (danh sách chặn URL)
   │ Lọc URL          │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  giữ ngôn ngữ mong muốn (fastText language ID)
   │ Nhận diện ngôn ngữ│
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  heuristic (độ dài, tỉ lệ ký tự lạ, lặp lại...)
   │ Lọc chất lượng   │  + bộ phân loại chất lượng (ví dụ: "có giá trị giáo dục?")
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  bỏ trang trùng lặp / gần trùng lặp (MinHash)
   │ Khử trùng lặp    │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  xóa email, số điện thoại, thông tin cá nhân
   │ Xóa PII          │
   └────────┬─────────┘
            ▼
   + Trộn thêm: code (GitHub), sách, Wikipedia, paper khoa học, toán, đa ngôn ngữ
            ▼
   Tokenize → chia thành các "shard" nhị phân → sẵn sàng huấn luyện
```

### Quy mô dữ liệu

| Mô hình | Năm | Số token huấn luyện |
|---|---|---|
| GPT-2 | 2019 | ~10 tỷ (WebText 40GB) |
| GPT-3 | 2020 | 300 tỷ |
| LLaMA 1 | 2023 | 1.4 nghìn tỷ |
| Llama 3 | 2024 | 15 nghìn tỷ |
| FineWeb (dataset mở) | 2024 | 15 nghìn tỷ |

> 🔗 **Bài học ngành:** **Chất lượng dữ liệu** quan trọng ngang (hoặc hơn) kiến trúc mô hình. FineWeb-Edu (lọc giữ nội dung giáo dục) cho kết quả tốt hơn hẳn với cùng số token.

---

## 3. Scaling laws — quy luật mở rộng

Phát hiện quan trọng nhất dẫn đến kỷ nguyên LLM: **loss giảm đều đặn và dự đoán được** khi tăng mô hình, dữ liệu, compute — theo **luật lũy thừa** (power law).

```text
   loss (thang log)
     │●
     │  ●
     │    ●                    Mỗi lần tăng compute 10×,
     │      ●                  loss giảm một lượng gần như cố định
     │        ●                → đường THẲNG trên đồ thị log-log
     │          ●
     │            ●
     │              ●
     └──────────────────── compute (thang log, FLOPs)
      10¹⁸   10²⁰   10²²   10²⁴
```

$$L(N, D) \approx E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- $N$: số tham số, $D$: số token, $E$: loss tối thiểu không thể giảm (entropy của ngôn ngữ)

**Compute huấn luyện** xấp xỉ:

$$C \approx 6ND \quad \text{(FLOPs)}$$

(2ND cho forward + 4ND cho backward)

### Chinchilla (2022): cân bằng mô hình và dữ liệu

Với ngân sách compute cố định, tối ưu là **~20 token cho mỗi tham số**.

```text
   Cùng compute:
   Gopher     280B tham số × 300B token   → kém hơn
   Chinchilla  70B tham số × 1.4T token   → TỐT HƠN (mô hình nhỏ hơn 4×, dữ liệu nhiều hơn)
```

Thực tế hiện nay: các công ty huấn luyện **vượt xa** tỉ lệ Chinchilla (Llama 3 8B dùng 15T token ≈ 1875 token/tham số) vì mô hình nhỏ hơn thì **rẻ hơn khi phục vụ người dùng**.

> 🔗 **Ý nghĩa:** Scaling laws cho phép công ty **dự đoán** hiệu năng của mô hình khổng lồ bằng cách thử nghiệm trên mô hình nhỏ — trước khi chi hàng trăm triệu USD.

---

## 4. Công thức huấn luyện

### Learning rate schedule: warmup + cosine decay

```text
   learning rate
     │
 max ┤      ╭──╮
     │     ╱    ╲___
     │    ╱         ╲___
     │   ╱              ╲___
     │  ╱                   ╲___
 min ┤ ╱                        ╲─────
     └─┬───────────────────────────────── bước
       warmup      cosine decay
       (vài trăm-   (giảm dần theo
        nghìn bước)  đường cosin)
```

- **Warmup:** đầu huấn luyện, trọng số ngẫu nhiên, gradient lớn và hỗn loạn → bắt đầu với LR nhỏ.
- **Decay:** cuối huấn luyện, giảm LR để "tinh chỉnh" vào đáy của loss.

### Các siêu tham số điển hình (GPT-3 style)

| Thành phần | Giá trị điển hình |
|---|---|
| Optimizer | AdamW, $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$ |
| Weight decay | 0.1 |
| Gradient clipping | chuẩn gradient tối đa 1.0 |
| Batch size | 0.5M – vài triệu token / bước |
| Max LR | ~6e-4 (mô hình nhỏ) → ~1e-4 (mô hình lớn) |

### Gradient accumulation — batch lớn trên GPU nhỏ

Muốn batch 0.5M token nhưng GPU chỉ chứa 16K token? Chạy 32 mini-batch, **cộng dồn gradient**, rồi mới `optimizer.step()`:

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps   # chia để lấy trung bình
    loss.backward()                         # gradient tự cộng dồn
optimizer.step(); optimizer.zero_grad()
```

---

## 5. Bộ nhớ và mixed precision

### Các định dạng số

```text
   FP32  │S│ Exponent (8) │        Mantissa (23)          │  4 byte — chính xác, chậm
   BF16  │S│ Exponent (8) │ Mant (7) │                       2 byte — cùng DẢI giá trị FP32,
                                                                      kém chính xác hơn ⭐
   FP16  │S│ Exp (5) │ Mantissa (10) │                       2 byte — dải hẹp, dễ tràn số
   FP8   │S│Exp(4)│M(3)│                                     1 byte — LLM mới nhất
```

**BF16** là chuẩn cho huấn luyện LLM: nhanh gấp nhiều lần FP32 trên GPU hiện đại, bộ nhớ giảm một nửa, và cùng dải số mũ với FP32 nên ít bị tràn số.

### Cần bao nhiêu bộ nhớ để huấn luyện?

Với mixed precision + AdamW, mỗi tham số tốn khoảng **16 byte**:

```text
   Trọng số BF16         2 byte
   Gradient BF16         2 byte
   Bản sao FP32          4 byte   (master weights)
   Adam m (FP32)         4 byte
   Adam v (FP32)         4 byte
   ─────────────────────────────
   Tổng                 16 byte / tham số   (chưa tính activation!)

   Mô hình 7B  → 112 GB  → không vừa 1 GPU 80GB → phải chia nhiều GPU
   Mô hình 70B → 1.1 TB
```

---

## 6. Huấn luyện phân tán

```text
   1. DATA PARALLEL (DDP)                    2. FSDP / ZeRO
   Mỗi GPU có BẢN SAO ĐẦY ĐỦ mô hình,        Chia NHỎ trọng số, gradient, optimizer
   xử lý dữ liệu khác nhau,                   state ra các GPU; ghép lại khi cần
   trung bình gradient (all-reduce)

   GPU0 [Model] ← batch 0 ─┐                  GPU0 [¼ model]  ┐
   GPU1 [Model] ← batch 1 ─┼─ all-reduce      GPU1 [¼ model]  ├─ all-gather khi
   GPU2 [Model] ← batch 2 ─┤  gradient        GPU2 [¼ model]  │  forward/backward
   GPU3 [Model] ← batch 3 ─┘                  GPU3 [¼ model]  ┘
   Đơn giản, nhưng mô hình phải vừa 1 GPU    Tiết kiệm bộ nhớ, mô hình lớn hơn nhiều

   3. TENSOR PARALLEL                        4. PIPELINE PARALLEL
   Chia MỖI PHÉP NHÂN MA TRẬN                 Chia CÁC LỚP ra các GPU
   ra nhiều GPU

        W = [W₁ │ W₂]                         GPU0: lớp 1–24  ──▶
   GPU0: X @ W₁ ┐                             GPU1: lớp 25–48 ──▶
   GPU1: X @ W₂ ┴─ ghép kết quả               GPU2: lớp 49–72 ──▶
   (cần kết nối cực nhanh — NVLink)           GPU3: lớp 73–96 ──▶ loss
```

LLM lớn dùng **kết hợp cả 4** ("3D/4D parallelism") trên hàng chục nghìn GPU.

```python
# DDP trong PyTorch — chạy bằng: torchrun --nproc_per_node=8 train.py
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

---

## 7. Tăng tốc: FLOPs, MFU, FlashAttention

- **FLOPs:** số phép tính dấu phẩy động. GPU H100 đạt ~1000 TFLOPS (BF16, dense).
- **MFU (Model FLOPs Utilization):** % sức mạnh GPU thực sự dùng. Huấn luyện LLM tốt đạt 40–60%.

**Nút thắt thường là bộ nhớ, không phải tính toán:**

```text
   ┌──────────────────────────────────────────┐
   │ GPU                                       │
   │  ┌────────────┐    chậm     ┌──────────┐  │
   │  │ HBM (80GB) │ ◀─────────▶ │ SRAM     │  │    SRAM nhanh hơn ~10× nhưng
   │  │ bộ nhớ lớn │   ~3 TB/s   │ (~50MB)  │  │    rất nhỏ
   │  └────────────┘             │ + nhân   │  │
   │                             │ tính toán│  │
   │                             └──────────┘  │
   └──────────────────────────────────────────┘
```

**FlashAttention:** tính attention **theo từng khối** trong SRAM, không bao giờ ghi ma trận $T \times T$ đầy đủ ra HBM → nhanh hơn 2–4×, tiết kiệm bộ nhớ. Kết quả **giống hệt** attention thường.

Các mẹo tăng tốc trong video *Reproduce GPT-2*: `torch.compile`, BF16 autocast, FlashAttention (`F.scaled_dot_product_attention`), kích thước là lũy thừa của 2 (vocab 50257 → 50304), fused AdamW.

---

## 8. Inference: sinh văn bản và KV cache

### Vấn đề

Sinh token thứ 1000 cần attention với 999 token trước. Nếu tính lại K, V cho **toàn bộ** chuỗi mỗi bước → lãng phí khổng lồ.

### KV cache

K, V của các token cũ **không đổi** (nhờ causal mask) → lưu lại, mỗi bước chỉ tính Q, K, V cho **token mới**:

```text
   Không có KV cache                     Có KV cache

   Bước 3: tính K,V cho [t1 t2 t3]       Bước 3: cache [K1 K2] + tính K3  → [K1 K2 K3]
   Bước 4: tính K,V cho [t1 t2 t3 t4]    Bước 4: cache [K1 K2 K3] + tính K4
   Bước 5: tính K,V cho [t1 ... t5]      Bước 5: cache [K1..K4]   + tính K5
           ↑ lặp lại công việc O(T²)             ↑ mỗi bước chỉ O(T)
```

**Cái giá:** bộ nhớ. Kích thước KV cache mỗi token:

$$2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}$$

Với context dài (100K+ token), KV cache có thể lớn hơn cả trọng số mô hình → động lực cho GQA (mục 9).

### Hai pha của inference

```text
   PREFILL (xử lý prompt)                  DECODE (sinh từng token)
   Toàn bộ prompt cùng lúc, song song      Một token mỗi bước, tuần tự
   → giới hạn bởi TÍNH TOÁN                → giới hạn bởi BĂNG THÔNG BỘ NHỚ
   → "time to first token"                 → "tokens per second"
```

---

## 9. Kiến trúc LLM hiện đại

So với GPT-2 (2019), các LLM hiện đại (LLaMA, Qwen, Mistral, DeepSeek...) có các cải tiến:

| Thành phần | GPT-2 | LLM hiện đại | Lý do |
|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | đơn giản, nhanh hơn, hiệu quả tương đương |
| Vị trí | Learned | **RoPE** | tương đối, mở rộng context dài |
| Activation MLP | GELU | **SwiGLU** | chất lượng tốt hơn |
| Attention | Multi-Head | **Grouped-Query (GQA)** | KV cache nhỏ hơn nhiều |
| Bias | có | **không** | ổn định hơn |
| MLP | dense | **Mixture of Experts** (một số mô hình) | nhiều tham số, ít tính toán |

### Grouped-Query Attention (GQA)

```text
   Multi-Head (MHA)          Grouped-Query (GQA) ⭐       Multi-Query (MQA)
   mỗi Q có K,V riêng        nhóm Q dùng chung K,V        mọi Q dùng chung 1 K,V

   Q: ● ● ● ● ● ● ● ●        Q: ● ● ● ● ● ● ● ●           Q: ● ● ● ● ● ● ● ●
      │ │ │ │ │ │ │ │           └┬┘ └┬┘ └┬┘ └┬┘              └──────┬──────┘
   KV:● ● ● ● ● ● ● ●        KV:  ●   ●   ●   ●            KV:      ●
   KV cache: 8 phần          KV cache: 4 phần             KV cache: 1 phần
   chất lượng tốt nhất       cân bằng                     nhanh nhất, kém hơn
```

### Mixture of Experts (MoE)

Thay 1 MLP lớn bằng **nhiều MLP "chuyên gia"**; một **router** chọn vài chuyên gia cho mỗi token:

```text
                    token x
                       │
                  ┌────▼────┐
                  │ Router  │  → chọn top-2 trong 8 chuyên gia
                  └────┬────┘
          ┌──────┬─────┼─────┬──────┬──────┬──────┬──────┐
          ▼      ▼     ▼     ▼      ▼      ▼      ▼      ▼
         [E1]  [E2]  [E3]  [E4]   [E5]   [E6]   [E7]   [E8]
          ✓                  ✓
          └──────┬───────────┘
                 ▼
          tổng có trọng số → output

   Mixtral 8×7B: tổng ~47B tham số, nhưng mỗi token chỉ dùng ~13B
   → "kiến thức" của mô hình lớn, tốc độ của mô hình nhỏ
```

> 🔗 **Liên hệ:** Nhiều mô hình hàng đầu được cho là (hoặc công bố là) dùng MoE — ví dụ DeepSeek-V3, Mixtral, và theo báo cáo kỹ thuật, một số phiên bản Gemini.

---

## 10. Base model hành xử thế nào?

```text
   Prompt:   "Thủ đô của Pháp là"
   Base:     "Paris. Thủ đô của Đức là Berlin. Thủ đô của Ý là Rome. Thủ đô của..."
             ↑ viết tiếp như một trang web/danh sách — không "trả lời" rồi dừng

   Prompt:   "Bạn có thể giúp tôi viết email xin nghỉ phép không?"
   Base:     "Bạn có thể giúp tôi viết CV không? Bạn có thể giúp tôi..."
             ↑ có thể nghĩ đây là một diễn đàn hỏi đáp và viết thêm câu hỏi
```

**Mẹo dùng base model — few-shot prompting:** tạo "tài liệu" có cấu trúc để nó viết tiếp đúng ý:

```text
   Tiếng Anh: cat      → Tiếng Việt: mèo
   Tiếng Anh: dog      → Tiếng Việt: chó
   Tiếng Anh: teacher  → Tiếng Việt: ▌     ← base model điền "giáo viên"
```

**Kiến thức trong trọng số = "trí nhớ mơ hồ":** mô hình nén internet một cách có mất mát. Thông tin xuất hiện nhiều → nhớ tốt; thông tin hiếm → nhớ mơ hồ, dễ **bịa (hallucination)**.

---

## 11. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | Toàn cảnh: dữ liệu, tokenize, pretrain, base model, SFT, RL (3.5 giờ) — **xem phần đầu cho giai đoạn này** |
| 2 | ⭐⭐ [Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | Huấn luyện GPT-2 thật: BF16, FlashAttention, torch.compile, DDP, FineWeb-Edu, đánh giá HellaSwag (4 giờ) |
| 3 | [Umar Jamil — LLaMA explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo) | KV cache, RoPE, RMSNorm, GQA, SwiGLU |
| 4 | [Umar Jamil — Mistral / Mixtral explained](https://www.youtube.com/watch?v=UiX8K-xBUpE) | Sliding window attention, Mixture of Experts |
| 5 | ⭐ [Stanford CS336 — Language Modeling from Scratch (2025)](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) | Khóa học đầy đủ nhất về xây LLM: tokenizer, kiến trúc, GPU, parallelism, scaling laws, dữ liệu, alignment |

### Đọc thêm

- 💻 [github.com/karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — code của video Reproduce GPT-2, commit theo từng bước
- 💻 [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c) — huấn luyện GPT-2 bằng C/CUDA thuần
- 📝 ⭐ [Hugging Face — FineWeb: decanting the web](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — cách xây dataset pretraining chất lượng
- 📝 ⭐ [Hugging Face — The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — huấn luyện LLM trên cụm GPU, giải thích DP/TP/PP/ZeRO bằng hình
- 📄 Paper nền tảng:
  - [GPT-2 (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [GPT-3 (2020)](https://arxiv.org/abs/2005.14165)
  - [Scaling Laws — Kaplan (2020)](https://arxiv.org/abs/2001.08361) · [Chinchilla (2022)](https://arxiv.org/abs/2203.15556)
  - [LLaMA (2023)](https://arxiv.org/abs/2302.13971) · [The Llama 3 Herd of Models (2024)](https://arxiv.org/abs/2407.21783)
  - [FineWeb (2024)](https://arxiv.org/abs/2406.17557)
- 📄 Paper kỹ thuật:
  - [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) · [GQA (2023)](https://arxiv.org/abs/2305.13245) · [GLU Variants / SwiGLU (2020)](https://arxiv.org/abs/2002.05202)
  - [Switch Transformers / MoE (2021)](https://arxiv.org/abs/2101.03961) · [Mixtral (2024)](https://arxiv.org/abs/2401.04088)
  - [ZeRO (2019)](https://arxiv.org/abs/1910.02054) · [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053)

---

## 12. Checklist & thực hành

### Lý thuyết

- [ ] Vẽ lại pipeline xử lý dữ liệu pretraining
- [ ] Giải thích scaling laws và công thức $C \approx 6ND$
- [ ] Giải thích kết luận Chinchilla và vì sao thực tế người ta vượt tỉ lệ này
- [ ] Giải thích warmup + cosine decay, gradient accumulation, gradient clipping
- [ ] Tính được bộ nhớ cần để huấn luyện mô hình $N$ tham số (16 byte/tham số)
- [ ] Phân biệt DDP, FSDP, tensor parallel, pipeline parallel
- [ ] Giải thích KV cache và công thức kích thước
- [ ] Giải thích RMSNorm, SwiGLU, GQA, MoE
- [ ] Giải thích vì sao base model chưa phải chatbot

### Thực hành (lưu trong `code/`)

- [ ] Theo *Let's reproduce GPT-2* → huấn luyện GPT-2 124M trên FineWeb-Edu (bản nhỏ trên Colab/Kaggle; bản đầy đủ thuê GPU ~vài chục USD)
- [ ] Tự cài đặt KV cache cho GPT của mình, đo tốc độ sinh token trước/sau
- [ ] Nâng cấp GPT: RoPE + RMSNorm + SwiGLU (kiến trúc kiểu LLaMA), so sánh loss
- [ ] Viết script tính: số tham số, FLOPs huấn luyện, bộ nhớ cần cho một cấu hình bất kỳ
- [ ] Tải một base model mở (ví dụ `Qwen/Qwen2.5-0.5B` bản base) bằng `transformers`, thử prompt thường vs few-shot

### Tiêu chí hoàn thành

✅ Giải thích được base model khác chatbot thế nào, vì sao cần giai đoạn 6, và ước tính được chi phí (compute + bộ nhớ) để huấn luyện một mô hình cho trước.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
