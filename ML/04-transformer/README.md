# Giai đoạn 4 — Transformer ⭐⭐ (trái tim của mọi LLM)

**Thời lượng:** 4 tuần
**Mục tiêu:** Tự tay viết một GPT từ đầu và hiểu **từng dòng code**. GPT, Claude, Gemini đều là Transformer — khác nhau chủ yếu ở kích thước, dữ liệu và các cải tiến nhỏ.

```text
                                                               Bạn đang ở đây
                                                                     ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Toàn cảnh kiến trúc GPT](#1-toàn-cảnh-kiến-trúc-gpt)
2. [Tokenization với BPE](#2-tokenization-với-bpe)
3. [Token embedding + positional embedding](#3-token-embedding--positional-embedding)
4. [Attention — trực giác](#4-attention--trực-giác)
5. [Self-attention — từng bước tính toán](#5-self-attention--từng-bước-tính-toán)
6. [Causal mask — không được nhìn tương lai](#6-causal-mask--không-được-nhìn-tương-lai)
7. [Multi-head attention](#7-multi-head-attention)
8. [Transformer block hoàn chỉnh](#8-transformer-block-hoàn-chỉnh)
9. [Đầu ra, loss và huấn luyện song song](#9-đầu-ra-loss-và-huấn-luyện-song-song)
10. [Mã hóa vị trí: sinusoidal, learned, RoPE](#10-mã-hóa-vị-trí-sinusoidal-learned-rope)
11. [Các họ Transformer](#11-các-họ-transformer)
12. [Đếm tham số GPT-2](#12-đếm-tham-số-gpt-2)
13. [Video bài giảng & tài liệu](#13-video-bài-giảng--tài-liệu)
14. [Checklist & thực hành](#14-checklist--thực-hành)

---

## 1. Toàn cảnh kiến trúc GPT

GPT là **decoder-only Transformer**: một chồng các block giống hệt nhau.

```text
                    "Hà Nội là thủ"
                           │
                           ▼
                ┌─────────────────────┐
                │     Tokenizer       │  văn bản → token IDs
                └──────────┬──────────┘
                           │  [18, 902, 55, 3071]              shape (B, T)
                           ▼
                ┌─────────────────────┐
                │  Token Embedding    │  mỗi ID → vector d chiều
                │        +            │
                │ Position Embedding  │  cộng thông tin vị trí
                └──────────┬──────────┘
                           │                                    shape (B, T, d)
            ┌──────────────▼──────────────┐
            │  ┌───────────────────────┐  │
            │  │ LayerNorm             │  │
            │  │ Masked Multi-Head     │  │  "giao tiếp": token
            │  │ Self-Attention        │  │  trao đổi thông tin
            │  └──────────┬────────────┘  │
            │            (+) ◀── residual │
            │  ┌──────────▼────────────┐  │    × N lớp
            │  │ LayerNorm             │  │    (GPT-2 small: 12
            │  │ MLP (Feed-Forward)    │  │     GPT-3: 96)
            │  │                       │  │  "suy nghĩ": từng token
            │  └──────────┬────────────┘  │  tự xử lý
            │            (+) ◀── residual │
            └──────────────┬──────────────┘
                           ▼                                    shape (B, T, d)
                ┌─────────────────────┐
                │  Final LayerNorm    │
                │  Linear (LM head)   │  vector → điểm cho từng token
                └──────────┬──────────┘
                           ▼                                    shape (B, T, vocab)
                     softmax → P(token tiếp theo)
                           │
                           ▼
                         "đô"
```

**Hai thao tác xen kẽ trong mỗi block:**
- **Attention = giao tiếp:** các token "nói chuyện" với nhau, thu thập thông tin từ token khác.
- **MLP = tính toán:** mỗi token tự "suy nghĩ" về thông tin vừa thu thập (độc lập với nhau).

> 🎮 **Xem trực quan 3D từng phép tính:** [bbycroft.net/llm](https://bbycroft.net/llm) và [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — **nên mở song song khi học phần này.**

---

## 2. Tokenization với BPE

### Byte Pair Encoding (BPE) — thuật toán

1. Bắt đầu với từ điển = 256 byte.
2. Đếm cặp token liền kề xuất hiện nhiều nhất.
3. Gộp cặp đó thành token mới, thêm vào từ điển.
4. Lặp lại đến khi đủ kích thước từ điển mong muốn.

```text
   Dữ liệu:  "aaabdaaabac"

   Bước 0:   a a a b d a a a b a c       cặp nhiều nhất: "aa"
   Bước 1:   Z a b d Z a b a c           Z = "aa"         cặp nhiều nhất: "Za"
   Bước 2:   Y b d Y b a c               Y = "Za" = "aaa" cặp nhiều nhất: "Yb"
   Bước 3:   X d X a c                   X = "Yb" = "aaab"

   Từ điển mới: {a, b, c, d, Z, Y, X}    chuỗi 11 token → 5 token
```

### Tokenizer thật trông thế nào?

```text
   "Hello world! Xin chào các bạn"   (tokenizer GPT-4, minh họa)

   |Hello| world|!| X|in| ch|ào| các| b|ạn|
      ↑      ↑           ↑
   từ tiếng Anh phổ biến  tiếng Việt bị chia nhỏ hơn
   = 1 token              (ít dữ liệu tiếng Việt khi học BPE)
```

### Hệ quả thú vị

- **LLM kém đếm chữ cái** ("strawberry có mấy chữ r?") vì nó **không thấy chữ cái** — chỉ thấy các token như `str|aw|berry`.
- **Tiếng Việt tốn nhiều token hơn** tiếng Anh cho cùng nội dung → đắt hơn khi dùng API, context ngắn hơn.
- **Số học khó** vì số bị chia token không đều: `1234567` → `123|456|7`.

> 🎮 Thử ngay: [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app) — gõ tiếng Việt và xem cách chia token.

---

## 3. Token embedding + positional embedding

```text
   token IDs:     [18]      [902]     [55]      [3071]
                    │         │         │          │
                    ▼         ▼         ▼          ▼
   Token emb:    [0.1,..]  [0.7,..]  [-0.2,..]  [0.4,..]    wte: (vocab, d)
                    +         +         +          +
   Position emb: [pos 0]   [pos 1]   [pos 2]    [pos 3]     wpe: (max_len, d)
                    =         =         =          =
   x:            [.....]   [.....]   [.....]    [.....]     shape (T, d)
```

**Vì sao cần vị trí?** Attention (mục 5) là phép **tổng có trọng số** — không phân biệt thứ tự. Không có thông tin vị trí, "chó cắn người" và "người cắn chó" là như nhau!

---

## 4. Attention — trực giác

Xét câu: *"Bức tường rất cao nên con mèo không nhảy qua được **nó**."*

"nó" chỉ gì — con mèo hay bức tường? Để hiểu, token "nó" cần **nhìn lại** các token trước và **lấy thông tin từ token liên quan**:

```text
   Bức  tường  rất  cao  nên  con  mèo  không  nhảy  qua  được  nó
    ░    ███    ░    ▒    ░    ░    ▒     ░     ▒     ▒     ░    ░
          ▲
          "nó" chú ý mạnh nhất vào "tường"
          (vật bị "nhảy qua" phải là bức tường, không phải con mèo)
```

### Phép so sánh: tra cứu thư viện

Mỗi token tạo ra 3 vector:

| Vector | Vai trò | Ví dụ |
|---|---|---|
| **Query (Q)** | "Tôi đang tìm gì?" | "nó" hỏi: *tôi là đại từ, tôi tìm danh từ mà tôi thay thế* |
| **Key (K)** | "Tôi chứa gì?" (nhãn) | "tường": *tôi là danh từ, vật thể* |
| **Value (V)** | "Thông tin tôi đưa ra" | "tường": *nội dung ý nghĩa của "tường"* |

```text
   Query của "nó" ──so khớp (tích vô hướng)──▶ Key của mọi token trước
                                                     │
                                                     ▼
                                     điểm liên quan → softmax → trọng số
                                                     │
                                                     ▼
                  Output của "nó" = Σ (trọng số × Value của từng token)
```

---

## 5. Self-attention — từng bước tính toán

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

### Bước 1: Tạo Q, K, V bằng 3 phép chiếu tuyến tính

```text
   X (T, d)  ──@ W_Q (d, d_k)──▶  Q (T, d_k)
   X (T, d)  ──@ W_K (d, d_k)──▶  K (T, d_k)
   X (T, d)  ──@ W_V (d, d_v)──▶  V (T, d_v)
```

### Bước 2: Tính điểm liên quan — mọi cặp token

```text
   Q (T, d_k)  @  Kᵀ (d_k, T)  =  Scores (T, T)

                     key:  "Hà"  "Nội"  "là"  "thủ"
                  ┌────────────────────────────────┐
   query: "Hà"    │        2.1   1.3   -0.5   0.2  │
          "Nội"   │        1.8   2.5    0.1   0.3  │
          "là"    │        0.4   1.1    1.9   0.6  │
          "thủ"   │        0.9   2.2    0.7   1.5  │
                  └────────────────────────────────┘
                  Scores[i, j] = độ liên quan của token j với token i
```

### Bước 3: Chia cho $\sqrt{d_k}$

Tích vô hướng của vector $d_k$ chiều có phương sai ~$d_k$. Nếu không chia, điểm quá lớn → softmax gần như one-hot → gradient ≈ 0. Chia $\sqrt{d_k}$ đưa phương sai về ~1.

### Bước 4: Causal mask + softmax theo hàng (xem mục 6)

### Bước 5: Tổng có trọng số các Value

```text
   Weights (T, T)  @  V (T, d_v)  =  Output (T, d_v)

   Output["thủ"] = 0.14·V["Hà"] + 0.50·V["Nội"] + 0.11·V["là"] + 0.25·V["thủ"]
```

### Code (single head)

```python
q, k, v = self.query(x), self.key(x), self.value(x)        # (B, T, hs)
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5        # (B, T, T)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v                                              # (B, T, hs)
```

---

## 6. Causal mask — không được nhìn tương lai

Khi huấn luyện, mô hình thấy cả câu. Nhưng khi dự đoán token ở vị trí $t$, nó **không được** nhìn token $t+1, t+2, ...$ (nếu không sẽ "gian lận" — nhìn đáp án).

```text
   Scores trước mask            Cộng mask M                  Sau softmax (mỗi hàng tổng = 1)

        Hà  Nội  là  thủ            Hà  Nội  là  thủ              Hà    Nội   là    thủ
   Hà  [2.1 1.3 -0.5 0.2]      Hà  [ 0  -∞   -∞   -∞]       Hà  [1.00  0     0     0   ]
   Nội [1.8 2.5  0.1 0.3]  +   Nội [ 0   0   -∞   -∞]  →    Nội [0.33  0.67  0     0   ]
   là  [0.4 1.1  1.9 0.6]      là  [ 0   0    0   -∞]       là  [0.13  0.27  0.60  0   ]
   thủ [0.9 2.2  0.7 1.5]      thủ [ 0   0    0    0]       thủ [0.14  0.50  0.11  0.25]

                               e^(-∞) = 0 → trọng số tương lai bằng 0
                               ma trận tam giác dưới (lower triangular)
```

---

## 7. Multi-head attention

Một head chỉ học được **một kiểu quan hệ**. Dùng nhiều head song song, mỗi head học một kiểu:

```text
                         x (T, d=768)
          ┌──────────┬───────┴───────┬──────────┐
          ▼          ▼               ▼          ▼
      ┌───────┐  ┌───────┐       ┌───────┐  ┌───────┐
      │Head 1 │  │Head 2 │  ...  │Head 11│  │Head 12│    mỗi head: d_k = 768/12 = 64
      │ngữ    │  │đại từ │       │token  │  │chủ ngữ│    (ví dụ minh họa — thực tế
      │pháp   │  │→ danh │       │liền   │  │↔ động │     head học gì phải phân tích
      │       │  │từ     │       │trước  │  │từ     │     mới biết)
      └───┬───┘  └───┬───┘       └───┬───┘  └───┬───┘
          └──────────┴──────┬────────┴──────────┘
                            ▼
                     Nối lại (T, 768)
                            ▼
                   Linear W_O (768 × 768)   ← trộn thông tin các head
                            ▼
                      output (T, 768)
```

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W_O$$

Tổng chi phí tính toán **bằng** một head lớn, nhưng biểu diễn phong phú hơn.

---

## 8. Transformer block hoàn chỉnh

```text
        x ─────────────────────────┐
        │                          │
        ▼                          │
   ┌──────────┐                    │
   │LayerNorm │                    │
   └────┬─────┘                    │
        ▼                          │
   ┌──────────────────┐            │
   │ Masked Multi-Head│            │
   │ Self-Attention   │            │
   └────┬─────────────┘            │
        ▼                          │
       (+) ◀───────────────────────┘   x = x + Attn(LN(x))
        │
        ├──────────────────────────┐
        ▼                          │
   ┌──────────┐                    │
   │LayerNorm │                    │
   └────┬─────┘                    │
        ▼                          │
   ┌──────────────────┐            │
   │ MLP:             │            │
   │ Linear d → 4d    │            │
   │ GELU             │            │
   │ Linear 4d → d    │            │
   └────┬─────────────┘            │
        ▼                          │
       (+) ◀───────────────────────┘   x = x + MLP(LN(x))
        │
        ▼
```

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # giao tiếp
        x = x + self.mlp(self.ln_2(x))    # tính toán
        return x
```

**Residual stream:** hãy tưởng tượng $x$ là một "dòng chảy" thông tin chạy thẳng từ đầu đến cuối mạng. Mỗi block **đọc** từ dòng chảy và **cộng thêm** đóng góp của mình vào. Đây là góc nhìn quan trọng trong nghiên cứu interpretability.

**Pre-norm vs post-norm:** paper gốc đặt LayerNorm **sau** phép cộng residual; GPT-2 trở đi đặt **trước** (pre-norm) → huấn luyện ổn định hơn nhiều.

---

## 9. Đầu ra, loss và huấn luyện song song

```text
   Input:   [Hà]   [Nội]   [là]   [thủ]
              │      │       │      │       qua N Transformer block
              ▼      ▼       ▼      ▼
   Logits:  [...]  [...]   [...]  [...]     mỗi vị trí → vector vocab_size
              │      │       │      │
   Target:  Nội     là      thủ    đô       (input dịch trái 1 vị trí)
              │      │       │      │
   Loss:    ℓ₁  +   ℓ₂  +   ℓ₃  +  ℓ₄   → trung bình
```

**Điểm mấu chốt:** nhờ causal mask, **một** lần forward trên chuỗi $T$ token cho ra **$T$ bài toán dự đoán** cùng lúc. Đây là lý do Transformer huấn luyện hiệu quả hơn RNN rất nhiều.

```python
logits = model(idx)                                          # (B, T, vocab)
loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
```

---

## 10. Mã hóa vị trí: sinusoidal, learned, RoPE

| Cách | Ý tưởng | Dùng ở |
|---|---|---|
| **Sinusoidal** | vector cố định từ hàm sin/cos nhiều tần số | paper gốc 2017 |
| **Learned** | bảng `nn.Embedding(max_len, d)` học được | GPT-2, GPT-3 |
| **RoPE** ⭐ | **xoay** vector Q, K theo góc tỉ lệ với vị trí | LLaMA, Qwen, Mistral, hầu hết LLM hiện đại |

### RoPE — Rotary Position Embedding

Chia vector thành các cặp số, mỗi cặp coi như một điểm 2D, **xoay** một góc $m\theta$ với $m$ là vị trí:

```text
   Vị trí 0          Vị trí 1           Vị trí 2
      ↑                 ↗                 →
      q               (xoay θ)         (xoay 2θ)

   Tích vô hướng q_m · k_n chỉ phụ thuộc vào GÓC LỆCH (m − n)θ
   → attention tự nhiên cảm nhận được KHOẢNG CÁCH TƯƠNG ĐỐI giữa 2 token
```

Ưu điểm: mã hóa vị trí **tương đối**, dễ mở rộng context dài hơn lúc huấn luyện.

---

## 11. Các họ Transformer

```text
   Encoder-only             Decoder-only ⭐             Encoder-Decoder
   (BERT)                   (GPT, Claude, Gemini,      (T5, Transformer gốc)
                             LLaMA...)
   ┌─────────┐              ┌─────────┐                ┌─────────┐   ┌─────────┐
   │ attention│             │ attention│               │ Encoder │──▶│ Decoder │
   │ 2 chiều  │             │ 1 chiều  │               │ 2 chiều │   │ 1 chiều │
   │(nhìn cả  │             │ (causal) │               └─────────┘   │+ cross- │
   │ trước &  │             └─────────┘                   input      │ attention│
   │ sau)     │                                                      └─────────┘
   └─────────┘                                                         output
   Hiểu văn bản:            Sinh văn bản:              Dịch, tóm tắt:
   phân loại, tìm kiếm      chatbot, viết code         input → output
```

Vì sao decoder-only thắng? Đơn giản, huấn luyện được trên **mọi** văn bản (chỉ cần đoán token tiếp theo), và mọi bài toán đều diễn đạt được dưới dạng "sinh văn bản tiếp theo".

---

## 12. Đếm tham số GPT-2

GPT-2 small: `vocab = 50257`, `context = 1024`, `d = 768`, `n_layer = 12`, `n_head = 12`

```text
   Thành phần                         Công thức                    Số tham số
   ─────────────────────────────────────────────────────────────────────────
   Token embedding (wte)              50257 × 768                  38.6 M
   Position embedding (wpe)           1024 × 768                    0.8 M
   Mỗi block:
     Attention (Q,K,V,O)              4 × 768²                      2.4 M
     MLP (768→3072→768)               2 × 768 × 3072                4.7 M
     (LayerNorm, bias: không đáng kể)
   × 12 block                         12 × 7.1 M                   85.0 M
   LM head                            dùng chung với wte (weight tying)
   ─────────────────────────────────────────────────────────────────────────
   TỔNG                                                           ≈ 124 M
```

> 💡 Nhận xét: MLP chiếm ~2/3 tham số mỗi block. Với LLM lớn, phần lớn tham số nằm ở các MLP.

---

## 13. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [3Blue1Brown — Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) | Toàn cảnh, embedding, dự đoán token tiếp theo |
| 2 | ⭐ [3Blue1Brown — Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) | Q, K, V, mask, multi-head — trực quan tuyệt vời |
| 3 | [3Blue1Brown — How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8) | MLP trong Transformer lưu kiến thức thế nào |
| 4 | ⭐⭐ [Karpathy — Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | **Video quan trọng nhất cả lộ trình.** Code GPT từ đầu (2 giờ) |
| 5 | ⭐ [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE từ đầu, các vấn đề do tokenization gây ra |
| 6 | [Umar Jamil — Attention is all you need (giải thích + toán)](https://www.youtube.com/watch?v=bCz4OMemCcA) | Đi qua paper gốc chi tiết |
| 7 | [Umar Jamil — Coding a Transformer from scratch](https://www.youtube.com/watch?v=ISNdQcPhsts) | Code encoder-decoder đầy đủ (bài toán dịch) |

### Công cụ trực quan (tương tác)

- 🎮 [LLM Visualization (bbycroft)](https://bbycroft.net/llm) — xem 3D từng phép tính trong GPT
- 🎮 [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — chạy GPT-2 ngay trên trình duyệt
- 🎮 [Tiktokenizer](https://tiktokenizer.vercel.app) — xem cách các tokenizer chia văn bản

### Đọc thêm

- 📝 ⭐ [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — giải thích bằng hình nổi tiếng nhất
- 📝 [Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- 📝 [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — paper gốc kèm code từng dòng
- 📝 [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- 💻 [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) · [github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- 📘 [Sebastian Raschka — LLMs from Scratch (code của sách)](https://github.com/rasbt/LLMs-from-scratch)
- 📄 Paper: ⭐ [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) · [BPE (2015)](https://arxiv.org/abs/1508.07909) · [RoFormer / RoPE (2021)](https://arxiv.org/abs/2104.09864)

---

## 14. Checklist & thực hành

### Lý thuyết

- [ ] Vẽ lại sơ đồ toàn cảnh GPT (mục 1) bằng tay, ghi shape ở mỗi bước
- [ ] Chạy tay thuật toán BPE trên một chuỗi ngắn
- [ ] Giải thích vì sao LLM kém đếm chữ cái
- [ ] Giải thích Q, K, V bằng lời của mình
- [ ] Viết công thức attention, giải thích vì sao chia $\sqrt{d_k}$
- [ ] Giải thích causal mask và vì sao dùng $-\infty$
- [ ] Giải thích vì sao 1 lần forward cho ra $T$ bài toán dự đoán
- [ ] Hiểu residual stream, pre-norm
- [ ] Tự đếm được 124M tham số của GPT-2
- [ ] Hiểu trực giác của RoPE

### Thực hành (lưu trong `code/`)

- [ ] Theo *Let's build GPT* → GPT nhỏ trên Shakespeare (`code/nanogpt/`)
- [ ] Theo *GPT Tokenizer* → tự viết BPE tokenizer (`code/bpe/`), huấn luyện trên văn bản tiếng Việt
- [ ] Huấn luyện GPT nhỏ trên văn bản tiếng Việt (truyện Kiều, thơ, truyện ngắn...)
- [ ] Trực quan hóa ma trận attention của một vài head bằng `matplotlib.imshow`
- [ ] Kiểm tra: loss ban đầu có ≈ ln(vocab_size) không?
- [ ] **Thử thách:** viết lại toàn bộ GPT mà không nhìn code mẫu

### Tiêu chí hoàn thành

✅ Viết được GPT từ đầu trong ~1 buổi; vẽ được sơ đồ một Transformer block bằng tay và giải thích từng thành phần.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
