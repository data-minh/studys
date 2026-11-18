# Giai đoạn 3 — PyTorch & Mô hình ngôn ngữ đầu tiên

**Thời lượng:** 3–4 tuần
**Mục tiêu:** Thành thạo PyTorch và hiểu bài toán **language modeling** — dự đoán token tiếp theo. Đây là bài toán mà GPT, Claude, Gemini giải quyết.

```text
                                            Bạn đang ở đây
                                                  ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [PyTorch: những gì cần biết](#1-pytorch-những-gì-cần-biết)
2. [Language model là gì?](#2-language-model-là-gì)
3. [Tokenization: biến chữ thành số](#3-tokenization-biến-chữ-thành-số)
4. [Mô hình bigram — language model đơn giản nhất](#4-mô-hình-bigram--language-model-đơn-giản-nhất)
5. [Embedding: biến token thành vector ý nghĩa](#5-embedding-biến-token-thành-vector-ý-nghĩa)
6. [MLP language model (Bengio 2003)](#6-mlp-language-model-bengio-2003)
7. [Đánh giá: loss và perplexity](#7-đánh-giá-loss-và-perplexity)
8. [Sinh văn bản (sampling)](#8-sinh-văn-bản-sampling)
9. [RNN và LSTM](#9-rnn-và-lstm)
10. [Vì sao Transformer thay thế RNN?](#10-vì-sao-transformer-thay-thế-rnn)
11. [CNN — nắm cơ bản](#11-cnn--nắm-cơ-bản)
12. [Video bài giảng & tài liệu](#12-video-bài-giảng--tài-liệu)
13. [Checklist & thực hành](#13-checklist--thực-hành)

---

## 1. PyTorch: những gì cần biết

PyTorch = **NumPy chạy được trên GPU** + **autograd** (micrograd phiên bản công nghiệp).

### Tensor & GPU

```python
import torch
x = torch.randn(32, 784)                 # giống np.random.randn
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                         # chuyển lên GPU
```

### Autograd

```python
w = torch.randn(3, requires_grad=True)   # đánh dấu: cần tính gradient
loss = (w ** 2).sum()
loss.backward()                          # backprop tự động
print(w.grad)                            # = 2w
```

### `nn.Module` — cách tổ chức mô hình

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # tự tạo W, b và đăng ký làm tham số
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

### Vòng lặp huấn luyện chuẩn — **thuộc lòng đoạn này**

```python
model = MLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for xb, yb in train_loader:                  # 1. lấy mini-batch
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                       # 2. forward
        loss = F.cross_entropy(logits, yb)       # 3. loss
        optimizer.zero_grad()                    # 4. xóa gradient cũ (vì grad cộng dồn!)
        loss.backward()                          # 5. backward
        optimizer.step()                         # 6. cập nhật tham số
```

```text
   ┌─────────┐   ┌─────────┐   ┌──────┐   ┌───────────┐   ┌──────────┐   ┌──────┐
   │ batch   │──▶│ forward │──▶│ loss │──▶│ zero_grad │──▶│ backward │──▶│ step │──┐
   └─────────┘   └─────────┘   └──────┘   └───────────┘   └──────────┘   └──────┘  │
        ▲                                                                          │
        └──────────────────────────────────────────────────────────────────────────┘
```

> 🔗 **Liên hệ GPT:** Code huấn luyện nanoGPT có **đúng 6 bước này**, chỉ thêm các tối ưu như mixed precision, gradient accumulation, learning rate schedule.

### `Dataset` & `DataLoader`

```python
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(ds, batch_size=64, shuffle=True)
```

---

## 2. Language model là gì?

**Language model** = mô hình gán xác suất cho chuỗi văn bản. Bằng quy tắc nhân xác suất, ta tách thành **các bài toán dự đoán token tiếp theo**:

$$P(w_1, w_2, \dots, w_n) = P(w_1)\cdot P(w_2 \mid w_1)\cdot P(w_3 \mid w_1, w_2)\cdots = \prod_{t=1}^{n} P(w_t \mid w_{<t})$$

```text
   Ví dụ: P("con mèo ngồi trên ghế")

   = P("con")
   × P("mèo"  | "con")
   × P("ngồi" | "con mèo")
   × P("trên" | "con mèo ngồi")
   × P("ghế"  | "con mèo ngồi trên")
```

### Sinh văn bản tự hồi quy (autoregressive)

Mô hình dự đoán **một token**, nối vào đầu vào, rồi dự đoán tiếp:

```text
   Bước 1:  "Hà Nội là"                    ──▶ [ LM ] ──▶ "thủ"
   Bước 2:  "Hà Nội là thủ"                ──▶ [ LM ] ──▶ "đô"
   Bước 3:  "Hà Nội là thủ đô"             ──▶ [ LM ] ──▶ "của"
   Bước 4:  "Hà Nội là thủ đô của"         ──▶ [ LM ] ──▶ "Việt"
   ...                                                    ▲
                                        ChatGPT "gõ từng chữ" chính vì thế!
```

### Tự giám sát: dữ liệu tự sinh nhãn

Một câu văn cho ra **nhiều** mẫu huấn luyện miễn phí — không cần người gán nhãn:

```text
   Văn bản:  [Hà] [Nội] [là] [thủ] [đô]

   input  →  target
   [Hà]                  →  Nội
   [Hà Nội]              →  là
   [Hà Nội là]           →  thủ
   [Hà Nội là thủ]       →  đô
```

> 🔗 **Liên hệ GPT:** Đây là **toàn bộ** mục tiêu pretraining của GPT/Claude/Gemini. Để đoán tốt token tiếp theo trên toàn bộ internet, mô hình buộc phải học ngữ pháp, kiến thức, suy luận...

---

## 3. Tokenization: biến chữ thành số

Mạng nơ-ron chỉ xử lý số → cần chia văn bản thành **token** và gán mỗi token một ID.

```text
   Văn bản: "xin chào"

   Mức ký tự (char)      Mức từ (word)          Mức subword (BPE) ⭐
   x|i|n| |c|h|à|o       xin|chào               xin| chào
   [23,8,13,0,...]       [1052, 873]            [7431, 21680]

   + từ điển nhỏ (~100)  + chuỗi ngắn           + cân bằng cả hai
   - chuỗi rất dài       - từ điển khổng lồ     + xử lý được từ mới
                         - không xử lý từ mới   → GPT, Claude, Gemini dùng
```

Giai đoạn này dùng **mức ký tự** cho đơn giản. BPE sẽ học chi tiết ở Giai đoạn 4.

---

## 4. Mô hình bigram — language model đơn giản nhất

**Giả định:** token tiếp theo chỉ phụ thuộc **một** token ngay trước.

$$P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-1})$$

### Cách 1: Đếm

Đếm số lần mỗi cặp ký tự xuất hiện trong dữ liệu (ví dụ danh sách tên người), rồi chuẩn hóa từng hàng thành xác suất:

```text
                 ký tự tiếp theo
               .    a    b    c   ...   n  ...
         .  ┌────────────────────────────────┐
   ký    a  │ 0.20 0.01 0.02 0.01 ...  0.16  │  ← hàng "a": sau chữ 'a',
   tự    b  │ 0.04 0.12 0.01 0.00 ...  0.00  │    xác suất ra '.' (kết thúc)
   hiện  c  │ ...                            │    là 0.20, ra 'n' là 0.16
   tại   ⋮  │                                │
            └────────────────────────────────┘
            ma trận 27×27 (26 chữ + ký tự '.' đánh dấu đầu/cuối)
```

### Cách 2: Mạng nơ-ron (cho kết quả tương đương)

```text
   ký tự 'a'  ──▶  one-hot (27)  ──▶  @ W (27×27)  ──▶  logits (27)  ──▶  softmax  ──▶  P(ký tự tiếp)
                   [0,1,0,...,0]      (học bằng
                                       gradient descent)
```

Khi huấn luyện đủ lâu, `softmax(W)` hội tụ về đúng bảng đếm ở cách 1! **Nhưng** cách mạng nơ-ron **mở rộng được** — ta có thể thay `W` bằng mạng phức tạp nhìn nhiều token hơn.

**Hạn chế:** chỉ nhìn 1 ký tự → sinh ra tên vô nghĩa như "anaelirh".

---

## 5. Embedding: biến token thành vector ý nghĩa

**Embedding** = bảng tra cứu: mỗi token ID ↔ một vector học được.

```text
   Bảng embedding E  shape (vocab_size, d)

   token ID          vector d chiều (học được)
      0  "."    →  [ 0.12, -0.53,  0.88, ...]
      1  "a"    →  [-0.41,  0.27,  0.03, ...]
      2  "b"    →  [ 0.95, -0.11, -0.72, ...]
      ⋮
   Tra cứu E[token_id]  ≡  one_hot(token_id) @ E    (nhưng nhanh hơn nhiều)
```

**Điều kỳ diệu:** sau huấn luyện, các token có ý nghĩa giống nhau có vector **gần nhau**, và các hướng trong không gian mang ý nghĩa:

```text
                    ▲
         nữ hoàng ● │         vua − đàn ông + phụ nữ ≈ nữ hoàng
              ↑     │
     (+giới   │     │  ● vua
      tính nữ)│     │  ↑
        phụ nữ ●    │  │ (+hoàng gia)
                    │  ● đàn ông
         ───────────┼──────────────▶
                    │     ● Hà Nội ─────▶ ● Việt Nam
                    │     ● Paris  ─────▶ ● Pháp
                    │       (cùng hướng "thủ đô của")
```

> 🔗 **Liên hệ GPT:** Lớp đầu tiên của GPT là `nn.Embedding(vocab_size, d_model)`. GPT-2: `(50257, 768)`. Lớp cuối thường **dùng chung** ma trận này (weight tying).

---

## 6. MLP language model (Bengio 2003)

Nhìn **nhiều** token trước đó (context window cố định, ví dụ 3):

```text
   context: "e"  "m"  "m"             → dự đoán "a"  (trong tên "emma")
             │    │    │
             ▼    ▼    ▼
          ┌────┬────┬────┐
          │ E  │ E  │ E  │   tra embedding (mỗi ký tự → vector 10 chiều)
          └─┬──┴─┬──┴─┬──┘
            └────┼────┘
                 ▼
          nối lại (30 chiều)
                 ▼
          ┌──────────────┐
          │ Linear + tanh│   lớp ẩn (200 nơ-ron)
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │   Linear     │   → 27 logits
          └──────┬───────┘
                 ▼
             softmax → P(ký tự tiếp theo)
```

**Hạn chế:** context window **cố định** và nhỏ. Tăng context → số tham số lớp đầu tăng tuyến tính, và vị trí nào cũng dùng trọng số riêng (không chia sẻ).

---

## 7. Đánh giá: loss và perplexity

**Negative log-likelihood (NLL)** trung bình trên tập test:

$$L = -\frac{1}{N}\sum_{t=1}^{N}\log P(w_t \mid w_{<t})$$

**Perplexity** = $e^{L}$ — trực giác: *"mô hình phân vân trung bình giữa bao nhiêu lựa chọn?"*

| Mô hình | Loss | Perplexity | Ý nghĩa |
|---|---|---|---|
| Đoán đều 27 ký tự | $\ln 27 = 3.30$ | 27 | phân vân giữa 27 lựa chọn |
| Bigram | ~2.45 | ~11.6 | |
| MLP | ~2.1 | ~8.2 | |
| Hoàn hảo | 0 | 1 | luôn chắc chắn và đúng |

> 💡 **Mẹo debug:** loss ban đầu **phải** ≈ $\ln(\text{vocab\_size})$. Nếu cao hơn nhiều → khởi tạo sai (logits quá lớn).

---

## 8. Sinh văn bản (sampling)

Từ phân phối $P(\text{token tiếp})$, chọn token như thế nào?

```text
   P:  "ghế" 0.40 │ "bàn" 0.25 │ "mái" 0.15 │ "sofa" 0.10 │ ... 0.10

   Greedy (argmax)       → luôn "ghế"  — lặp lại, nhàm chán
   Sampling              → rút thăm theo xác suất — đa dạng
   Temperature T         → chia logits cho T trước softmax
   Top-k (k=2)           → chỉ rút trong {"ghế", "bàn"}
   Top-p (p=0.8)         → rút trong nhóm nhỏ nhất có tổng xác suất ≥ 0.8
```

```python
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

---

## 9. RNN và LSTM

### RNN — xử lý chuỗi độ dài bất kỳ

Ý tưởng: duy trì một **trạng thái ẩn** $h_t$ tóm tắt mọi thứ đã đọc, cập nhật sau mỗi token, **dùng chung trọng số** ở mọi bước.

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

```text
   Dạng gọn            Dạng "trải ra" theo thời gian

    ┌──┐                 h₀      h₁      h₂      h₃
    │  │ h               │       ▲│      ▲│      ▲│
    ▼  │                 └─▶[RNN]─┴─▶[RNN]─┴─▶[RNN]─┴─▶ ...
   [RNN]─┘                   ▲       ▲       ▲
    ▲                        │       │       │
    x                       x₁      x₂      x₃
                           "con"   "mèo"   "ngồi"
                        (cùng một W ở mọi bước)
```

### Vấn đề của RNN

1. **Vanishing gradient qua thời gian:** gradient từ token 100 về token 1 phải nhân qua 100 lần $W_{hh}$ → biến mất. RNN "quên" thông tin xa.
2. **Tuần tự:** phải tính $h_1$ xong mới tính được $h_2$ → **không song song hóa được** trên GPU.
3. **Nút thắt cổ chai:** toàn bộ ngữ cảnh bị nén vào một vector $h_t$ cố định.

### LSTM — RNN có "cổng" và "băng chuyền trí nhớ"

```text
            ┌───────────────────────────────────────┐
   c_{t-1} ─┼──(×)──────────(+)─────────────────────┼─▶ c_t   ← cell state: "băng chuyền"
            │   ▲            ▲                      │          thông tin chảy qua gần như
            │ forget      input                     │          nguyên vẹn
            │  gate        gate × candidate         │
            │   │            │                      │
   h_{t-1} ─┼───┴────────────┴───────── output gate ┼─▶ h_t
            └───────────────────────────────────────┘
                              ▲
                              x_t
   forget gate: quên gì?  input gate: nhớ thêm gì?  output gate: xuất ra gì?
```

Cell state hoạt động giống **residual connection** — giúp gradient chảy xa hơn. LSTM thống trị NLP giai đoạn 2014–2017 (Google Translate cũ).

---

## 10. Vì sao Transformer thay thế RNN?

```text
   RNN: thông tin đi TUẦN TỰ            Transformer: MỌI token nhìn TRỰC TIẾP
                                        mọi token trước nó (attention)

   w₁ → w₂ → w₃ → w₄ → w₅                w₁   w₂   w₃   w₄   w₅
   (w₅ muốn biết w₁ phải đi qua          ▲╲   ▲╲  ▲╲   ▲    │
    4 bước, thông tin bị pha loãng)      │ ╲──┼─╲─┼─╲──┼────┤
                                         └────┴───┴────┴────┘
                                         (1 bước, tính song song)
```

| | RNN/LSTM | Transformer |
|---|---|---|
| Song song hóa khi huấn luyện | ❌ tuần tự | ✅ toàn bộ chuỗi cùng lúc |
| Phụ thuộc xa | khó (vanishing) | dễ (kết nối trực tiếp) |
| Mở rộng quy mô (scaling) | kém | **cực tốt** → sinh ra LLM |
| Chi phí theo độ dài chuỗi | $O(n)$ | $O(n^2)$ (nhược điểm) |

→ Giai đoạn 4 sẽ học chi tiết Transformer.

---

## 11. CNN — nắm cơ bản

Chủ yếu cho ảnh. Ý tưởng: một **bộ lọc (kernel)** nhỏ trượt khắp ảnh, **dùng chung trọng số** ở mọi vị trí, phát hiện mẫu cục bộ (cạnh, góc...).

```text
   Ảnh đầu vào (5×5)        Kernel 3×3         Feature map (3×3)
   ┌─────────────────┐      ┌───────┐
   │[1  0  1] 0  1   │      │ 1 0 1 │          ┌───────────┐
   │[0  1  0] 1  0   │  ⊛   │ 0 1 0 │    =     │ 4  ·  ·   │
   │[1  0  1] 0  1   │      │ 1 0 1 │          │ ·  ·  ·   │
   │ 0  1  0  1  0   │      └───────┘          │ ·  ·  ·   │
   │ 1  0  1  0  1   │   trượt từng bước        └───────────┘
   └─────────────────┘   nhân từng phần tử rồi cộng
```

> 🔗 **Liên hệ Gemini:** Mô hình đa phương thức xử lý ảnh bằng **Vision Transformer** (chia ảnh thành các ô, coi mỗi ô là một token) — sẽ học ở Giai đoạn 7. Ý tưởng WaveNet (makemore Part 5) cũng là tích chập 1 chiều.

---

## 12. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [Karpathy — makemore Part 1: bigram](https://www.youtube.com/watch?v=PaCmpygFfXo) | Language model đơn giản nhất, đếm & mạng nơ-ron, sampling, NLL loss |
| 2 | ⭐ [Karpathy — makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | Embedding, MLP language model, train/val/test, tìm learning rate |
| 3 | ⭐ [Karpathy — makemore Part 3: Activations, Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | Khởi tạo, chẩn đoán mạng, BatchNorm — **cực kỳ thực tế** |
| 4 | [Karpathy — makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Tự viết backward cho toàn bộ mạng bằng tensor |
| 5 | [Karpathy — makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | Kiến trúc phân cấp, tổ chức code giống PyTorch |
| 6 | [Stanford CS224N (Spring 2024)](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) | Bài 1–2 (word vectors), bài về RNN/LSTM — tham khảo |

### Đọc thêm

- 📝 [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — **làm trước khi xem makemore**
- 💻 [github.com/karpathy/makemore](https://github.com/karpathy/makemore)
- 📝 [Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — bài blog kinh điển
- 📝 [Colah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — giải thích LSTM bằng hình rõ nhất
- 📝 [Jay Alammar — The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) — embedding bằng hình
- 📝 [CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- 📘 [Dive into Deep Learning — RNN](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- 📄 Paper: [Bengio et al. 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) · [Word2vec (2013)](https://arxiv.org/abs/1301.3781)

---

## 13. Checklist & thực hành

### Lý thuyết

- [ ] Viết được vòng lặp huấn luyện PyTorch 6 bước không cần tra cứu
- [ ] Giải thích language model bằng quy tắc nhân xác suất
- [ ] Hiểu sinh văn bản tự hồi quy và vì sao pretraining là tự giám sát
- [ ] So sánh tokenization mức ký tự / từ / subword
- [ ] Giải thích embedding và vì sao `E[id] ≡ one_hot(id) @ E`
- [ ] Tính perplexity từ loss; biết loss ban đầu phải ≈ ln(vocab_size)
- [ ] Phân biệt greedy, sampling, temperature, top-k, top-p
- [ ] Nêu 3 vấn đề của RNN và vì sao Transformer giải quyết được

### Thực hành (lưu trong `code/`)

- [ ] Hoàn thành PyTorch *Learn the Basics*
- [ ] Theo makemore Part 1–3: bigram → MLP → BatchNorm (gõ lại code, không copy)
- [ ] Huấn luyện mô hình sinh **tên tiếng Việt** từ danh sách tên tự thu thập
- [ ] Viết char-level RNN (dùng `nn.RNN` hoặc tự viết) sinh văn bản, so sánh với MLP
- [ ] Thí nghiệm temperature: sinh 10 mẫu với T = 0.5, 1.0, 1.5 và nhận xét

### Tiêu chí hoàn thành

✅ Tự viết vòng lặp huấn luyện PyTorch không cần tra cứu; giải thích được perplexity; mô hình sinh tên tiếng Việt tạo ra tên nghe hợp lý.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
