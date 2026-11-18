# Giai đoạn 2 — Mạng nơ-ron & Backpropagation ⭐

**Thời lượng:** 3–4 tuần
**Mục tiêu:** Hiểu tận gốc **backpropagation** — thuật toán giúp mọi mạng nơ-ron (kể cả GPT) học được. Sau giai đoạn này, "deep learning" sẽ không còn là hộp đen.

```text
                              Bạn đang ở đây
                                   ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Vì sao mô hình tuyến tính chưa đủ?](#1-vì-sao-mô-hình-tuyến-tính-chưa-đủ)
2. [Nơ-ron và mạng nhiều lớp (MLP)](#2-nơ-ron-và-mạng-nhiều-lớp-mlp)
3. [Hàm kích hoạt](#3-hàm-kích-hoạt-activation-functions)
4. [Forward pass: dữ liệu chảy qua mạng](#4-forward-pass-dữ-liệu-chảy-qua-mạng)
5. [Backpropagation](#5-backpropagation--trái-tim-của-deep-learning)
6. [Backprop cho lớp tuyến tính (dạng ma trận)](#6-backprop-cho-lớp-tuyến-tính-dạng-ma-trận)
7. [Khởi tạo trọng số & vanishing/exploding gradient](#7-khởi-tạo-trọng-số--vanishingexploding-gradient)
8. [Optimizer: từ SGD đến AdamW](#8-optimizer-từ-sgd-đến-adamw)
9. [Normalization: BatchNorm và LayerNorm](#9-normalization-batchnorm-và-layernorm)
10. [Dropout và residual connection](#10-dropout-và-residual-connection)
11. [Video bài giảng & tài liệu](#11-video-bài-giảng--tài-liệu)
12. [Checklist & thực hành](#12-checklist--thực-hành)

---

## 1. Vì sao mô hình tuyến tính chưa đủ?

Mô hình tuyến tính chỉ vẽ được **đường thẳng** làm biên quyết định. Bài toán XOR kinh điển:

```text
   x₂                             Không có đường thẳng nào tách được
   1 │  ●(0,1)=1      ○(1,1)=0    ● và ○!
     │
     │                            → Cần biên quyết định CONG
   0 │  ○(0,0)=0      ●(1,0)=1    → Cần mạng nơ-ron nhiều lớp
     └──────────────────── x₁
        0               1
```

**Xếp chồng nhiều lớp tuyến tính cũng không giúp gì:**

$$W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x} = W'\mathbf{x}$$

→ vẫn là một phép tuyến tính! Cần chèn **hàm phi tuyến** (activation) giữa các lớp.

---

## 2. Nơ-ron và mạng nhiều lớp (MLP)

### Một nơ-ron

```text
   x₁ ──w₁──╲
   x₂ ──w₂───▶ ( Σ wᵢxᵢ + b ) ──▶ [ hàm kích hoạt φ ] ──▶ output
   x₃ ──w₃──╱         z                                     a = φ(z)
               tổng có trọng số
```

### Multi-Layer Perceptron (MLP)

```text
   Lớp vào         Lớp ẩn 1        Lớp ẩn 2        Lớp ra
   (784)           (128)           (64)            (10)

    ○ ─────────╲    ○ ────────╲     ○ ───────╲     ○  P("0")
    ○ ──────────╳── ○ ─────────╳─── ○ ────────╳─── ○  P("1")
    ○ ──────────╳── ○ ─────────╳─── ○ ────────╳─── ○  ...
    ⋮          ╱    ⋮         ╱     ⋮        ╱     ○  P("9")
    ○ ────────╱     ○ ───────╱      ○ ──────╱

          W₁,b₁  ReLU     W₂,b₂  ReLU      W₃,b₃  softmax
         (128×784)       (64×128)         (10×64)
```

$$\mathbf{h}_1 = \text{ReLU}(W_1\mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(W_2\mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{\mathbf{p}} = \text{softmax}(W_3\mathbf{h}_2 + \mathbf{b}_3)$$

Số tham số: $784·128 + 128 + 128·64 + 64 + 64·10 + 10 = 109\,386$.

**Định lý xấp xỉ vạn năng (Universal Approximation):** một MLP đủ rộng có thể xấp xỉ **mọi hàm liên tục**. Mạng sâu (nhiều lớp) làm việc này hiệu quả hơn mạng rộng.

**Mỗi lớp học đặc trưng trừu tượng hơn:**

```text
   Pixel  ──▶  Nét, cạnh  ──▶  Hình dạng (vòng, móc)  ──▶  Chữ số
   (lớp vào)   (lớp 1)         (lớp 2)                     (lớp ra)
```

> 🔗 **Liên hệ GPT:** Mỗi Transformer block trong GPT chứa **một MLP 2 lớp** (gọi là feed-forward network). GPT-3 có 96 block như vậy. Nghiên cứu cho thấy các MLP này là nơi mô hình "lưu trữ kiến thức".

---

## 3. Hàm kích hoạt (Activation Functions)

```text
     Sigmoid                 Tanh                   ReLU ⭐                 GELU (GPT dùng)
  1 ┤      ●●●●         1 ┤      ●●●●                │       ╱              │       ╱
    │    ●                │    ●                      │      ╱               │      ╱
.5  ┤   ●               0 ┼───●───────                │     ╱                │     ╱
    │  ●                  │  ●                        │    ╱                 │    ╱
  0 ┤●●                -1 ┤●●                    0 ───●───╱           0 ────╲_╱
    └───────              └───────                    └──────                └──────
   (0, 1)                (-1, 1)                  max(0, x)            ≈ x·Φ(x), mượt
```

| Hàm | Công thức | Ưu | Nhược | Dùng ở đâu |
|---|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | output là xác suất | gradient ≈ 0 ở 2 đầu (bão hòa) | lớp ra nhị phân, cổng LSTM |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | tâm tại 0 | vẫn bão hòa | RNN, LSTM |
| **ReLU** | $\max(0, x)$ | đơn giản, nhanh, không bão hòa khi $x>0$ | nơ-ron "chết" nếu luôn $x<0$ | CNN, MLP phổ biến |
| **GELU** | $x \cdot \Phi(x)$ | mượt, hiệu quả cho Transformer | tính chậm hơn chút | **GPT, BERT** |
| SwiGLU | $\text{Swish}(xW) \odot (xV)$ | tốt hơn nữa | thêm tham số | **LLaMA, các LLM hiện đại** |

---

## 4. Forward pass: dữ liệu chảy qua mạng

Luôn theo dõi **shape** khi dữ liệu đi qua mạng (batch $B = 32$):

```text
   X              (32, 784)
   │  @ W₁ᵀ + b₁   W₁: (128, 784)
   ▼
   Z₁             (32, 128)
   │  ReLU
   ▼
   H₁             (32, 128)
   │  @ W₂ᵀ + b₂   W₂: (10, 128)
   ▼
   logits         (32, 10)
   │  softmax + cross-entropy với nhãn y (32,)
   ▼
   loss           ()  ← một con số duy nhất
```

---

## 5. Backpropagation — trái tim của deep learning

### Vấn đề

Loss là một con số. Có hàng trăm nghìn (hoặc hàng tỷ) tham số. Cần biết $\frac{\partial L}{\partial w}$ cho **từng** tham số $w$.

- Cách ngây thơ: thay đổi từng $w$ một chút, đo loss thay đổi → cần hàng tỷ lần chạy forward. **Quá chậm.**
- **Backpropagation:** tính **tất cả** gradient chỉ với **1 lần forward + 1 lần backward**.

### Ý tưởng: đồ thị tính toán + chain rule cục bộ

Mỗi phép toán là một **nút**. Mỗi nút chỉ cần biết **đạo hàm cục bộ** của nó. Gradient chảy ngược từ loss, nhân dồn qua từng nút.

```text
   Ví dụ: L = (a·b + c)²    với a=2, b=-3, c=10

   FORWARD (xuôi) ──────────────────────────────────────────▶

    a=2  ──╮
           ├──[ × ]── d=-6 ──╮
    b=-3 ──╯                 ├──[ + ]── e=4 ──[ ^2 ]── L=16
    c=10 ────────────────────╯

   ◀────────────────────────────────────────── BACKWARD (ngược)

   ∂L/∂a = -24 ◀─╮
                 ├── [×]: ∂L/∂d = 8 ◀─╮
   ∂L/∂b = 16  ◀─╯                    ├── [+]: ∂L/∂e = 8 ◀── [^2]: ∂L/∂L = 1
   ∂L/∂c = 8   ◀──────────────────────╯       (2e = 8)
```

Từng bước:
- $\frac{\partial L}{\partial e} = 2e = 8$
- Nút **cộng** chuyển gradient nguyên vẹn: $\frac{\partial L}{\partial d} = 8$, $\frac{\partial L}{\partial c} = 8$
- Nút **nhân** hoán đổi đầu vào: $\frac{\partial L}{\partial a} = 8 \cdot b = -24$, $\frac{\partial L}{\partial b} = 8 \cdot a = 16$

### Quy tắc cục bộ cần thuộc

| Phép toán | Forward | Backward (gradient gửi về đầu vào) |
|---|---|---|
| Cộng | $z = x + y$ | $\bar{x} = \bar{z}$, $\bar{y} = \bar{z}$ — **phân phát** |
| Nhân | $z = x \cdot y$ | $\bar{x} = \bar{z}\cdot y$, $\bar{y} = \bar{z}\cdot x$ — **hoán đổi** |
| ReLU | $z = \max(0,x)$ | $\bar{x} = \bar{z}$ nếu $x>0$, ngược lại $0$ — **cổng đóng/mở** |
| Tanh | $z = \tanh x$ | $\bar{x} = \bar{z}(1 - z^2)$ |
| Exp | $z = e^x$ | $\bar{x} = \bar{z}\cdot z$ |
| Rẽ nhánh | $x$ dùng ở 2 nơi | $\bar{x} = \bar{x}_1 + \bar{x}_2$ — **cộng dồn** |

(Ký hiệu $\bar{x} = \frac{\partial L}{\partial x}$)

> ⚠️ Quy tắc **rẽ nhánh → cộng dồn** là nguồn bug phổ biến nhất khi tự viết backprop. Đó là lý do PyTorch **cộng dồn** `.grad` và bạn phải gọi `zero_grad()` mỗi bước.

### Đây chính là micrograd

Karpathy xây dựng lớp `Value` lưu: giá trị `data`, gradient `grad`, các nút cha, và hàm `_backward` cục bộ. Gọi `loss.backward()` → sắp xếp topo đồ thị → gọi `_backward` từ loss ngược về. **PyTorch autograd hoạt động y hệt**, chỉ là trên tensor thay vì số đơn.

---

## 6. Backprop cho lớp tuyến tính (dạng ma trận)

Với $Y = XW^\top + \mathbf{b}$, trong đó $X$: `(B, in)`, $W$: `(out, in)`, $Y$: `(B, out)`.
Biết $\bar{Y} = \frac{\partial L}{\partial Y}$ shape `(B, out)`:

$$\bar{X} = \bar{Y}\,W \qquad\qquad \bar{W} = \bar{Y}^\top X \qquad\qquad \bar{\mathbf{b}} = \sum_{\text{batch}} \bar{Y}$$

**Mẹo nhớ — khớp shape:** gradient luôn có **cùng shape** với biến gốc. Chỉ có một cách sắp xếp phép nhân ma trận để shape khớp!

```text
   W̄ phải có shape (out, in)
   Ȳᵀ: (out, B)   @   X: (B, in)   =   (out, in)  ✓
```

---

## 7. Khởi tạo trọng số & vanishing/exploding gradient

Gradient đi qua $n$ lớp = **tích** của $n$ thừa số.

```text
   Mỗi lớp nhân gradient với ~0.5:      Mỗi lớp nhân với ~2:
   1 → 0.5 → 0.25 → ... → 0.5²⁰ ≈ 10⁻⁶  1 → 2 → 4 → ... → 2²⁰ ≈ 10⁶
   VANISHING: lớp đầu không học được    EXPLODING: loss thành NaN
```

**Giải pháp 1 — Khởi tạo đúng thang đo:** giữ phương sai của activation ổn định qua các lớp.
- **Xavier/Glorot** (cho tanh): $W \sim \mathcal{N}(0, \frac{1}{n_{\text{in}}})$
- **Kaiming/He** (cho ReLU): $W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}}})$

```text
   Histogram activation qua 5 lớp:

   Khởi tạo quá nhỏ         Khởi tạo Kaiming ✓        Khởi tạo quá lớn
   lớp 1: ▁▃█▃▁             lớp 1: ▂▅█▅▂              lớp 1: █▁▁▁█
   lớp 3:   █                lớp 3: ▂▅█▅▂              lớp 3: █   █   (bão hòa)
   lớp 5:   |  (≈ 0)         lớp 5: ▂▅█▅▂              lớp 5: █   █
```

**Giải pháp 2:** Normalization (mục 9), residual connection (mục 10), gradient clipping.

---

## 8. Optimizer: từ SGD đến AdamW

```text
   SGD thuần                    SGD + Momentum              Adam
   đi zig-zag                   có "quán tính", lăn mượt     tự điều chỉnh bước
                                                             cho TỪNG tham số
   ●╲  ╱╲  ╱╲                   ●‾‾‾╲___                     ●──────╲___
     ╲╱  ╲╱  ╲╱ ★                       ╲___★                       ★
```

### SGD

$$\theta \leftarrow \theta - \eta\, g$$

### Momentum — nhớ hướng đi trước đó

$$v \leftarrow \beta v + g \qquad \theta \leftarrow \theta - \eta\, v$$

Như quả bóng lăn xuống dốc: tích lũy vận tốc, vượt qua các gờ nhỏ, giảm zig-zag.

### Adam — optimizer mặc định của deep learning

$$m \leftarrow \beta_1 m + (1-\beta_1)\, g \qquad \text{(trung bình gradient — momentum)}$$
$$v \leftarrow \beta_2 v + (1-\beta_2)\, g^2 \qquad \text{(trung bình bình phương gradient)}$$
$$\hat{m} = \frac{m}{1-\beta_1^t},\quad \hat{v} = \frac{v}{1-\beta_2^t} \qquad \text{(hiệu chỉnh bias ban đầu)}$$
$$\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

**Trực giác:** tham số nào có gradient lớn/nhiễu → bước nhỏ lại; tham số có gradient nhỏ → bước lớn hơn. Mặc định $\beta_1 = 0.9$, $\beta_2 = 0.999$ (LLM thường dùng $0.95$).

### AdamW = Adam + weight decay tách rời

$$\theta \leftarrow \theta - \eta\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + \lambda\theta\right)$$

> 🔗 **Liên hệ GPT:** GPT-2/3, LLaMA, và hầu hết LLM được huấn luyện bằng **AdamW**. Lưu ý: Adam lưu thêm $m$ và $v$ cho mỗi tham số → bộ nhớ optimizer gấp 2 lần kích thước mô hình.

---

## 9. Normalization: BatchNorm và LayerNorm

Chuẩn hóa activation về trung bình 0, phương sai 1, rồi cho mô hình học lại thang đo ($\gamma$, $\beta$):

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \qquad y = \gamma\hat{x} + \beta$$

**Khác biệt nằm ở chỗ tính $\mu, \sigma$ theo chiều nào:**

```text
   Tensor activation shape (Batch, Features)

         BatchNorm                          LayerNorm ⭐ (Transformer dùng)
         chuẩn hóa theo CỘT                 chuẩn hóa theo HÀNG
         (qua các mẫu trong batch)          (qua các đặc trưng của 1 mẫu)

          f1  f2  f3  f4                     f1  f2  f3  f4
   mẫu 1 │ ┃ │ ┃ │ ┃ │ ┃ │              mẫu 1 ━━━━━━━━━━━━━━━
   mẫu 2 │ ┃ │ ┃ │ ┃ │ ┃ │              mẫu 2 ━━━━━━━━━━━━━━━
   mẫu 3 │ ┃ │ ┃ │ ┃ │ ┃ │              mẫu 3 ━━━━━━━━━━━━━━━
          ↑ mỗi cột một μ,σ                   ↑ mỗi hàng một μ,σ

   Phụ thuộc batch size;                   Độc lập với batch → phù hợp
   hành vi khác khi train/test             chuỗi độ dài thay đổi, LLM
```

> 🔗 **Liên hệ GPT:** GPT dùng **LayerNorm**, LLaMA dùng **RMSNorm** (bỏ bước trừ trung bình: $\hat{x} = x / \text{RMS}(x)$).

---

## 10. Dropout và residual connection

### Dropout

Khi huấn luyện, **tắt ngẫu nhiên** mỗi nơ-ron với xác suất $p$ (ví dụ 0.1). Mạng buộc phải học các đặc trưng dư thừa, không phụ thuộc vào vài nơ-ron → giảm overfitting. Khi test, dùng toàn bộ nơ-ron.

```text
   Huấn luyện (p=0.5)              Test
   ○ ── ○ ── ○                     ○ ── ○ ── ○
   ○ ── ✗    ○        ──▶          ○ ── ○ ── ○
   ○ ── ○ ── ✗                     ○ ── ○ ── ○
```

### Residual connection (kết nối tắt) — xem trước cho Transformer

$$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$

```text
        x ──────────────────┐
        │                   │ (đường tắt)
        ▼                   │
   ┌─────────┐              │
   │  F(x)   │              │
   │ (lớp)   │              │
   └────┬────┘              │
        ▼                   │
       (+) ◀────────────────┘
        │
        ▼  y = x + F(x)
```

**Vì sao quan trọng:** Nhớ quy tắc "nút cộng phân phát gradient nguyên vẹn"? Đường tắt tạo ra **"đường cao tốc gradient"** từ loss về thẳng các lớp đầu → huấn luyện được mạng hàng trăm lớp.

> 🔗 **Liên hệ GPT:** Mỗi Transformer block có **2 residual connection**. Không có chúng, không thể huấn luyện GPT 96 lớp.

---

## 11. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | Trực quan về nơ-ron, lớp, trọng số |
| 2 | ⭐ [3Blue1Brown — Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Loss, gradient trên mạng nơ-ron |
| 3 | ⭐ [3Blue1Brown — Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | Trực giác backprop |
| 4 | [3Blue1Brown — Neural Networks (cả playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Có thêm video giải tích backprop |
| 5 | ⭐⭐ [Karpathy — Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | **Video quan trọng nhất giai đoạn này.** Tự code autograd từ đầu (2.5 giờ) |
| 6 | [StatQuest — Neural Networks / Deep Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1) | Giải thích từng bước rất chậm, dễ hiểu |
| 7 | [Karpathy — makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Backprop tay qua cả mạng dạng tensor (xem sau Giai đoạn 3) |

### Đọc thêm

- 💻 [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) — code micrograd gốc
- 📝 [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — bài giảng backprop kinh điển
- 📝 [CS231n — Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- 📝 [Karpathy — Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- 📝 [Sebastian Ruder — An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)
- 📝 [Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/) — trực quan hóa tương tác
- 📘 [Dive into Deep Learning — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)
- 📘 [Dive into Deep Learning — Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html)
- 📄 Paper: [Adam (2014)](https://arxiv.org/abs/1412.6980) · [Batch Normalization (2015)](https://arxiv.org/abs/1502.03167) · [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450) · [Kaiming init (2015)](https://arxiv.org/abs/1502.01852) · [ResNet (2015)](https://arxiv.org/abs/1512.03385)

---

## 12. Checklist & thực hành

### Lý thuyết

- [ ] Giải thích vì sao cần hàm phi tuyến (bài toán XOR)
- [ ] Tính tay forward + backward cho đồ thị tính toán nhỏ (như ví dụ mục 5)
- [ ] Thuộc bảng quy tắc gradient cục bộ, hiểu quy tắc "rẽ nhánh → cộng dồn"
- [ ] Suy ra công thức backprop cho lớp tuyến tính bằng cách khớp shape
- [ ] Giải thích vanishing/exploding gradient và cách khởi tạo Kaiming
- [ ] Viết ra công thức Adam, giải thích từng thành phần
- [ ] Phân biệt BatchNorm vs LayerNorm
- [ ] Giải thích vì sao residual connection giúp huấn luyện mạng sâu

### Thực hành (lưu trong `code/`)

- [ ] Xem video micrograd → tự viết lại autograd engine trong `code/micrograd/` (~100 dòng)
- [ ] Dùng micrograd của mình huấn luyện MLP nhỏ giải bài XOR
- [ ] Viết MLP bằng NumPy (tự code backward) phân loại MNIST
- [ ] Viết lại MLP đó bằng PyTorch, so sánh gradient với bản NumPy (`torch.allclose`)
- [ ] Tự cài đặt SGD, Momentum, Adam → vẽ đồ thị loss so sánh
- [ ] Thí nghiệm: khởi tạo quá nhỏ / Kaiming / quá lớn → vẽ histogram activation mỗi lớp

### Tiêu chí hoàn thành

✅ Tính tay được backward của một mạng 2 lớp; micrograd tự viết cho gradient **khớp với PyTorch**.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
