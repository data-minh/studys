# Giai đoạn 0 — Nền tảng Toán & Python

**Thời lượng:** 4–6 tuần
**Mục tiêu:** Nắm vững các công cụ toán học mà *mọi* mạng nơ-ron đều dùng — từ hồi quy tuyến tính đến GPT, Claude, Gemini.

> 💡 Mở file này bằng **Markdown Preview** trong VS Code (`Ctrl+Shift+V`) để xem công thức toán hiển thị đẹp.

```text
Bạn đang ở đây
     ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Vì sao cần toán?](#1-vì-sao-cần-toán)
2. [Vector và tích vô hướng](#2-vector-và-tích-vô-hướng)
3. [Ma trận và phép nhân ma trận](#3-ma-trận-và-phép-nhân-ma-trận)
4. [Tensor, shape và broadcasting](#4-tensor-shape-và-broadcasting)
5. [Đạo hàm và gradient](#5-đạo-hàm-và-gradient)
6. [Quy tắc chuỗi (chain rule)](#6-quy-tắc-chuỗi-chain-rule)
7. [Gradient descent](#7-gradient-descent)
8. [Xác suất, softmax và cross-entropy](#8-xác-suất-softmax-và-cross-entropy)
9. [Video bài giảng & tài liệu](#9-video-bài-giảng--tài-liệu)
10. [Checklist & thực hành](#10-checklist--thực-hành)

---

## 1. Vì sao cần toán?

Một mô hình như GPT, về bản chất, chỉ là **một hàm số khổng lồ** với hàng tỷ tham số:

```text
         ┌──────────────────────────────────────┐
"Hôm nay │                                      │  P("trời") = 0.31
 trời"  →│   f(x; θ)   θ = hàng tỷ con số       │→ P("tôi")  = 0.12
         │   (toàn phép nhân ma trận + hàm      │  P("đẹp")  = 0.08
         │    phi tuyến + softmax)              │  ...
         └──────────────────────────────────────┘
```

Ba nhánh toán tương ứng với ba câu hỏi:

| Câu hỏi | Nhánh toán | Công cụ |
|---|---|---|
| Hàm `f` tính toán thế nào? | **Đại số tuyến tính** | vector, ma trận, nhân ma trận |
| Làm sao chỉnh `θ` để `f` tốt hơn? | **Giải tích** | đạo hàm, gradient, chain rule |
| "Tốt hơn" nghĩa là gì? Output là gì? | **Xác suất** | phân phối, softmax, cross-entropy |

Bạn **không cần** học toán ở mức chứng minh định lý. Cần hiểu **trực giác** và **tính toán được**.

---

## 2. Vector và tích vô hướng

### Vector là gì?

Vector là **một danh sách các con số**. Có hai cách nhìn:

```text
  Cách nhìn đại số:   v = [3, 2]

  Cách nhìn hình học: mũi tên từ gốc tọa độ

      y
      │
    2 ┤       ↗ v = [3, 2]
      │     ↗
    1 ┤   ↗
      │ ↗
      └──┬──┬──┬── x
         1  2  3
```

Trong AI, vector dùng để **biểu diễn mọi thứ**:
- Một ảnh 28×28 pixel → vector 784 số
- Một từ → vector **embedding** (GPT-2 dùng 768 số, GPT-3 dùng 12 288 số)
- Một câu → vector ý nghĩa

### Tích vô hướng (dot product) — phép toán quan trọng nhất

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^{n} a_i b_i$$

Ví dụ: $[1, 2, 3] \cdot [4, 5, 6] = 1·4 + 2·5 + 3·6 = 32$

**Ý nghĩa hình học:** $\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta$ — đo **mức độ cùng hướng** của 2 vector.

```text
   Cùng hướng            Vuông góc             Ngược hướng
   a·b > 0 (lớn)         a·b = 0               a·b < 0

     ↗ a                   ↑ a                   ↗ a
    ↗ b                    │                    ╱
   ╱                       └──→ b              ╱
                                              ↙ b
   "giống nhau"          "không liên quan"    "trái ngược"
```

> 🔗 **Liên hệ GPT:** Cơ chế **attention** tính tích vô hướng giữa vector "câu hỏi" (query) của từ này với vector "khóa" (key) của từ khác để đo **từ nào liên quan đến từ nào**. Chính là phép toán trên!

---

## 3. Ma trận và phép nhân ma trận

### Ma trận là gì?

Ma trận là **bảng số** có `m` hàng, `n` cột — ký hiệu shape `(m, n)`.

$$W = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{shape } (2, 3)$$

**Trực giác quan trọng:** Ma trận là một **phép biến đổi** — nó biến một vector thành vector khác (xoay, co giãn, chiếu sang không gian khác số chiều).

```text
   x (3 chiều)    ──[ W shape (2,3) ]──▶    y (2 chiều)
   [x1, x2, x3]                             [y1, y2]
```

### Nhân ma trận với vector

Mỗi phần tử của kết quả = **tích vô hướng của một hàng với vector**:

```text
      W (2×3)          x (3)        y (2)
  ┌ 1  2  3 ┐       ┌ 1 ┐      ┌ 1·1 + 2·0 + 3·2 ┐   ┌ 7  ┐
  └ 4  5  6 ┘   ×   │ 0 │  =   └ 4·1 + 5·0 + 6·2 ┘ = └ 16 ┘
                    └ 2 ┘
   hàng 1 · x = 7
   hàng 2 · x = 16
```

### Nhân hai ma trận — quy tắc shape

```text
    A          B           C = A @ B
  (m, k)  @  (k, n)   =   (m, n)
       └──┬──┘
     phải bằng nhau!

  C[i, j] = hàng i của A  ·  cột j của B
```

Ví dụ: `(32, 768) @ (768, 3072) → (32, 3072)` — đây đúng là một phép tính trong lớp feed-forward của GPT-2.

### Một lớp mạng nơ-ron = nhân ma trận + cộng bias

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

```text
    đầu vào x          trọng số W          đầu ra y
      (3)                (4×3)               (4)
       ○ ─────────────╲──────────────▶ ○
       ○ ──────────────╳─────────────▶ ○     Mỗi đường nối = 1 phần tử của W
       ○ ─────────────╱──────────────▶ ○     Mỗi nút ra = tổng có trọng số
                                      ○      của tất cả nút vào (+ bias)
```

> 🔗 **Liên hệ GPT:** Hơn 90% phép tính khi chạy GPT/Claude/Gemini là **nhân ma trận**. Đó là lý do GPU (chuyên nhân ma trận song song) quan trọng đến vậy.

---

## 4. Tensor, shape và broadcasting

**Tensor** = mảng nhiều chiều (tổng quát hóa của vector và ma trận):

```text
  Scalar      Vector       Matrix          Tensor 3D
  (0 chiều)   (1 chiều)    (2 chiều)       (3 chiều)

    5         [1 2 3]      ┌1 2 3┐         ┌─────────┐
                           │4 5 6│        ┌┴────────┐│
                           └7 8 9┘       ┌┴────────┐││
                                         │         │┘┘
  shape ()    shape (3,)   shape (3,3)   shape (2,3,3)
```

Trong GPT, dữ liệu luôn có shape **`(B, T, C)`**:
- `B` = batch — số câu xử lý cùng lúc
- `T` = time — số token trong mỗi câu
- `C` = channels — số chiều embedding

> ⚠️ **80% lỗi khi code deep learning là lỗi shape.** Thói quen tốt nhất: luôn `print(x.shape)`.

### Broadcasting

NumPy/PyTorch tự động "kéo giãn" tensor nhỏ để khớp với tensor lớn:

```text
   X (3, 4)              b (4,)               X + b (3, 4)
  ┌ 1  2  3  4 ┐                            ┌ 11 22 33 44 ┐
  │ 5  6  7  8 │   +   [10 20 30 40]   =    │ 15 26 37 48 │
  └ 9 10 11 12 ┘         ↓ tự nhân bản      └ 19 30 41 52 ┘
                       [10 20 30 40]
                       [10 20 30 40]
                       [10 20 30 40]
```

Quy tắc: so sánh shape **từ phải sang trái**, mỗi chiều phải **bằng nhau** hoặc **bằng 1**.

---

## 5. Đạo hàm và gradient

### Đạo hàm = độ dốc

Đạo hàm $f'(x)$ cho biết: **nếu tăng x một chút, f thay đổi bao nhiêu?**

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

```text
   f(x)
    │ ╲                   ╱
    │  ╲                 ╱  ← f'(x) > 0: đi lên
    │   ╲               ╱       (tăng x → f tăng)
    │    ╲    ───      ╱
    │     ╲_________╱
    │  ↑        ↑
    │  f'<0     f'=0: đáy (cực tiểu)
    └─────────────────────── x
```

Các đạo hàm cần thuộc:

| $f(x)$ | $f'(x)$ | Gặp ở đâu |
|---|---|---|
| $x^2$ | $2x$ | loss MSE |
| $e^x$ | $e^x$ | softmax |
| $\ln x$ | $1/x$ | cross-entropy |
| $\max(0, x)$ (ReLU) | $1$ nếu $x>0$, ngược lại $0$ | hàm kích hoạt |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | hồi quy logistic |

### Đạo hàm riêng & gradient

Khi hàm có **nhiều biến** $f(w_1, w_2)$, đạo hàm riêng $\frac{\partial f}{\partial w_1}$ = độ dốc theo hướng $w_1$ (giữ $w_2$ cố định).

**Gradient** = vector gom tất cả đạo hàm riêng:

$$\nabla f = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_n} \right]$$

```text
   Bản đồ đường đồng mức của loss (nhìn từ trên xuống)

         ╭───────────────╮
       ╭─┤  ╭─────────╮  ├─╮
       │ │  │  ╭───╮  │  │ │       ● = vị trí hiện tại
       │ │  │  │ ★ │  │  │ │       ★ = cực tiểu (loss thấp nhất)
       │ │  │  ╰───╯  │ ●──→ ∇f    ∇f chỉ hướng DỐC LÊN nhanh nhất
       │ │  ╰─────────╯ ←─ -∇f     -∇f chỉ hướng XUỐNG nhanh nhất
       ╰─┤               ├─╯
         ╰───────────────╯
```

> 🔗 **Liên hệ GPT:** GPT-3 có 175 tỷ tham số → gradient là vector 175 tỷ chiều. Mỗi bước huấn luyện tính gradient này một lần.

---

## 6. Quy tắc chuỗi (chain rule)

Đây là **khái niệm toán quan trọng nhất** của deep learning — nền tảng của backpropagation.

Nếu $y = f(g(x))$, thì:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**Trực giác:** Nếu xe đạp nhanh gấp 2 lần người đi bộ, ô tô nhanh gấp 4 lần xe đạp → ô tô nhanh gấp 2 × 4 = 8 lần người đi bộ. **Tốc độ thay đổi nhân dồn qua từng khâu.**

### Ví dụ tính tay

$L = (3w + 1)^2$. Đặt $z = 3w + 1$, khi đó $L = z^2$.

```text
   Đồ thị tính toán (computational graph):

   Lan truyền xuôi (forward) ──────────────────▶
     w=1  ──[×3, +1]──▶  z=4  ──[bình phương]──▶  L=16

   Lan truyền ngược (backward) ◀──────────────────
   dL/dw = 8·3 = 24  ◀──  dL/dz = 2z = 8  ◀──  dL/dL = 1
            ↑
         dz/dw = 3
```

$$\frac{dL}{dw} = \frac{dL}{dz} \cdot \frac{dz}{dw} = 2z \cdot 3 = 2 \cdot 4 \cdot 3 = 24$$

> 🔗 **Liên hệ GPT:** GPT có ~100 lớp xếp chồng. Backpropagation = áp dụng chain rule **ngược từ loss về từng tham số**, nhân dồn đạo hàm qua từng lớp. Bạn sẽ tự code điều này ở Giai đoạn 2.

### Kiểm tra đạo hàm bằng số

Luôn có thể kiểm tra đạo hàm tự tính bằng công thức:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}, \quad h \approx 10^{-5}$$

---

## 7. Gradient descent

**Ý tưởng:** Muốn tìm đáy thung lũng khi bị bịt mắt? Cảm nhận độ dốc dưới chân, bước **ngược hướng dốc lên**, lặp lại.

$$\theta \leftarrow \theta - \eta \cdot \nabla L(\theta)$$

- $\theta$: tham số (trọng số)
- $\eta$: **learning rate** (độ dài bước)
- $\nabla L$: gradient của loss

```text
  loss
   │ ●
   │  ╲ bước 1
   │   ●
   │    ╲ bước 2
   │     ●
   │      ╲●  bước 3
   │        ●●● ← hội tụ về đáy
   └─────────────────── θ
```

### Learning rate quan trọng thế nào?

```text
  η quá nhỏ               η vừa phải              η quá lớn
  (rất chậm)              (tốt)                   (phân kỳ, nổ tung)

  ╲               ╱      ╲               ╱       ●               ●
   ●             ╱        ●             ╱         ╲             ╱
    ●           ╱          ╲           ╱           ╲     ●     ╱
     ●         ╱            ●         ╱             ╲  ╱   ╲  ╱
      ●●●●...╱               ╲___●___╱               ╲╱     ╲╱
```

> 🔗 **Liên hệ GPT:** Huấn luyện GPT = gradient descent (biến thể Adam) chạy hàng trăm nghìn bước trên hàng nghìn GPU. Thuật toán cốt lõi **không khác** notebook `02_gradient_descent_tu_dau.ipynb`.

---

## 8. Xác suất, softmax và cross-entropy

### Phân phối xác suất

Một danh sách số không âm, tổng bằng 1. Ví dụ phân phối "từ tiếp theo" sau câu *"Con mèo ngồi trên"*:

```text
   "ghế"   ████████████████  0.40
   "bàn"   ██████████        0.25
   "mái"   ██████            0.15
   "sofa"  ████              0.10
   ...     ████              0.10
                             ────
                             1.00
```

**Kỳ vọng** $\mathbb{E}[X] = \sum_x x \cdot P(x)$ — giá trị trung bình có trọng số.

### Softmax — biến điểm số bất kỳ thành xác suất

Mô hình tính ra các **điểm số thô** (logits) có thể âm, lớn tùy ý. Softmax chuẩn hóa chúng:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

```text
   logits z         exp(z)          chia cho tổng       xác suất
   ┌──────┐        ┌───────┐         (7.39+2.72+1.11)    ┌──────┐
   │  2.0 │  ──▶   │  7.39 │   ──▶      = 11.22     ──▶  │ 0.66 │
   │  1.0 │  ──▶   │  2.72 │                             │ 0.24 │
   │  0.1 │  ──▶   │  1.11 │                             │ 0.10 │
   └──────┘        └───────┘                             └──────┘
   (bất kỳ)        (luôn dương)                          (tổng = 1)
```

**Temperature** — chia logits cho $T$ trước khi softmax:
- $T < 1$: phân phối "nhọn" hơn → câu trả lời chắc chắn, ít sáng tạo
- $T > 1$: phân phối "phẳng" hơn → đa dạng, sáng tạo, dễ sai

> Đây chính là tham số `temperature` bạn thấy trong API của ChatGPT/Claude!

**Mẹo ổn định số học:** $e^{1000}$ bị tràn số. Vì $\text{softmax}(z) = \text{softmax}(z - c)$ với mọi hằng số $c$, ta trừ $c = \max(z)$ trước.

### Cross-entropy — thước đo "đoán sai bao nhiêu"

Nếu mô hình gán xác suất $p$ cho đáp án đúng:

$$L = -\log p$$

```text
  loss = -log(p)
    │
  7 ┤●                      p = 0.001 → loss = 6.9  (đoán sai tự tin: phạt nặng)
    │
  5 ┤
    │
  3 ┤ ●                     p = 0.1   → loss = 2.3
    │
  1 ┤    ●                  p = 0.5   → loss = 0.69
    │           ●  ●  ●     p = 0.99  → loss = 0.01 (đoán đúng tự tin: gần 0)
  0 ┼──────────────────── p
    0         0.5        1
```

Với một batch, lấy trung bình:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\text{model}}(y_i \mid x_i)$$

### Maximum likelihood

Tối thiểu hóa cross-entropy **tương đương** với tối đa hóa xác suất mô hình gán cho dữ liệu thật (maximum likelihood). Nói cách khác: *dạy mô hình để dữ liệu thật trở nên "ít bất ngờ" nhất*.

> 🔗 **Liên hệ GPT:** Toàn bộ giai đoạn pretraining của GPT/Claude/Gemini chỉ tối ưu **một** hàm:
> $$L = -\log P(\text{token tiếp theo đúng} \mid \text{các token trước})$$
> Với từ điển 50 000 token và mô hình chưa học gì (đoán đều), loss ban đầu ≈ $-\log(1/50000) \approx 10.8$. Khi bạn huấn luyện GPT ở Giai đoạn 4, hãy kiểm tra con số này!

---

## 9. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung | Thời lượng |
|---|---|---|---|
| 1 | ⭐ [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Vector, ma trận như phép biến đổi, tích vô hướng. Xem tập 1–4, 9. | ~3 giờ |
| 2 | ⭐ [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Đạo hàm, chain rule (tập 4 quan trọng nhất). | ~3 giờ |
| 3 | [3Blue1Brown — LLMs explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) | Xem để có động lực: mọi thứ học ở đây dùng vào đâu. | 8 phút |

### Đọc thêm

- 📘 [Mathematics for Machine Learning](https://mml-book.github.io) — sách miễn phí, chương 2 (Đại số tuyến tính), 5 (Giải tích vector), 6 (Xác suất)
- 📘 [Dive into Deep Learning — Preliminaries](https://d2l.ai/chapter_preliminaries/index.html) — toán + code song song
- 📝 [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- 📝 [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## 10. Checklist & thực hành

### Lý thuyết

- [ ] Giải thích được ý nghĩa hình học của tích vô hướng
- [ ] Nhân ma trận bằng tay, nắm quy tắc shape `(m,k) @ (k,n) → (m,n)`
- [ ] Hiểu broadcasting, đoán được shape kết quả
- [ ] Tính đạo hàm của các hàm cơ bản trong bảng mục 5
- [ ] Tính tay chain rule qua đồ thị tính toán
- [ ] Giải thích vì sao đi ngược gradient thì loss giảm
- [ ] Giải thích softmax, temperature, cross-entropy

### Thực hành

- [ ] [`code/01_numpy_co_ban.ipynb`](code/01_numpy_co_ban.ipynb) — NumPy, nhân ma trận, softmax, cross-entropy
- [ ] [`code/02_gradient_descent_tu_dau.ipynb`](code/02_gradient_descent_tu_dau.ipynb) — gradient descent & hồi quy tuyến tính
- [ ] Tự tính tay đạo hàm của 3 hàm hợp, kiểm tra bằng numerical gradient
- [ ] Viết ghi chú trong [`notes/`](notes/README.md)

### Tiêu chí hoàn thành

✅ Giải thích được bằng lời cho một người khác: *vì sao đi ngược hướng gradient thì loss giảm*, và *softmax + cross-entropy đo cái gì*.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
