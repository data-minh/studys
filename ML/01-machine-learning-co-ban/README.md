# Giai đoạn 1 — Machine Learning cơ bản

**Thời lượng:** 3–4 tuần
**Mục tiêu:** Hiểu **vòng lặp học** cốt lõi: dự đoán → tính loss → tính gradient → cập nhật tham số. Vòng lặp này giống hệt khi huấn luyện GPT.

```text
                Bạn đang ở đây
                     ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Machine Learning là gì?](#1-machine-learning-là-gì)
2. [Bốn thành phần của mọi bài toán học](#2-bốn-thành-phần-của-mọi-bài-toán-học)
3. [Hồi quy tuyến tính](#3-hồi-quy-tuyến-tính-linear-regression)
4. [Chuẩn hóa dữ liệu](#4-chuẩn-hóa-dữ-liệu-feature-scaling)
5. [Hồi quy logistic — phân loại 2 lớp](#5-hồi-quy-logistic--phân-loại-2-lớp)
6. [Softmax regression — phân loại nhiều lớp](#6-softmax-regression--phân-loại-nhiều-lớp)
7. [Các biến thể gradient descent](#7-các-biến-thể-gradient-descent)
8. [Overfitting, underfitting và regularization](#8-overfitting-underfitting-và-regularization)
9. [Đánh giá mô hình](#9-đánh-giá-mô-hình)
10. [Video bài giảng & tài liệu](#10-video-bài-giảng--tài-liệu)
11. [Checklist & thực hành](#11-checklist--thực-hành)

---

## 1. Machine Learning là gì?

```text
  LẬP TRÌNH TRUYỀN THỐNG                MACHINE LEARNING

  Dữ liệu ──┐                           Dữ liệu ──┐
            ├──▶ [Máy tính] ──▶ Kết quả           ├──▶ [Máy tính] ──▶ Quy tắc
  Quy tắc ──┘                           Kết quả ──┘      (học)       (mô hình)
  (người viết)                          (ví dụ mẫu)
```

Thay vì viết tay quy tắc *"nếu email chứa 'trúng thưởng' thì là spam"*, ta đưa cho máy **hàng nghìn email đã gán nhãn** và để nó tự tìm quy tắc.

**Ba loại học chính:**

| Loại | Dữ liệu | Ví dụ | Trong LLM |
|---|---|---|---|
| **Supervised** (có giám sát) | cặp (đầu vào, nhãn đúng) | ảnh → "mèo" | SFT: (câu hỏi, câu trả lời mẫu) |
| **Self-supervised** (tự giám sát) | nhãn tự tạo từ chính dữ liệu | che một từ, đoán từ đó | **Pretraining**: đoán token tiếp theo |
| **Reinforcement** (tăng cường) | phần thưởng cho hành động | chơi cờ, thắng +1 | **RLHF**: câu trả lời tốt được thưởng |

> 🔗 **Liên hệ GPT:** ChatGPT/Claude/Gemini dùng **cả ba** — lần lượt ở giai đoạn pretrain, SFT và RLHF.

---

## 2. Bốn thành phần của mọi bài toán học

Mọi mô hình — từ hồi quy tuyến tính đến GPT-4 — đều gồm đúng 4 thành phần:

```text
  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐
  │  1. DỮ LIỆU │───▶│  2. MÔ HÌNH  │───▶│  3. HÀM LOSS│───▶│ 4. OPTIMIZER  │
  │   (x, y)    │    │  ŷ = f(x; θ) │    │   L(ŷ, y)   │    │ θ ← θ - η·∇L  │
  └─────────────┘    └──────────────┘    └─────────────┘    └───────┬───────┘
                            ▲                                       │
                            └──────────── cập nhật θ ───────────────┘
```

| Thành phần | Hồi quy tuyến tính | GPT |
|---|---|---|
| Dữ liệu | (diện tích nhà, giá) | hàng nghìn tỷ token văn bản |
| Mô hình | $\hat y = wx + b$ (2 tham số) | Transformer (hàng tỷ tham số) |
| Loss | MSE | Cross-entropy |
| Optimizer | Gradient descent | AdamW |

**Code của vòng lặp huấn luyện gần như không đổi** giữa hai trường hợp này.

---

## 3. Hồi quy tuyến tính (Linear Regression)

### Bài toán

Dự đoán một **số thực** (giá nhà, nhiệt độ...) từ các đặc trưng đầu vào.

### Mô hình

$$\hat{y} = w x + b \qquad \text{(1 đặc trưng)}$$

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = w_1x_1 + w_2x_2 + \dots + w_dx_d + b \qquad \text{(nhiều đặc trưng)}$$

```text
   giá (y)
     │                         ● 
     │                    ●  ╱
     │               ●   ╱ ●         ─── đường ŷ = wx + b
     │           ●  ┆  ╱             ┆   sai số (residual) = ŷ - y
     │         ●  ┆ ╱ ●
     │      ●   ┆╱                    w = độ dốc
     │     ┆  ╱ ●                     b = điểm cắt trục y
     │   ● ┆╱
     │   ╱ ●
     └──────────────────────── diện tích (x)
```

### Hàm loss: Mean Squared Error (MSE)

$$L(w, b) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

**Vì sao bình phương?** (1) sai số âm/dương không triệt tiêu nhau, (2) phạt nặng sai số lớn, (3) đạo hàm đẹp.

### Gradient (áp dụng chain rule)

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_{i}(\hat{y}_i - y_i)\,x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_{i}(\hat{y}_i - y_i)$$

Dạng vector (nhiều đặc trưng, $X$ shape `(N, d)`):

$$\nabla_{\mathbf{w}} L = \frac{2}{N} X^\top(\hat{\mathbf{y}} - \mathbf{y})$$

### Loss là một "cái bát"

```text
           L(w, b)
             │
         ╲   │   ╱        MSE của hồi quy tuyến tính là hàm LỒI (convex):
          ╲  │  ╱         chỉ có MỘT đáy duy nhất → gradient descent
           ╲ │ ╱          chắc chắn tìm ra nghiệm tốt nhất.
            ╲│╱
    ─────────★──────── w   (Mạng nơ-ron thì KHÔNG lồi — nhiều đáy, nhưng
                            thực tế gradient descent vẫn hoạt động tốt!)
```

> Hồi quy tuyến tính còn có nghiệm đóng (normal equation) $\mathbf{w} = (X^\top X)^{-1}X^\top \mathbf{y}$, nhưng ta học gradient descent vì nó **mở rộng được** cho mọi mô hình.

---

## 4. Chuẩn hóa dữ liệu (Feature Scaling)

Nếu đặc trưng có thang đo rất khác nhau (diện tích: 50–500 m², số phòng: 1–5), loss có dạng **thung lũng hẹp dài** → gradient descent đi zig-zag, rất chậm.

```text
   Chưa chuẩn hóa                     Đã chuẩn hóa
   (thung lũng hẹp)                   (bát tròn)

   ╭──────────────────────╮           ╭────────╮
   │  ╭────────────────╮  │          ╭┤ ╭────╮ ├╮
   │  │ ●╲  ╱╲  ╱╲  ★  │  │          ││ │ ★  │ ││
   │  ╰────╲╱──╲╱──────╯  │          ││ ╰──↖─╯ ││
   ╰──────────────────────╯          ╰┤      ● ├╯
      đi zig-zag, chậm                ╰────────╯  đi thẳng, nhanh
```

**Standardization** (dùng nhiều nhất):

$$x' = \frac{x - \mu}{\sigma}$$

→ mỗi đặc trưng có trung bình 0, độ lệch chuẩn 1.

> 🔗 **Liên hệ GPT:** Ý tưởng "giữ các giá trị ở thang đo ổn định" xuất hiện lại dưới dạng **LayerNorm** trong mọi lớp của Transformer.

---

## 5. Hồi quy logistic — phân loại 2 lớp

### Bài toán

Dự đoán **xác suất** thuộc lớp 1 (spam/không spam, bệnh/không bệnh).

### Hàm sigmoid — ép mọi số về khoảng (0, 1)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```text
   σ(z)
   1.0 ┤                    ●●●●●●●
       │                 ●●
       │               ●
   0.5 ┤─ ─ ─ ─ ─ ─ ─ ●  ← z = 0 → σ = 0.5
       │            ●
       │          ●●
   0.0 ┤●●●●●●●●●
       └───────┬──────┬──────┬──── z
              -5      0      5
```

### Mô hình

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b)$$

Dự đoán lớp 1 nếu xác suất > 0.5, tức là khi $\mathbf{w}^\top\mathbf{x} + b > 0$. Tập các điểm $\mathbf{w}^\top\mathbf{x} + b = 0$ là **đường biên quyết định** (decision boundary) — một đường thẳng.

```text
   x₂
    │  ○   ○          ╱
    │    ○    ○      ╱   ●
    │  ○   ○   ○    ╱  ●   ●       ○ = lớp 0
    │     ○        ╱ ●   ●         ● = lớp 1
    │  ○    ○     ╱    ●    ●      ╱ = biên quyết định w·x + b = 0
    │       ○    ╱  ●     ●
    └────────────────────────── x₁
```

### Hàm loss: Binary Cross-Entropy

$$L = -\frac{1}{N}\sum_i \Big[ y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \Big]$$

Chỉ là $-\log(\text{xác suất gán cho đáp án đúng})$ viết gọn cho 2 lớp.

**Điều kỳ diệu:** gradient có dạng giống hệt hồi quy tuyến tính:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N} X^\top(\hat{\mathbf{p}} - \mathbf{y})$$

---

## 6. Softmax regression — phân loại nhiều lớp

Mở rộng cho $K$ lớp (ví dụ: chữ số 0–9 trong MNIST).

```text
   ảnh 28×28                      W (10×784)          logits (10)    softmax      xác suất
   ┌──────────┐    duỗi thẳng    ┌──────────┐        ┌───────┐                 ┌───────┐
   │   ████   │   ──────────▶    │          │        │  1.2  │  ─────────▶     │ 0.02  │ "0"
   │  █    █  │    x (784)       │  nhân    │  ───▶  │ -0.5  │                 │ 0.00  │ "1"
   │     ██   │                  │  ma trận │        │  ...  │                 │  ...  │
   │   ██     │                  │  + b     │        │  6.1  │                 │ 0.91  │ "7" ✓
   │  ██████  │                  └──────────┘        │  ...  │                 │  ...  │
   └──────────┘                                      └───────┘                 └───────┘
```

$$\hat{\mathbf{p}} = \text{softmax}(W\mathbf{x} + \mathbf{b})$$

**Nhãn one-hot:** chữ số "7" được mã hóa thành `[0,0,0,0,0,0,0,1,0,0]`.

**Loss (categorical cross-entropy):** $L = -\log \hat{p}_{\text{lớp đúng}}$

**Gradient theo logits** — kết quả đẹp nhất trong deep learning:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}_{\text{one-hot}}$$

> 🔗 **Liên hệ GPT:** Lớp cuối cùng của GPT **chính là softmax regression** với $K$ = kích thước từ điển (~50 000–200 000 token). Mô hình "phân loại" xem token tiếp theo là token nào!

---

## 7. Các biến thể gradient descent

```text
  Batch GD                 Stochastic GD (SGD)       Mini-batch GD ⭐
  dùng TOÀN BỘ dữ liệu     dùng 1 mẫu / bước          dùng B mẫu / bước (B = 32..4096)

  ● ─────────▶ ★           ●  ╱╲    ╱╲               ●  ╲  ╱╲
  mượt nhưng mỗi            ╲╱  ╲  ╱  ╲ ★             ╲╱  ╲___★
  bước rất chậm            rất nhiễu                 cân bằng: đủ mượt,
                           nhưng mỗi bước nhanh      tận dụng GPU song song
```

**Thuật ngữ:**
- **Batch size:** số mẫu mỗi bước
- **Iteration / step:** một lần cập nhật tham số
- **Epoch:** một lượt đi qua toàn bộ dữ liệu

> 🔗 **Liên hệ GPT:** GPT-3 dùng mini-batch 3.2 triệu token/bước. Các LLM hiện đại thường chỉ đi qua dữ liệu **khoảng 1 epoch** vì dữ liệu quá lớn.

---

## 8. Overfitting, underfitting và regularization

### Chia dữ liệu

```text
  ┌──────────────────────────────┬──────────┬──────────┐
  │        Train (70-80%)        │ Val (10%)│Test (10%)│
  │   dùng để học tham số        │ chọn     │ đánh giá │
  │                              │ siêu tham│ CUỐI CÙNG│
  │                              │ số       │ (1 lần)  │
  └──────────────────────────────┴──────────┴──────────┘
```

### Ba trạng thái của mô hình

```text
   Underfitting             Vừa khít ✓               Overfitting
   (quá đơn giản)                                    (học thuộc lòng)

    │    ●  ●                │    ●  ●                │    ●╲ ●
    │  ●      ●              │  ●╱‾‾‾‾╲●              │  ●╱ ╲╱ ╲●
    │─────────────           │  ╱      ╲              │  ╱       ╲╱╲
    │●          ●            │●╱        ╲●            │●╱           ●
    └──────────              └──────────              └──────────
   train loss CAO            train thấp, val thấp     train RẤT thấp, val CAO
```

### Nhìn đồ thị loss để chẩn đoán

```text
   loss
    │╲
    │ ╲         val loss   ╱‾‾‾‾   ← bắt đầu overfit từ đây:
    │  ╲___________╱‾‾‾‾‾‾          dừng sớm (early stopping)
    │   ╲    ↑
    │    ╲___│________ train loss
    │        │         ‾‾‾‾‾‾‾‾‾‾‾
    └────────┴──────────────────── epoch
         điểm tốt nhất
```

### Chống overfitting

| Kỹ thuật | Ý tưởng |
|---|---|
| **Thêm dữ liệu** | cách tốt nhất — lý do LLM cần nghìn tỷ token |
| **L2 regularization / weight decay** | $L_{\text{total}} = L + \lambda\sum w^2$ — phạt trọng số lớn |
| **Early stopping** | dừng khi val loss bắt đầu tăng |
| **Dropout** (Giai đoạn 2) | tắt ngẫu nhiên nơ-ron khi huấn luyện |

> 🔗 **Liên hệ GPT:** AdamW (optimizer của GPT) = Adam + **W**eight decay.

---

## 9. Đánh giá mô hình

**Accuracy** = số dự đoán đúng / tổng số. Nhưng dễ đánh lừa khi dữ liệu mất cân bằng (99% email không phải spam → mô hình luôn đoán "không spam" vẫn đạt 99%!).

```text
                    Dự đoán
                 Spam    Không
         ┌──────────┬──────────┐
   Thật  │   TP     │    FN    │  Spam
         ├──────────┼──────────┤
         │   FP     │    TN    │  Không
         └──────────┴──────────┘

   Precision = TP / (TP + FP)   "trong số đoán là spam, bao nhiêu đúng?"
   Recall    = TP / (TP + FN)   "trong số spam thật, bắt được bao nhiêu?"
```

> 🔗 **Liên hệ GPT:** Language model được đo bằng **perplexity** $= e^{L}$ (Giai đoạn 3) và các benchmark (Giai đoạn 7).

---

## 10. Video bài giảng & tài liệu

### Video

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [Andrew Ng — Machine Learning Specialization](https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI) | Khóa 1 (Supervised ML): hồi quy tuyến tính, logistic, gradient descent, overfitting. **Xem toàn bộ khóa 1.** |
| 2 | [3Blue1Brown — Gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Trực quan hóa gradient descent trên mạng nơ-ron |
| 3 | [StatQuest — Neural Networks / Deep Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1) | Giải thích cực dễ hiểu, xem các video đầu về gradient descent |

### Đọc thêm

- 📘 [Dive into Deep Learning — Linear Regression](https://d2l.ai/chapter_linear-regression/index.html)
- 📘 [Dive into Deep Learning — Linear Classification](https://d2l.ai/chapter_linear-classification/index.html)
- 📘 [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) — chương 1–2 (toán chặt chẽ hơn)
- 📝 [Google — Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---

## 11. Checklist & thực hành

### Lý thuyết

- [ ] Kể được 4 thành phần của bài toán học và ví dụ cho GPT
- [ ] Tự suy ra gradient của MSE cho hồi quy tuyến tính
- [ ] Giải thích vì sao cần chuẩn hóa dữ liệu
- [ ] Hiểu sigmoid, decision boundary, binary cross-entropy
- [ ] Hiểu softmax regression và vì sao gradient là $\hat{p} - y$
- [ ] Phân biệt batch / SGD / mini-batch, epoch / step
- [ ] Chẩn đoán overfitting từ đồ thị loss

### Thực hành (lưu trong `code/`)

- [ ] Hồi quy tuyến tính nhiều đặc trưng bằng NumPy thuần (không dùng sklearn), có chuẩn hóa dữ liệu
- [ ] Hồi quy logistic phân loại 2 lớp, vẽ đường biên quyết định
- [ ] Softmax regression trên MNIST bằng NumPy (tải MNIST qua `sklearn.datasets.fetch_openml('mnist_784')` hoặc `torchvision`)
- [ ] Vẽ đồ thị train/val loss, thử L2 regularization và quan sát thay đổi

### Tiêu chí hoàn thành

✅ Tự viết từ đầu (không nhìn mẫu) một bộ phân loại softmax đạt **>90%** accuracy trên tập test MNIST.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
