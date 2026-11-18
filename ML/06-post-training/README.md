# Giai đoạn 6 — Post-training: biến base model thành trợ lý ⭐⭐

**Thời lượng:** 4–6 tuần
**Mục tiêu:** Hiểu phần làm nên **ChatGPT, Claude, Gemini**: SFT, RLHF, Constitutional AI, DPO và RL cho suy luận. Tự huấn luyện một chatbot nhỏ.

```text
                                                                                           Bạn đang ở đây
                                                                                                 ▼
[Toán & Python] → [ML cơ bản] → [Neural Net] → [Language Model] → [Transformer] → [Pretrain] → [Post-train] → [Sản phẩm]
```

## Mục lục

1. [Toàn cảnh post-training](#1-toàn-cảnh-post-training)
2. [SFT — Supervised Fine-Tuning](#2-sft--supervised-fine-tuning)
3. [Vì sao SFT chưa đủ?](#3-vì-sao-sft-chưa-đủ)
4. [Reward model — dạy máy chấm điểm](#4-reward-model--dạy-máy-chấm-điểm)
5. [RLHF với PPO](#5-rlhf-với-ppo)
6. [Constitutional AI / RLAIF — cách của Anthropic](#6-constitutional-ai--rlaif--cách-của-anthropic)
7. [DPO — bỏ qua reward model và RL](#7-dpo--bỏ-qua-reward-model-và-rl)
8. [RL cho suy luận: o1, DeepSeek-R1, GRPO](#8-rl-cho-suy-luận-o1-deepseek-r1-grpo)
9. [LoRA & QLoRA — fine-tune với ít tài nguyên](#9-lora--qlora--fine-tune-với-ít-tài-nguyên)
10. [Những gì post-training dạy thêm](#10-những-gì-post-training-dạy-thêm)
11. [Video bài giảng & tài liệu](#11-video-bài-giảng--tài-liệu)
12. [Checklist & thực hành](#12-checklist--thực-hành)

---

## 1. Toàn cảnh post-training

```text
  ┌────────────┐   ┌──────────────┐   ┌────────────────────────┐   ┌─────────────────┐   ┌───────────┐
  │ BASE MODEL │──▶│     SFT      │──▶│  Preference tuning     │──▶│ RL cho suy luận │──▶│ TRỢ LÝ AI │
  │            │   │ học định dạng│   │  RLHF / RLAIF / DPO    │   │ (reasoning)     │   │           │
  │ "viết tiếp │   │ hội thoại từ │   │  học điều gì là câu    │   │ học tự suy nghĩ │   │ ChatGPT   │
  │  internet" │   │ ví dụ mẫu    │   │  trả lời TỐT HƠN       │   │ giải bài khó    │   │ Claude    │
  └────────────┘   └──────────────┘   └────────────────────────┘   └─────────────────┘   │ Gemini    │
                    ~10K–1M hội thoại   ~100K–1M+ so sánh           bài toán có đáp án   └───────────┘
                                                                     kiểm chứng được
   ◀── ~95%+ compute ──▶ ◀──────────────── ~vài % compute, nhưng quyết định "tính cách" ──────────────▶
```

**So sánh với việc học ở trường** (ví dụ của Karpathy):

| Trong sách giáo khoa | Giai đoạn LLM |
|---|---|
| Đọc lý thuyết, kiến thức nền | **Pretraining** |
| Xem ví dụ có lời giải mẫu | **SFT** |
| Tự làm bài tập, chỉ biết đáp án cuối | **Reinforcement Learning** |

---

## 2. SFT — Supervised Fine-Tuning

### Ý tưởng

Tiếp tục huấn luyện base model (**cùng thuật toán** đoán token tiếp theo), nhưng trên dữ liệu **hội thoại mẫu** do con người (hoặc AI) viết.

### Chat template — hội thoại được "làm phẳng" thành chuỗi token

```text
   Hội thoại:                              Chuỗi token thực tế đưa vào mô hình:

   System: Bạn là trợ lý hữu ích.          <|im_start|>system
   User:   2 + 2 bằng mấy?                 Bạn là trợ lý hữu ích.<|im_end|>
   Assistant: 2 + 2 = 4.                   <|im_start|>user
                                           2 + 2 bằng mấy?<|im_end|>
                                           <|im_start|>assistant
                                           2 + 2 = 4.<|im_end|>

                                           <|im_start|>, <|im_end|>: token đặc biệt
                                           mới thêm vào từ điển (định dạng ChatML)
```

Khi dùng thật, hệ thống điền phần của user rồi dừng ở `<|im_start|>assistant` — mô hình **viết tiếp** phần trả lời, và sinh `<|im_end|>` khi xong.

### Loss masking — chỉ học phần trả lời

```text
   Token:  <system> Bạn là... <user> 2 + 2 bằng mấy? <assistant> 2 + 2 = 4. <end>
   Loss:   ─────────── không tính loss ────────────   ████ tính loss ████████

   Mô hình học cách TRẢ LỜI, không học cách đặt câu hỏi như user.
```

### Dữ liệu SFT đến từ đâu?

- **Người gán nhãn** chuyên nghiệp viết câu trả lời lý tưởng theo **hướng dẫn chi tiết** (hàng trăm trang) của công ty: hữu ích, trung thực, vô hại...
- Ngày nay phần lớn là **dữ liệu tổng hợp** do LLM khác sinh ra, được người/AI lọc lại.

> 💡 **Hiểu đúng bản chất:** Khi bạn chat với ChatGPT, bạn đang nói chuyện với một **mô phỏng thống kê của người gán nhãn** làm theo hướng dẫn của công ty — dựa trên kiến thức từ pretraining.

---

## 3. Vì sao SFT chưa đủ?

1. **Viết câu trả lời hoàn hảo thì khó, so sánh thì dễ.** Người gán nhãn khó tự viết bài thơ hay nhất, nhưng dễ nói "bài A hay hơn bài B".
2. **SFT chỉ bắt chước**, không vượt được chất lượng của người viết mẫu.
3. **Không khớp kiến thức:** người gán nhãn viết câu trả lời tự tin về điều mà **mô hình không biết** → mô hình học cách **bịa một cách tự tin** (hallucination).
4. Cần tín hiệu về **điều gì KHÔNG nên làm** — SFT chỉ cho ví dụ tốt.

→ Cần học từ **so sánh / phần thưởng**: Reinforcement Learning.

---

## 4. Reward model — dạy máy chấm điểm

### Thu thập dữ liệu so sánh

```text
   Prompt: "Giải thích lỗ đen cho trẻ 5 tuổi"

   Mô hình SFT sinh nhiều câu trả lời:
   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │ A: "Lỗ đen là vùng   │  │ B: "Hãy tưởng tượng  │  │ C: "Theo thuyết      │
   │ không-thời gian có   │  │ một cái máy hút bụi  │  │ tương đối rộng của   │
   │ độ cong vô hạn..."   │  │ siêu mạnh trên trời" │  │ Einstein..."         │
   └──────────────────────┘  └──────────────────────┘  └──────────────────────┘

   Người gán nhãn xếp hạng:   B  >  A  >  C
```

### Huấn luyện reward model

Reward model $r_\phi(x, y)$ = một LLM (thường khởi tạo từ mô hình SFT) với đầu ra là **một con số** thay vì phân phối token.

```text
   (prompt, câu trả lời) ──▶ [ Transformer ] ──▶ [ Linear → 1 số ] ──▶ điểm thưởng
```

**Loss Bradley-Terry:** muốn điểm câu tốt hơn ($y_w$, "win") cao hơn câu kém ($y_l$, "lose"):

$$L_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

---

## 5. RLHF với PPO

Quy trình **InstructGPT** (2022) — nền tảng của ChatGPT:

```text
   BƯỚC 1: SFT               BƯỚC 2: Reward Model           BƯỚC 3: RL (PPO)
   ┌──────────────┐          ┌─────────────────────┐         ┌──────────────────────────┐
   │ Người viết   │          │ Mô hình sinh nhiều  │         │ Prompt mới               │
   │ câu trả lời  │          │ câu trả lời         │         │    ▼                     │
   │ mẫu          │          │    ▼                │         │ Policy sinh câu trả lời  │
   │   ▼          │          │ Người xếp hạng      │         │    ▼                     │
   │ Fine-tune    │          │ A > B > C           │         │ Reward model chấm điểm   │
   │ base model   │          │    ▼                │         │    ▼                     │
   │              │          │ Huấn luyện RM       │         │ PPO cập nhật policy      │
   └──────────────┘          └─────────────────────┘         │ để tăng điểm             │
                                                             └──────────────────────────┘
```

### Hàm mục tiêu

$$\max_{\pi_\theta}\; \mathbb{E}_{x,\, y \sim \pi_\theta}\big[r_\phi(x, y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)$$

- Số hạng 1: tối đa hóa điểm thưởng
- Số hạng 2 (**KL penalty**): không được đi quá xa mô hình SFT ban đầu $\pi_{\text{ref}}$

### Vì sao cần KL penalty? — Reward hacking

Reward model chỉ là **xấp xỉ** sở thích con người, có lỗ hổng. Tối ưu quá mức → mô hình tìm ra cách "lừa" nó:

```text
   Điểm thưởng
   (reward model)   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾  ← RM vẫn cho điểm cao...
                   ╱
                  ╱  ·····
   Chất lượng    ╱ ·      ·····      ← ...nhưng chất lượng THẬT giảm
   thật         ╱·             ····     (câu trả lời dài dòng, nịnh bợ,
               ╱                   ··    lặp từ khen, "The the the...")
   ───────────┴────────────────────── số bước RL
              ↑ nên dừng quanh đây
```

### Bộ nhớ: 4 mô hình cùng lúc

```text
   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Policy   │  │ Reference    │  │ Reward model │  │ Value model  │
   │ (đang    │  │ (SFT, đóng   │  │ (đóng băng)  │  │ (critic, cho │
   │  học)    │  │  băng, cho KL)│  │              │  │  PPO)        │
   └──────────┘  └──────────────┘  └──────────────┘  └──────────────┘
   → Phức tạp, tốn tài nguyên, khó ổn định → động lực cho DPO và GRPO
```

---

## 6. Constitutional AI / RLAIF — cách của Anthropic

**Vấn đề:** Cần rất nhiều người gán nhãn, đặc biệt cho nội dung độc hại (tốn kém, ảnh hưởng tâm lý người gán nhãn), và các tiêu chí ngầm định khó kiểm soát.

**Ý tưởng:** Viết ra một **"hiến pháp"** — danh sách nguyên tắc bằng ngôn ngữ tự nhiên — và để **AI tự đánh giá** theo các nguyên tắc đó.

```text
   GIAI ĐOẠN 1: Học có giám sát (tự phê bình & sửa)

   Prompt độc hại ──▶ Mô hình trả lời (có thể có hại)
                            │
                            ▼
                     "Hãy phê bình câu trả lời theo nguyên tắc:
                      [chọn ngẫu nhiên 1 nguyên tắc từ hiến pháp]"
                            │
                            ▼
                     Mô hình tự phê bình ──▶ "Hãy viết lại câu trả lời"
                            │
                            ▼
                     Câu trả lời đã sửa ──▶ dùng làm dữ liệu SFT

   GIAI ĐOẠN 2: RL từ phản hồi của AI (RLAIF)

   Prompt ──▶ Sinh 2 câu trả lời A, B
                            │
                            ▼
                     AI được hỏi: "Theo nguyên tắc [X],
                     câu nào tốt hơn?" ──▶ nhãn so sánh
                            │
                            ▼
                     Huấn luyện preference model ──▶ RL (như RLHF)
```

**Ví dụ nguyên tắc:** *"Chọn câu trả lời hữu ích, trung thực và vô hại nhất"*, *"Chọn câu trả lời ít phân biệt đối xử nhất"*...

**Ưu điểm:**
- **Minh bạch:** hành vi mô hình dựa trên nguyên tắc viết rõ ràng, đọc được
- **Mở rộng quy mô:** AI tạo phản hồi nhanh và rẻ hơn con người
- Mô hình vừa **vô hại** vừa **hữu ích** hơn: giải thích lý do khi từ chối thay vì né tránh

> 🔗 **Liên hệ:** Claude được huấn luyện với Constitutional AI kết hợp phản hồi của con người. Anthropic đã công bố hiến pháp của Claude (xem link ở mục 11).

---

## 7. DPO — bỏ qua reward model và RL

**Direct Preference Optimization (2023):** chứng minh toán học rằng có thể tối ưu **trực tiếp** trên dữ liệu so sánh, không cần reward model riêng và không cần vòng lặp RL.

```text
   RLHF:  dữ liệu so sánh ──▶ Reward model ──▶ PPO (4 mô hình, phức tạp) ──▶ mô hình
   DPO:   dữ liệu so sánh ─────────────────▶ 1 hàm loss đơn giản ─────────▶ mô hình
```

$$L_{\text{DPO}} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

**Trực giác:** tăng xác suất câu tốt ($y_w$) và giảm xác suất câu kém ($y_l$) — **so với** mô hình tham chiếu, để không đi quá xa.

**Ưu:** đơn giản như SFT, ổn định, chỉ cần 2 mô hình (policy + reference). Được dùng rộng rãi trong các mô hình mở.

---

## 8. RL cho suy luận: o1, DeepSeek-R1, GRPO

### Ý tưởng mới (2024–2025)

Với bài toán **có đáp án kiểm chứng được** (toán, code chạy test, logic), không cần reward model — **kiểm tra đáp án trực tiếp**:

```text
   Bài toán: "Tìm x biết 3x + 7 = 22"

   Sinh 8 lời giải cho CÙNG bài toán:
   ┌─────────────────────────────────────────────┬────────┬──────────┐
   │ Lời giải                                    │ Đáp án │ Thưởng   │
   ├─────────────────────────────────────────────┼────────┼──────────┤
   │ 1: "3x = 22-7 = 15, x = 5"                  │   5    │   1 ✓    │
   │ 2: "3x = 22+7 = 29, x = 29/3"               │  9.67  │   0 ✗    │
   │ 3: "Trừ 7 hai vế... 3x=15... chia 3... x=5" │   5    │   1 ✓    │
   │ ...                                         │  ...   │  ...     │
   └─────────────────────────────────────────────┴────────┴──────────┘
   → Tăng xác suất các lời giải đúng, giảm xác suất các lời giải sai
```

### GRPO — Group Relative Policy Optimization (DeepSeek)

Thay vì cần value model (như PPO), so sánh mỗi lời giải với **trung bình của nhóm**:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

- $A_i > 0$: lời giải tốt hơn trung bình nhóm → tăng xác suất
- $A_i < 0$: kém hơn → giảm xác suất

→ Bỏ được value model, tiết kiệm bộ nhớ, đơn giản hơn PPO.

### Điều kỳ diệu: suy luận tự xuất hiện

Chỉ thưởng cho **đáp án đúng**, không dạy cách suy nghĩ — nhưng mô hình **tự học** được:

```text
   Độ dài câu trả lời
        │                              ╱‾‾‾   Mô hình tự học cách:
        │                         ╱‾‾‾        • suy nghĩ từng bước dài hơn
        │                    ╱‾‾‾             • tự kiểm tra lại ("Đợi đã, thử lại...")
        │               ╱‾‾‾                  • thử nhiều cách tiếp cận
        │          ╱‾‾‾                       • quay lui khi sai
        │     ╱‾‾‾
        └──────────────────────── bước huấn luyện RL
```

Đây là cách các **mô hình suy luận** (OpenAI o1/o3, DeepSeek-R1, Claude với extended thinking, Gemini Thinking) được huấn luyện: "nghĩ" trong một chuỗi token dài trước khi trả lời.

> 🔗 **Ý nghĩa sâu xa:** SFT chỉ **bắt chước** con người. RL cho phép mô hình **tự khám phá** chiến lược — về nguyên tắc có thể vượt qua con người (như AlphaGo tìm ra nước đi mà kỳ thủ không nghĩ tới).

---

## 9. LoRA & QLoRA — fine-tune với ít tài nguyên

**Vấn đề:** Fine-tune toàn bộ mô hình 7B cần ~112GB bộ nhớ (xem Giai đoạn 5).

**LoRA (Low-Rank Adaptation):** đóng băng $W$, chỉ học một **bản cập nhật hạng thấp** $\Delta W = BA$:

```text
                   x (d)
          ┌────────┴────────┐
          ▼                 ▼
   ┌─────────────┐     ┌─────────┐
   │             │     │ A (r×d) │  ← học được, r nhỏ (ví dụ 8)
   │  W (d × d)  │     └────┬────┘
   │  ĐÓNG BĂNG  │          ▼
   │             │     ┌─────────┐
   └──────┬──────┘     │ B (d×r) │  ← học được, khởi tạo = 0
          │            └────┬────┘
          └──────(+)────────┘
                  ▼
           h = Wx + BAx

   d = 4096, r = 8:
   W có 4096 × 4096 = 16.8M tham số
   A + B có 2 × 4096 × 8 = 65K tham số  → chỉ ~0.4%!
```

**QLoRA:** lượng tử hóa $W$ đóng băng xuống **4-bit** + LoRA → fine-tune mô hình 7B trên **một GPU ~16GB** (Colab/Kaggle).

---

## 10. Những gì post-training dạy thêm

| Khả năng | Cách dạy |
|---|---|
| **Định dạng hội thoại**, lượt nói | SFT với chat template |
| **Tính cách, giọng điệu** ("Tôi là Claude...") | SFT + preference tuning |
| **Biết nói "tôi không biết"** | Dò xem mô hình không biết gì → tạo dữ liệu dạy nó từ chối trả lời điều đó |
| **Dùng công cụ** (tìm kiếm, chạy code) | SFT/RL với token đặc biệt gọi tool, ví dụ `<search>...</search>` |
| **Từ chối yêu cầu có hại** | Dữ liệu an toàn, Constitutional AI, red-teaming |
| **Làm theo system prompt** | Dữ liệu có system prompt đa dạng |
| **Suy luận nhiều bước** | RL với phần thưởng kiểm chứng được |

### Tóm tắt: các công ty làm thế nào (theo công bố)

| | Nền tảng công bố |
|---|---|
| **ChatGPT** (OpenAI) | InstructGPT: SFT + RLHF (PPO); dòng o1/o3: RL quy mô lớn cho suy luận |
| **Claude** (Anthropic) | RLHF + **Constitutional AI** (RLAIF); hiến pháp công khai |
| **Gemini** (Google) | SFT + RLHF (theo báo cáo kỹ thuật Gemini); đa phương thức từ pretraining |
| **DeepSeek-R1** | RL thuần với GRPO + phần thưởng dựa trên quy tắc; công bố chi tiết, trọng số mở |

---

## 11. Video bài giảng & tài liệu

### Video (xem theo thứ tự)

| # | Video | Nội dung |
|---|---|---|
| 1 | ⭐ [Karpathy — State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) | Quy trình tạo ChatGPT: pretrain → SFT → RM → RLHF, rõ ràng nhất |
| 2 | ⭐ [Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | **Nửa sau video:** SFT, hallucination, RL, DeepSeek-R1, RLHF |
| 3 | ⭐ [Umar Jamil — RLHF explained (toán + code)](https://www.youtube.com/watch?v=qGyFrqc34yc) | Reward model, policy gradient, PPO chi tiết |
| 4 | ⭐ [Umar Jamil — DPO explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) | Bradley-Terry, suy ra công thức DPO |
| 5 | [Yannic Kilcher — GRPO explained (DeepSeekMath)](https://www.youtube.com/watch?v=bAWV_yrqx4w) | Thuật toán GRPO |
| 6 | [Umar Jamil — LoRA explained + code](https://www.youtube.com/watch?v=PXWYUTMt-AU) | LoRA từ đầu bằng PyTorch |
| 7 | [Stanford CS336 (2025)](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) | Các bài về alignment, SFT, RLHF, RL |

### Đọc thêm

- 💻 ⭐ [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) — pipeline ChatGPT thu nhỏ hoàn chỉnh: tokenizer → pretrain → SFT → RL → giao diện web (~100 USD)
- 📝 ⭐ [Hugging Face — Illustrating RLHF](https://huggingface.co/blog/rlhf)
- 📘 [Nathan Lambert — RLHF Book](https://rlhfbook.com) — sách miễn phí về RLHF và post-training
- 📝 [Anthropic — Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)
- 📝 [Lilian Weng — Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- 📘 [Hugging Face — LLM Course](https://huggingface.co/learn/llm-course) · [TRL docs](https://huggingface.co/docs/trl)
- 📄 Paper:
  - ⭐ [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) · [Anthropic HH-RLHF (2022)](https://arxiv.org/abs/2204.05862) · [PPO (2017)](https://arxiv.org/abs/1707.06347)
  - ⭐ [Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)
  - [DPO (2023)](https://arxiv.org/abs/2305.18290)
  - [Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903) · [DeepSeekMath / GRPO (2024)](https://arxiv.org/abs/2402.03300) · ⭐ [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)
  - [LoRA (2021)](https://arxiv.org/abs/2106.09685) · [QLoRA (2023)](https://arxiv.org/abs/2305.14314)

---

## 12. Checklist & thực hành

### Lý thuyết

- [ ] Vẽ lại sơ đồ toàn cảnh post-training
- [ ] Giải thích chat template và loss masking
- [ ] Nêu 4 lý do SFT chưa đủ
- [ ] Viết loss Bradley-Terry của reward model
- [ ] Giải thích hàm mục tiêu RLHF và vai trò của KL penalty
- [ ] Giải thích reward hacking bằng ví dụ
- [ ] Mô tả 2 giai đoạn của Constitutional AI
- [ ] Giải thích trực giác của loss DPO
- [ ] Giải thích GRPO và vì sao suy luận tự xuất hiện
- [ ] Tính số tham số LoRA cho một lớp cho trước

### Thực hành (lưu trong `code/`)

- [ ] SFT một mô hình nhỏ (`HuggingFaceTB/SmolLM2-360M` hoặc `Qwen/Qwen2.5-0.5B`) bằng TRL `SFTTrainer`, dùng LoRA
- [ ] Tự tạo ~200 cặp (chosen, rejected) tiếng Việt và chạy DPO bằng TRL `DPOTrainer`
- [ ] Chạy GRPO (TRL `GRPOTrainer`) trên bài toán số học đơn giản, vẽ đồ thị reward và độ dài câu trả lời
- [ ] Tự cài đặt LoRA từ đầu cho một lớp `nn.Linear`
- [ ] ⭐ Chạy **nanochat** — huấn luyện chatbot hoàn chỉnh của riêng bạn

### Tiêu chí hoàn thành

✅ Có một chatbot nhỏ **tự huấn luyện** trả lời theo định dạng hội thoại; giải thích được sự khác nhau giữa SFT, RLHF, Constitutional AI, DPO và GRPO.

---

## Nhật ký học

| Ngày | Nội dung | Điều đã hiểu | Còn thắc mắc |
|------|----------|--------------|--------------|
|      |          |              |              |
