# Lộ trình Deep Learning: từ con số 0 đến hiểu cách tạo ra ChatGPT / Claude / Gemini

> Không thể tự huấn luyện mô hình cỡ ChatGPT (cần hàng trăm triệu USD), nhưng có thể
> **tự tay làm phiên bản thu nhỏ của từng bước** — và đó là cách hiểu bản chất.

## Bức tranh tổng thể

```
[1] Tokenizer → [2] Pretraining (Transformer đoán token tiếp theo trên nghìn tỷ token)
      → [3] SFT (dạy trả lời dạng hội thoại)
      → [4] RLHF / RLAIF / DPO (căn chỉnh: hữu ích, trung thực, an toàn)
      → [5] RL cho suy luận (reasoning)
      → [6] Triển khai: inference, tool use, RAG, đa phương thức
```

## Tiến độ

Tổng thời gian dự kiến: **8–10 tháng** (10–15 giờ/tuần). Đổi ⬜ → 🟨 (đang học) → ✅ (xong).

| # | Giai đoạn | Thời lượng | Trạng thái |
|---|-----------|-----------|-----------|
| 0 | [Nền tảng Toán & Python](00-toan-va-python/README.md) | 4–6 tuần | ⬜ |
| 1 | [Machine Learning cơ bản](01-machine-learning-co-ban/README.md) | 3–4 tuần | ⬜ |
| 2 | [Mạng nơ-ron & Backpropagation ⭐](02-neural-network-backprop/README.md) | 3–4 tuần | ⬜ |
| 3 | [PyTorch & Mô hình ngôn ngữ đầu tiên](03-pytorch-va-language-model/README.md) | 3–4 tuần | ⬜ |
| 4 | [Transformer ⭐⭐ (trái tim của mọi LLM)](04-transformer/README.md) | 4 tuần | ⬜ |
| 5 | [Pretraining LLM (tạo base model)](05-pretraining-llm/README.md) | 4–6 tuần | ⬜ |
| 6 | [Post-training: biến base model thành trợ lý ⭐⭐](06-post-training/README.md) | 4–6 tuần | ⬜ |
| 7 | [Từ mô hình đến sản phẩm](07-tu-mo-hinh-den-san-pham/README.md) | 3–4 tuần | ⬜ |
| 8 | [Dự án cuối khóa](08-capstone/README.md) | 3–4 tuần | ⬜ |

Danh sách paper cần đọc: [papers/README.md](papers/README.md)  
Video YouTube theo từng giai đoạn: [TAI-LIEU-YOUTUBE.md](TAI-LIEU-YOUTUBE.md)

## Nguồn học cốt lõi

1. **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube) + *Deep Dive into LLMs like ChatGPT*
2. **Stanford CS336 — Language Modeling from Scratch**
3. **Dive into Deep Learning** — https://d2l.ai
4. **Hugging Face LLM Course**
5. Stanford CS224n (NLP), CS231n (Vision)

## Nguyên tắc học

- **Tự viết lại từ đầu** mọi thứ quan trọng trước khi dùng thư viện.
- **Ghi chú bằng lời của mình** trong `notes/` — không giải thích được là chưa hiểu.
- **Đọc paper sau khi đã code** — khi đó paper dễ hiểu hơn nhiều.
- Mỗi giai đoạn phải có **sản phẩm code chạy được** trước khi đi tiếp.

## Cài đặt môi trường

```powershell
# (Xóa .venv cũ nếu bị lỗi, rồi tạo lại)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Giai đoạn 0–2 chỉ cần CPU. Từ giai đoạn 3 nên dùng GPU: Google Colab / Kaggle (miễn phí),
hoặc thuê GPU (Lambda, RunPod, Vast.ai) cho giai đoạn 5–6.

## Cấu trúc thư mục

```
ML/
├── 00-toan-va-python/          # Bắt đầu từ đây
├── 01-machine-learning-co-ban/
├── 02-neural-network-backprop/
├── 03-pytorch-va-language-model/
├── 04-transformer/
├── 05-pretraining-llm/
├── 06-post-training/
├── 07-tu-mo-hinh-den-san-pham/
├── 08-capstone/
├── papers/                     # Danh sách & ghi chú paper
└── data/                       # Dataset (không commit)
```

Mỗi giai đoạn có `README.md` gồm: **lý thuyết chi tiết kèm sơ đồ minh họa**, công thức, liên hệ với GPT/Claude/Gemini, **link video bài giảng + tài liệu đọc thêm**, checklist và nhật ký học; cùng thư mục `notes/` và `code/`.

> 💡 Mở README bằng Markdown Preview (`Ctrl+Shift+V` trong VS Code) để công thức toán hiển thị đẹp.
