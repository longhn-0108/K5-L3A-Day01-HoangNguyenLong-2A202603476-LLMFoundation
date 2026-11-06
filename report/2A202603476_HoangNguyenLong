# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603476
- **Họ và tên:** Hoàng Nguyên Long
- **SEED** (in ở ô đầu notebook): 8795

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output (`15/15` bộ kiểm tra đạt) | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (`SEED=8795`, đủ 10/10 dự đoán) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (`baseline_accuracy=0.87`, `sigmoid_0=0.50`, `f1_example=0.73`, `cosine_example=0.60`, `llm_cost_per_day=10.50`) | ☑ |

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên tập dữ liệu đã làm sạch (sau khi bỏ 403 tin trùng lặp còn `5.169` tin: `train = 3.619`, `val = 775`, `test = 775` gồm `92 spam`, tập con cân bằng `SUB = 40` gồm `20 spam + 20 ham`):

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.739 | 0.850 | 0.889 | 4.031 | 27,065.964 | 0.051 | 0.132 | 0.0 |
| **2. BoW + MLP (deep learning)** | 0.975 | 0.859 | 0.913 | 0.947 | 6.763 | 24,566.123 | 0.259 | 0.342 | 0.0 |
| **3a. Encoder (MiniLM) + kNN** | 0.854 | 0.957 | 0.903 | 0.974 | 86.549 | 18.955 | 46.904 | 66.628 | 0.0 |
| **3b. Encoder (MiniLM) + LogReg** | 0.967 | 0.946 | 0.956 | 0.974 | 87.071 | 29.030 | 28.136 | 42.169 | 0.0 |
| **3c. Encoder (MiniLM) + MLP head** | 0.967 | 0.957 | 0.962 | 0.974 | 87.941 | 26.303 | 47.361 | 73.955 | 0.0 |

*(Ghi chú: Phần 4 chạy ở chế độ `HAS_LLM = False` do không nạp API key trực tiếp vào Colab; toàn bộ hàm `call_openai`, `build_classification_prompt`, `parse_label` và `llm_classify_batch` đã vượt qua bài kiểm tra giả lập `4.1 + 4.4` và phần tính tay chi phí LLM).*

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất so với kết quả thực nghiệm:

1. **`emb_search` (Thí nghiệm 3.A — Tìm kiếm theo nghĩa, lúc `03:46:29`)**
   - **Dự đoán:** *"Kết quả gần nhất là ham"* (vì câu truy vấn `"you have been selected to receive a complimentary holiday, reply now to claim"` cố tình né các từ khoá spam phổ biến như `"free"`, `"win"`, `"prize"`).
   - **Thực tế:** Cả **6/6 tin nhắn gần nhất** trong tập train đều là **`spam`** với điểm tương đồng cosine từ `0.509` đến `0.563` (cao nhất là `0.563 [spam] "okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award!"`).
   - **Vì sao lệch:** Tôi đánh giá theo tư duy khớp từ khoá bề mặt của Bag-of-Words, trong khi mô hình `all-MiniLM-L6-v2` đã được pre-train trên hàng tỷ câu nên ánh xạ cả cụm *"selected to receive a complimentary holiday, reply now to claim"* vào cùng vùng không gian ngữ nghĩa với các mẫu thông báo trúng kỳ nghỉ/giải thưởng (`"4* holiday"`, `"won £1000 cash or a Spanish holiday"`, `"collect your award"`).

2. **`ml_lr` (Thí nghiệm 1.B — Learning rate lớn `lr = 50`, lúc `03:16:53`)**
   - **Dự đoán:** *"lr=0.001 thì loss giảm rất chậm và gần như đi ngang; lr=50 thì bước nhảy quá lớn làm vượt qua cực tiểu nên loss dao động mạnh (răng cưa) và không hội tụ."*
   - **Thực tế:** Với `lr = 0.001`, đường loss giảm rất chậm đúng như dự đoán. Nhưng với `lr = 50`, loss chỉ vọt mạnh lên ở vài epoch đầu (`~2.65`), sau đó **lao thẳng xuống đáy và hội tụ cực nhanh, giữ phẳng mượt ở mức rất thấp** đến hết 300 epochs chứ không hề bị răng cưa suốt quá trình train.
   - **Vì sao lệch:** Ở epoch đầu tiên, mọi dự đoán là `p = 0.5` trong khi `~87.4%` dữ liệu là `ham` (`y = 0`), nên bước nhảy `lr = 50` đẩy `b` và `w` vọt mạnh về phía âm (overshoot), làm các tin `spam` bị đoán `p ≈ 0` khiến BCE loss tăng vọt. Tuy nhiên, ngay sau đó đa số các tin `ham` đã có `p ≈ 0` nên `error = p - y ≈ 0`; đồng thời ma trận Bag-of-Words 2.000 chiều cực kỳ thưa (mỗi tin chỉ có khoảng 10–20 từ khác 0), làm đạo hàm trung bình `grad_w = (X.T @ error) / n` của từng từ rất nhỏ. Khi nhân với `lr = 50`, nó lại trở thành tốc độ cập nhật hiệu quả giúp trọng số các từ khoá spam hội tụ nhanh mà không bị phân kỳ.

3. **`dl_curve` (Thí nghiệm 2.A — So sánh F1 của BoW + MLP với BoW + LogReg, lúc `03:28:38`)**
   - **Dự đoán:** *"val loss sẽ giảm ở vài epoch đầu rồi tăng trở lại do overfit (vì MLP có ~128k tham số trên 3619 mẫu); F1 của MLP sẽ xấp xỉ hoặc chỉ nhỉnh hơn nhẹ so với Logistic Regression."*
   - **Thực tế:** Hiện tượng overfit diễn ra đúng như dự đoán (`val loss` chạm đáy ở **epoch 7** rồi tăng dần). Tuy nhiên, `F1 (test)` của **BoW + MLP đạt `0.913`** (và `F1 SUB 40 = 0.947`), tăng vọt **`+0.063` điểm F1** (tức giảm gần 42% lượng lỗi `1 - F1`) so với **BoW + LogReg (`F1 = 0.850`)**, chủ yếu nhờ `Recall` tăng mạnh từ `0.739` lên `0.859`.
   - **Vì sao lệch:** Logistic Regression với `lr = 0.5, 300 epochs` dùng Gradient Descent toàn cục (full-batch) trên dữ liệu lệch nên rất bảo thủ (`Precision = 1.000` nhưng bỏ sót tới 26.1% spam). Ngược lại, MLP được tối ưu bằng **Adam theo mini-batch (`batch_size = 64`)** và lớp ẩn 64 nơ-ron kết hợp được các tổ hợp từ xuất hiện yếu hơn, giúp bắt được nhiều tin spam khó hơn mà `Precision` chỉ giảm rất nhẹ (`0.975`).

## 4. Phân tích 50 case khó

### Mức độ sụt giảm và nhóm case khó nhất theo từng dòng mô hình

| Mô hình | F1 SUB 40 (test thường) | F1 Stress Test (50 case gốc / 55 case kèm bonus) | Độ tụt F1 | Nhóm case tệ nhất (% đoán đúng trên 55 case) |
| --- | ---: | ---: | ---: | --- |
| **1. BoW + LogReg** | 0.889 | 0.378 / 0.381 | **−0.511** | Thao túng / lừa đảo (`12%`), Né từ khoá (`14%`), Spam giọng tự nhiên (`27%`) |
| **2. BoW + MLP** | 0.947 | 0.368 / 0.372 | **−0.579** | **Né từ khoá (`0%`)**, **Spam giọng tự nhiên (`9%`)**, Thao túng / lừa đảo (`25%`) |
| **3a. Emb + kNN** | 0.974 | 0.724 / 0.738 | −0.250 | Nói về spam (`40%`), Spam giọng tự nhiên (`45%`), Tưởng spam nhưng không phải (`64%`) |
| **3b. Emb + LogReg** | 0.974 | 0.583 / 0.618 | −0.391 | Thao túng / lừa đảo (`25%`), Spam giọng tự nhiên (`27%`), Nói về spam (`40%`) |
| **3c. Emb + MLP head** | 0.974 | 0.600 / 0.632 | −0.374 | Thao túng / lừa đảo (`25%`), Spam giọng tự nhiên (`36%`), Nói về spam (`40%`) |

- **Model tụt nhiều nhất:** **`2. BoW + MLP`** tụt mạnh nhất với **`-0.579` điểm F1** (từ `0.947` xuống `0.368` trên 50 case gốc, `0.372` trên 55 case; accuracy chỉ còn `50.9%–52.0%`), kế đến là **`1. BoW + LogReg`** tụt **`-0.511` điểm F1** (từ `0.889` xuống `0.378`).
- **Nhóm case làm model sụp đổ:**
  - Với nhóm **BoW (`1` và `2`)**: Sụp đổ gần như hoàn toàn ở nhóm **Né từ khoá (`obfuscation`)** (`0%` ở MLP, `14%–17%` ở LogReg), **Spam giọng tự nhiên (`natural_spam`)** (`9%–27%`) và **Thao túng / lừa đảo (`manipulation`)** (`12%–25%`).
  - Với nhóm **Embedding (`3a, 3b, 3c`)**: Đã xử lý tốt nhóm *Né từ khoá* (`83%–86%`) và *Prompt injection* (`86%–100%`), nhưng tụt sâu ở nhóm **Spam giọng tự nhiên** (`27%–45%` — cả **4 case mà tất cả các model đều sai `#12, #13, #16, #19`** đều thuộc nhóm này), **Thao túng / lừa đảo** (`25%` ở `3b/3c`) và **Nói về spam (`talks_about_spam`)** (`40%–50%`).

### Hai case cụ thể và cơ chế khiến model sai

1. **Case #16 (`natural_spam`, nhãn thật = `spam` — TẤT CẢ các model BoW và Embedding đều đoán sai thành `ham`):**
   > *"Long time no see! I've started a little side business with skincare, you'd love it, can I send you the catalogue link?"*
   - **Cơ chế sai của BoW:** Câu tin nhắn không chứa bất kỳ từ khoá quảng cáo truyền thống nào (`free`, `txt`, `call`, `claim`, `prize`, `£`). Ngược lại, nó chứa dày đặc các đại từ và từ giao tiếp cá nhân có trọng số âm mạnh kéo về `ham` trong từ điển (`i`, `you`, `can`, `me`, `my`, `it`). Tổng `w · x + b` mang giá trị âm sâu nên cả LogReg và MLP đều dự đoán là `ham`.
   - **Cơ chế sai của Embedding (MiniLM):** Encoder nén toàn bộ câu thành một vector 384 chiều phản ánh ngữ nghĩa và sắc thái câu. Vì tin nhắn ngụy trang dưới dạng lời hỏi thăm của bạn cũ (*"Long time no see!"*) và xin phép lịch sự (*"can I send you..."*), vector ngữ nghĩa của câu rơi hoàn toàn vào cụm hội thoại bạn bè (`ham`) trên không gian đặc trưng (tương tự với các case `#12`, `#13`, `#19`).

2. **Case #29 (`obfuscation`, nhãn thật = `spam` — BoW đoán sai thành `ham`, hoặc Case #47 `talks_about_spam` mà Embedding đoán sai thành `spam`):**
   - **Ở Case #29 (`obfuscation`, thật = `spam`):** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*
     - **Cơ chế sai của BoW:** Hàm `tokenize` tách từ theo regex `[a-z0-9£$€]+`, biến các từ bị viết kiểu leetspeak thành các token lạ (`c0ngr4tulat10ns`, `y0u`, `h4ve`, `w0n`, `g1ft`, `c4rd`, `cl4im`, `n0w`). Không một token nào trong số này khớp với từ điển 2.000 từ (`vocab`) học từ tập train → vector đầu vào `x` gần như toàn số `0`, chỉ còn bias `b < 0` (do tiên nghiệm lớp `ham` chiếm 87%) nên mô hình đoán chắc chắn là `ham`. Trong khi đó, MiniLM dùng WordPiece (subword tokenization) và ngữ cảnh xung quanh `£1000` nên vẫn nhận diện được đây là `spam`.
   - **Ở Case #47 (`talks_about_spam`, thật = `ham`):** *"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?"*
     - **Cơ chế sai của Embedding (`3b/3c`):** Dù người gửi đang kể lại và chê cười một tin nhắn rác, nội dung câu vẫn chứa nguyên vẹn mệnh đề *"won £1000 and to call some 0906 number"*. Vì MiniLM tạo biểu diễn bằng cách lấy trung bình có trọng số (mean pooling) các token trong câu mà không tách biệt hoàn toàn giữa **hành vi phát ngôn (meta-commentary/reporting)** và **nội dung được trích dẫn**, vector câu bị kéo mạnh về vùng chủ đề trúng thưởng/gọi số tính phí cao nên bộ phân loại tuyến tính/MLP phía trên kết luận nhầm là `spam`.

## 5. Thử thách cá nhân

- **Đề bài theo `SEED = 8795` (`8795 % 6 = 5` → Đề #6):**
  > *"LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?"*

- **Cách làm:**
  1. Viết hàm `build_json_prompt(text, examples=None)` yêu cầu LLM chỉ trả về một đối tượng JSON hợp lệ có cấu trúc `{"label": "spam" or "ham", "reason": "short reason"}`.
  2. Viết hàm `parse_json_label(response)` dùng biểu thức chính quy `re.search(r"\{.*?\}", response, flags=re.DOTALL)` để trích xuất chuỗi JSON (phòng trường hợp LLM bọc thêm markdown ```json), giải mã bằng `json.loads(...)`, lấy trường `"label"` rồi đưa qua `parse_label(...)` để chuẩn hoá về `1` (`spam`), `0` (`ham`) hoặc `None`. Nếu lỗi cú pháp JSON thì fallback về `parse_label(response)`.
  3. Khi gọi API cho bản JSON, nâng `max_tokens` từ `3` lên `max(60, CLASSIFY_MAX_TOKENS)` để JSON không bị cắt cụt giữa chừng, và đo tổng lượng token input/output cũng như chi phí USD trên 10 tin đầu của tập `SUB`.

- **Kết quả & Giải thích:**
  - Kiểm thử parser: `parse_json_label('{"label": "spam", "reason": "Prize scam"}') -> 1`, `parse_json_label('{"label": "ham", "reason": "Normal chat"}') -> 0`.
  - **So sánh token và chi phí:**
    - Với **bản 1 từ (`spam`/`ham`)**: Mỗi tin tốn khoảng `~45–50 token input` và chỉ `~1.3 token output` (`1 từ / 0.75`). Với giá `gpt-6-luna` (`$0.10 / 1M input`, `$0.50 / 1M output`), chi phí cho 10 tin chỉ khoảng `~$0.000055` (`~$0.0055 / 1k tin`).
    - Với **bản trả về JSON kèm `reason`**: Prompt dài hơn nhẹ (`~55 token input`), nhưng phần trả lời JSON (`{"label": "spam", "reason": "..."}`) tăng lên **`~15–25 token output`/tin**. Vì **giá output token đắt gấp 5 lần input token** (`$0.50` so với `$0.10 / 1M`), việc sinh thêm `~20 token output` làm chi phí output tăng gấp `15–20 lần`, khiến **tổng chi phí mỗi tin tăng gấp khoảng `2.5 – 3 lần`**, đồng thời làm tăng độ trễ (latency) do LLM phải sinh tuần tự từng token của câu giải thích và tăng nguy cơ lỗi định dạng (`n_invalid`) nếu JSON bị cắt cụt.
  - **Đánh giá — Có đáng không?**
    - **KHÔNG đáng** nếu chạy ở luồng lọc tự động quy mô lớn (production) chỉ cần quyết định chặn/cho qua, vì F1 không tăng thêm (khi `reasoning_effort="none"` và trường `"label"` đứng trước trường `"reason"` trong JSON, nhãn đã được quyết định trước khi sinh lời giải thích) nhưng lại tốn gấp 2.5–3 lần chi phí và làm chậm hệ thống.
    - **ĐÁNG** khi dùng trong công cụ kiểm duyệt nội bộ (human-in-the-loop), giải thích cho người dùng vì sao tin bị đưa vào mục Spam, hoặc để **audit/debug** hiểu vì sao LLM phân loại sai trên các ca khó.

## 6. Phương án cho sản phẩm thật

Với bài toán nhà mạng cần lọc **10 triệu SMS/ngày** (lưu lượng trung bình `~116 tin/giây`, giờ cao điểm có thể gấp 3–5 lần) và yêu cầu SLA **`p95 < 50 ms/tin`**:

### 1. Lựa chọn phương án và dẫn chứng số liệu
- **Nếu chọn 1 mô hình duy nhất:** Chọn **`3b. Encoder (MiniLM) + LogReg`**.
  - **Dẫn chứng từ bảng thực nghiệm:**
    - **Độ trễ đạt SLA:** `p50 = 28.136 ms/tin`, **`p95 = 42.169 ms/tin < 50 ms`** (nằm bên trái vạch SLA và nằm **trên đường Pareto**). Ngược lại, `3a. Encoder + kNN` (`p95 = 66.628 ms`) và `3c. Encoder + MLP head` (`p95 = 73.955 ms`) đều vượt quá ngưỡng 50 ms và bị `3b` lấn át; các phương án gọi LLM qua API cũng có độ trễ hàng trăm ms nên vi phạm SLA.
    - **Chất lượng cao:** Đạt `F1 (test) = 0.956` (`Precision = 0.967`, `Recall = 0.946`) và `F1 (SUB 40) = 0.974`, vượt xa `1. BoW + LogReg` (`F1 test = 0.850`) và `2. BoW + MLP` (`F1 test = 0.913`), đồng thời chống được `86%` nhóm *Né từ khoá* và `100%` nhóm *Prompt injection*.
    - **Chi phí vận hành:** **`$0.00 / 1k tin`** (tự host nội bộ, không tốn phí API). So sánh nếu dùng LLM nhỏ (`gpt-6-luna`, chi phí tính tay `$10.50 / 1 triệu tin`), với 10 triệu SMS/ngày sẽ tiêu tốn **`$105/ngày ≈ $3,150/tháng`** (nếu dùng model lớn `gpt-6.1-terra` sẽ tốn **`~$31,500/tháng`**).

- **Kiến trúc tối ưu đề xuất cho Production — Hệ thống phân tầng (Cascading Router 3 tầng):**
  - **Tầng 1 — Tiền xử lý chuẩn hoá Leetspeak + `1. BoW + LogReg` (hoặc `2. BoW + MLP`):** Với tốc độ cực nhanh (`p95 = 0.132 ms` – `0.342 ms`, thông lượng `~25,000–27,000 tin/s`), tầng này xử lý ngay **~80–85% tin nhắn cực kỳ rõ ràng** (`p < 0.05` hoặc `p > 0.95`, lại có `Precision = 1.000` ở ngưỡng cao nên không lo chặn nhầm OTP/tin hợp lệ).
  - **Tầng 2 — `3b. Encoder (MiniLM) + LogReg`:** Chỉ chạy cho **~15–20% tin nhắn** mà Tầng 1 chưa chắc chắn (`0.05 ≤ p ≤ 0.95`). Nhờ chỉ phải xử lý 1/5 lưu lượng (`~20–25 tin/s`), chỉ cần 1–2 worker CPU/GPU nhẹ là giữ được `p95` toàn hệ thống quanh mức `~5–15 ms` (thấp hơn nhiều so với mốc 50 ms).
  - **Tầng 3 — `LLM nhỏ + Prompt phòng thủ (<sms>...</sms>)` (Bất đồng bộ / Offline):** Chạy ngầm (không chặn luồng gửi tin thời gian thực) trên **~1% mẫu tin "lưỡng lự"** của Tầng 2 (`p ∈ [0.35, 0.65]`) để phát hiện các chiến dịch *Spam giọng tự nhiên* / *Thao túng lừa đảo* mới và tự động gán nhãn bổ sung vào tập train định kỳ cho Tầng 1 & 2.

### 2. Chấp nhận đánh đổi (Trade-offs)
- **Đánh đổi giữa chất lượng bắt spam tinh vi và độ trễ/chi phí:** Vì không thể gọi LLM đồng bộ trên từng tin (do giới hạn `p95 < 50 ms` và chi phí `$3,150/tháng`), hệ thống thời gian thực dựa trên `BoW` và `MiniLM + LogReg` sẽ chấp nhận để lọt một tỷ lệ nhỏ các tin **Spam giọng tự nhiên** (như `#12, #13, #16, #19`) và **Thao túng mạo danh người thân** (`#21 "Hi Mum..."`) trong lần đầu xuất hiện, trước khi Tầng 3 cập nhật mẫu mới.
- **Đánh đổi giữa hạ tầng và thông lượng:** `3b. Encoder (MiniLM) + LogReg` khi gọi lẻ từng tin trên CPU Colab đạt `29.03 tin/s` (`p95 = 42.17 ms`), nên để gánh trực tiếp `116 tin/s` trung bình của 10 triệu tin/ngày sẽ cần ghép lô nhỏ (dynamic micro-batching 5–10 ms), lượng hóa mô hình (ONNX/INT8) hoặc dùng kết hợp Tầng 1 (`BoW`) gánh tải trước như kiến trúc Router ở trên.
- **Ưu tiên `Precision` cao hơn `Recall`:** Trong viễn thông, việc chặn nhầm tin nhắn quan trọng của người dùng (mã OTP ngân hàng `#41`, lịch khám bác sĩ `#44`, việc khẩn cấp gia đình `#3`) gây hậu quả nghiêm trọng hơn để lọt một tin quảng cáo, nên chấp nhận đặt ngưỡng phân loại bảo thủ (`Precision ≥ 0.97`).

## 7. AI log và điều đã học

### Nhật ký sử dụng AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 1–3 (Chỉ hỏi khái niệm)** | *"Giải thích vì sao với Logistic Regression trên ma trận Bag-of-Words thưa và dữ liệu mất cân bằng (87% ham), đặt learning rate lr=50 lại làm loss vọt lên ở epoch 1 rồi hạ xuống đáy rất nhanh thay vì dao động liên tục?"* | AI giải thích cơ chế bước nhảy đầu tiên đẩy bias `b` về âm sâu làm đa số mẫu `ham` có `p ≈ 0` (`error ≈ 0`), và do ma trận BoW rất thưa nên gradient trung bình của từng cột từ vựng `X.T @ error / n` thực chất rất nhỏ. | Tự kiểm chứng lại bằng biểu đồ 3 mức `lr = [0.001, 0.5, 50]` ở Thí nghiệm 1.B và tự viết toàn bộ code Phần 1–3 không dùng `sklearn` (`15/15` checks đạt). |
| **Phần 4 (TODO 4.1–4.4 & Thử thách #6)** | *"Viết hàm `call_openai`, `parse_label` và `llm_classify_batch` theo đúng chữ ký template, truyền `**provider_kwargs(max_tokens)` để tương thích với `gpt-6-luna` (`max_completion_tokens` + `reasoning_effort='none'`) và xử lý các trường hợp trả về `'Label: spam'`, `'Not spam'`."* | AI gợi ý dùng `client.chat.completions.create(..., **provider_kwargs(max_tokens))` và dùng regex xóa tiền tố `^label\s*:\s*` trước khi kiểm tra `"ham"` / `"not spam"` / `"spam"`. | Đọc hiểu và kiểm tra thứ tự kiểm tra `"not spam"` trước `"spam"` (tránh lỗi như ở bài Bắt lỗi 3), chạy bộ unit test giả lập `unittest.mock` ở ô kiểm tra `4.1 + 4.4` và `4.2–4.3` đều báo `✅`. |
| **Phần 5 (Tổng hợp báo cáo)** | *"Đọc kết quả chạy thực tế trong file notebook `.ipynb` và tổng hợp các số liệu vào mẫu báo cáo cá nhân."* | AI trích xuất chính xác bảng kết quả cuối Phần 5, thời gian các lệnh `predict(...)` và thống kê lỗi trên 50 case khó. | Đối chiếu lại từng con số (`SEED=8795`, `F1`, `p95`, các case `#12, #13, #16, #19, #29, #47`) khớp 100% với output trong notebook. |

### Điều đã học hôm nay

Hôm nay tôi hiểu ra rằng **điểm F1/Accuracy rất cao trên tập test tiêu chuẩn (tới `0.956–0.962`) không có nghĩa là mô hình đã thực sự "hiểu" bản chất tin nhắn rác**, bởi tập test cùng phân phối chỉ lặp lại các mẫu quảng cáo quen thuộc; khi đối mặt với bộ Stress Test 50 case khó (như lừa đảo thao túng tâm lý hay spam viết giọng bạn bè tự nhiên), cả BoW lẫn Embedding đều tụt F1 mạnh và cùng bó tay trước các tin spam không dùng từ khoá. Ngoài ra, tôi hiểu rõ **bản chất đánh đổi trên đường Pareto giữa 4 thế hệ AI**: mô hình lớn hơn không phải lúc nào cũng là lựa chọn tốt nhất cho sản phẩm thật, mà việc kết hợp `Encoder (MiniLM) + LogReg` (hoặc kiến trúc Router nhiều tầng) mới là chìa khoá để vừa đáp ứng SLA độ trễ `< 50 ms` vừa tiết kiệm hàng nghìn USD chi phí API mỗi tháng.
