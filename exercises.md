# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Câu trả lời của mô hinh tro nen ngau hung va sang tao hon khi tang temperature. Tham so nay chia nho cac diem so tho truoc khi dua vao ham softmax, lam cho phan phoi xac suat tro nen phang hon.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Chatbot ho tro khach hang uu tien cau tra loi thuc te, on dinh, khong can bay bong sang tao, nen temp tu 0.0-0.4 la hop ly*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *10.000 nguoi x 3 lan = 30.000 luot/ngay, x 350 token output = 10,5 trieu token output moi ngay (10.500 don vi 1K token). Theo bang gia  GPT-4o = 10.500 x $0,010 = ~$105/ngay (~$3.150/thang); GPT-4o-mini = 10.500 x $0,0006 = ~$6,3/ngay (~$189/thang), GPT-4o dat hon khoang 16,7 lan. Ty le gia input cung dung la 16,7 lan ($0,0025 so voi $0,00015) nen con so nay khong doi du prompt dai hay ngan. Dang bo tien cho GPT-4o: cac tac vu suy luan nhieu buoc va sai mot phat la hong, vi du sinh/review code, doc hop dong hay tom tat tai lieu phap ly. Nen dung mini: cac tac vu ngan va lap lai theo mau nhu phan loai intent, gan nhan cam xuc, tra loi FAQ, chat luong gan nhu khong thua GPT-4o nhung chi phi lech gan 17 lan.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Do dai hai ban gan bang nhau (522 tu / 2.396 ky tu cho giao vien, 507 tu / 2.683 ky tu cho chuyen gia) nhung mat do thong tin rat khac: ban chuyen gia co so ky tu tren moi tu cao hon vi day thuat ngu dai nhu "Distributed Ledger Technology", "Merkle Root", "Byzantine Generals Problem", "Proof of Work", con ban giao vien dung tu ngan va quen thuoc. Ve vi du, giao vien dung an du "cuon so ma thuat" va canh ca lop cung ghi so khi doi keo/doi bi, xung ho theo vai co tro, chuyen gia thi chia muc I, II va mo ta cau truc block header, ham bam SHA-256, co che dong thuan. Nhu vay system prompt khong them kien thuc moi cho model, no la lop ngu canh dat truoc toan bo hoi thoai nen anh huong len moi luot tra loi sau do.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Doan van tieng Viet 115 tu, uoc luong 115/0,75 = 153 token, tiktoken dem duoc 163 token voi bo ma hoa o200k_base (gpt-4o). Model duoc huan luyen chu yeu tren du lieu tieng Anh nen tu tieng Anh thuong nam tron trong tu dien token, con chu tieng Viet co dau thanh la ky tu UTF-8 nhieu byte va it xuat hien trong corpus, nen bi cat thanh 2-3 manh.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming dang gia nhat khi co nguoi ngoi doi truoc man hinh va phan hoi dai: chatbot, tro ly viet noi dung, sinh code. Tong thoi gian sinh chu khong giam, nhung thoi gian cho ky tu dau tien giam tu vai chuc giay xuong duoi mot giay — nhu hai luot goi o cau 2.1 mat 14-21 giay cho tron 500 tu, neu khong stream thi nguoi dung nhin man hinh trang suot ca khoang do va de tuong la treo. Nguoc lai, non-streaming hop hon khi khong ai doi truc tiep hoac khi can toan bo output moi xu ly duoc: job chay nen, batch xu ly hang nghin ban ghi, va nhat la khi phai parse/validate ket qua (JSON, function call, structured output) can mot chuoi hoan chinh de xu ly hon nhieu so voi ghep tung chunk roi phai doan xem da du chua. Non-streaming cung don gian hon ve code va de retry.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff (delay = base_delay * 2^attempt, nhu trong retry_with_backoff) tang khoang cho gap doi sau moi lan hong: 0,1s - 0,2s - 0,4s. Nho vay tan suat request giam dan khi API con dang qua tai, cho no thoi gian hoi phuc, trong khi loi chi la trục trac nhat thoi (mang chop chop, 503) thi van duoc thu lai gan nhu ngay lap tuc. Delay co dinh thi nguoc lai: client cu dap vao API deu dan voi cung mot nhip, chinh no gop them tai vao luc he thong yeu nhat. Neu hang nghin client cung retry sau dung 1 giay, tat ca se dong bo thanh tung dot (thundering herd): API vua ngoi day lai an nguyen mot con song request, lai sap, roi ca dam lai cung cho 1 giay va lap lai vong lap do mai. Trong thuc te nen them jitter — cong them mot luong ngau nhien vao delay — de pha vo su dong bo giua cac client, va nen dat tran delay toi da thay vi de no tang vo han.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona minh chon la tro giang cua khoa AI. System prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Hai lua chon quan trong: (1) "tra loi ngan gon" vi output token la phan dat nhat trong bang gia, nen cat do dai la cach giam chi phi truc tiep nhat. (2) "bang tieng Viet" vi model se bam theo ngon ngu cua prompt, ma cau hoi ve AI thuong lan thuat ngu tieng Anh (embedding, fine-tune, token), rat de keo model tra loi luon bang tieng Anh; chi dinh ro ngon ngu giu output on dinh. Tu "than thien" thi dat tone phu hop voi vai tro tro giang, khong kho khan.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Han che lon nhat la cach cat lich su: history = history[-6:] trong run_assistant. Cat cung theo so message nen den luot thu 4 la thong tin luot 1 bien mat han, va vi cat theo so luong chu khong theo token nen 3 luot dai van co the lam tran context trong khi 3 luot ngan lai lang phi cho trong. Cai thien cu the: thay viec vut bo bang cat theo ngan sach token cong voi tom tat. Trien khai: dat MAX_HISTORY_TOKENS (vi du 2000), sau moi luot dung count_tokens cong don so token cua history; khi vuot nguong thi lay cac message cu nhat vua bi day ra, goi mot lan API voi prompt "tom tat ngan gon cac y chinh va thong tin nguoi dung da cung cap", roi chen ket qua vao ngay sau system prompt duoi dang {"role": "system", "content": "Tom tat hoi thoai truoc do: ..."}. Nhu vay cac luot gan nhat van giu nguyen van, con phan cu chi mat chi tiet chu khong mat hoan toan.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
