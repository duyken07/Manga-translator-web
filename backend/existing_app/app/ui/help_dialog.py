"""help_dialog.py (bản web): CHỈ phần dữ liệu hướng dẫn trích nguyên văn từ app/ui/help_dialog.py của Desktop,
để app.services.app_guide (chatbot hướng dẫn dùng app) hoạt động mà không cần Qt. Không chứa mã giao diện."""
from __future__ import annotations

from typing import List, Tuple

GUIDE: List[Tuple[str, str]] = [
    ("📘 Bắt đầu nhanh", """
<h2>Bắt đầu nhanh</h2>
<p>Quy trình thường dùng, từ ảnh gốc đến ảnh đã dịch. <b>Người dùng mới:</b> làm bước 1 và 2 trước tiên.</p>
<ol>
<li><b>Tạo API key Gemini</b> (miễn phí để bắt đầu): làm theo mục <i>Thiết lập → Tạo API key Gemini</i>. App dịch bằng Gemini nên chưa có key thì chưa dịch được.</li>
<li><b>Nhập key vào app:</b> [[menu]] → <b>Thiết lập…</b> → tab <i>Dịch thuật</i> → dán key vào ô API key → chọn <i>Cách dịch</i> → bấm <b>Lưu</b>.</li>
<li><b>Nạp ảnh:</b> ở <b>Trang chủ</b>, kéo thả tệp hoặc cả thư mục vào khung nét đứt, hoặc <b>nhấp vào khung</b> rồi chọn <i>Chọn tệp ảnh…</i> / <i>Chọn thư mục…</i>. Có thể nạp thêm bất cứ lúc nào ở [[menu]] → <b>Project</b> → Nạp thêm ảnh.</li>
<li><b>Dịch:</b> bấm <b>Dịch trang này</b> hoặc <b>Dịch toàn bộ</b> ở thanh trên cùng. Trang nào dịch xong sẽ tự redraw (xoá chữ gốc, chèn chữ dịch) nếu đã bật Redraw.</li>
<li><b>Kiểm tra:</b> chuyển giữa <b>Gốc</b> và <b>Bản dịch</b> (Ctrl+T). Sửa câu dịch trong bảng thoại bên phải.</li>
<li><b>Chỉnh:</b> kéo khung chữ ([[square-dashed]] Khung), đổi font / cỡ ở thanh định dạng, vẽ che chữ sót ([[brush]] Vẽ), vá nền ([[wand-sparkles]] Fill), chọn nhanh bóng thoại bằng [[wand]] Đũa thần.</li>
<li><b>Lưu và xuất:</b> [[menu]] → Project → <b>Lưu JSON</b>; [[menu]] → <b>Xuất</b> để ra ảnh, PSD hoặc TXT.</li>
</ol>
<p><b>Lưu ý:</b> Vẽ và Fill <b>không tự redraw</b> lại trang và không tự xếp lại khung chữ. Muốn áp dụng thì bấm <b>Redraw lại trang này</b> ở tab Redraw (xem mục Redraw).</p>
<p>Tiến độ và kết quả mọi tác vụ (dịch, dịch lại, redraw, Vẽ / Fill / Đũa thần) hiện ở <b>góc phải dưới</b> của cửa sổ: ký tự quay là đang chạy, ✓ xanh là xong, ✕ đỏ là lỗi. Rê chuột vào đó để xem đầy đủ các dòng.</p>
<p>Mẹo: di chuột lên bất kỳ nút nào để xem mô tả ngắn. Chatbot trợ lý (bong bóng tròn góc dưới bên phải) chỉ hiện khi đã vào Workspace.</p>
"""),
    ("⚙️ Thiết lập", """
<h2>Thiết lập</h2>
<p>Mở bằng [[menu]] → <b>Thiết lập…</b>. Cửa sổ có ba tab:</p>
<ul>
<li><b>Dịch thuật:</b> cách dịch, API key, model, bóng thoại và OCR (chưa có key: xem mục <i>Tạo API key Gemini</i>).</li>
<li><b>Redraw:</b> xoá chữ gốc, font, màu chữ, koharu.</li>
<li><b>Bộ thuật ngữ:</b> thuật ngữ riêng cho Gemini.</li>
</ul>
<p>Bấm nút lưu của tab nào thì chỉ tab đó được áp dụng và cửa sổ vẫn mở để bạn chỉnh tiếp tab khác. Ngoài ra: <b>Giao diện</b> ở [[menu]] → Giao diện, <b>Tuỳ chọn PSD</b> ở [[menu]] → Xuất → Tuỳ chọn PSD.</p>
<p>Chọn một mục con ở bên trái để xem chi tiết.</p>
"""),
]

SETTINGS_ITEMS: List[Tuple[str, str]] = [
    ("Tạo API key Gemini", """
<h2>Tạo API key Gemini (cho người dùng mới)</h2>
<p><b>API key</b> là một chuỗi ký tự riêng của bạn, giống mật khẩu để app được phép gọi Gemini dịch truyện. Google cấp key miễn phí để bắt đầu (có giới hạn số lượt dùng); bạn không cần nhập thẻ để lấy key.</p>
<h3>Các bước lấy key</h3>
<ol>
<li>Mở <a href="https://aistudio.google.com/apikey">aistudio.google.com/apikey</a> (Google AI Studio) và <b>đăng nhập bằng tài khoản Google</b>.</li>
<li>Lần đầu vào, tick đồng ý <b>Điều khoản dịch vụ</b> rồi bấm Continue. Với người dùng mới, AI Studio tự tạo sẵn một project và một key mặc định.</li>
<li>Ở trang <b>API Keys</b> (hoặc <i>Dashboard → API Keys</i> ở menu trái), bấm <b>Create API key</b>.</li>
<li>Đặt tên cho key (ví dụ <i>Manga Translator</i>), chọn project rồi bấm <b>Create key</b>. Nếu bạn đã có tài khoản Google Cloud, AI Studio không tự tạo project: vào <i>Dashboard → Projects</i> để <b>Import</b> project của bạn trước.</li>
<li>Bấm biểu tượng <b>sao chép</b> cạnh key để copy.</li>
</ol>
<h3>Nhập key vào app</h3>
<ol>
<li>[[menu]] → <b>Thiết lập…</b> → tab <b>Dịch thuật</b>.</li>
<li>Dán key vào ô <i>Dán Gemini API Key vào đây...</i> (tick <b>Hiện</b> để kiểm tra đã dán đúng, không thừa khoảng trắng).</li>
<li>Chọn <b>Cách dịch</b> (mặc định <i>YOLO + Gemini Vision</i> là ổn) và <b>Model Gemini</b>.</li>
<li>Bấm <b>Lưu</b>. App kiểm tra key và model trước khi lưu; báo thành công là dùng được ngay.</li>
</ol>
<h3>Lưu ý quan trọng</h3>
<ul>
<li><b>Giữ bí mật key.</b> Không đăng lên mạng, không gửi trong ảnh chụp màn hình. Nếu lỡ lộ, vào trang API Keys xoá key đó và tạo key mới.</li>
<li>Google đang chuyển sang loại key mới (<i>authorization key</i>); key tạo mới ở AI Studio đã là loại này. Nếu một key cũ bị báo lỗi quyền truy cập (403 / permission denied), hãy tạo key mới.</li>
<li>Bản miễn phí có giới hạn số lượt mỗi phút và mỗi ngày. Dịch cả bộ truyện dài có thể bị báo hết hạn mức: đợi một lúc, đổi sang model khác, hoặc bật thanh toán ở Google Cloud nếu cần dùng nhiều.</li>
<li>Không bấm được <b>Create API key</b> (báo không có quyền tạo key trong project): tài khoản chưa có quyền trên project đó. Dùng project khác hoặc nhờ quản trị viên project cấp quyền.</li>
</ul>
<p>Các lỗi thường gặp khác xem ở mục <i>Xử lý sự cố</i>.</p>
"""),
    ("Dịch thuật & API key", """
<h2>Thiết lập dịch thuật</h2>
<p>Mở bằng [[menu]] → <b>Thiết lập…</b>, tab <b>Dịch thuật</b>.</p>
<h3>Cách dịch</h3>
<ul>
<li><b>YOLO + Manga-OCR + Gemini:</b> YOLO / koharu tìm bóng thoại, Manga-OCR đọc chữ, Gemini dịch. Chính xác về vị trí, đọc chữ tại máy.</li>
<li><b>YOLO + Gemini Vision:</b> YOLO định vị, Gemini đọc và dịch trên ảnh đã đánh số. Là lựa chọn mặc định.</li>
</ul>
<h3>API key</h3>
<ul>
<li>Key đã lưu được điền sẵn mỗi lần mở (hiện dạng dấu chấm; tick <b>Hiện</b> để xem).</li>
<li>Key được lưu vào kho khoá của hệ điều hành (Windows Credential Manager) nếu máy có thư viện <code>keyring</code>; nếu không sẽ lưu dự phòng trong registry.</li>
<li>Bấm <b>Lưu</b> để kiểm tra key và model trước khi lưu. Chưa có key: xem mục <i>Tạo API key Gemini</i>.</li>
</ul>
<h3>Model Gemini và Suy luận</h3>
<ul>
<li>Chọn model trong danh sách hoặc gõ tên model mới. Nếu báo không tìm thấy model, đổi sang model khác.</li>
<li><b>Suy luận</b> (low / medium / high): cao hơn dịch kỹ hơn nhưng chậm và tốn token hơn.</li>
</ul>
<h3>Giữ ngữ cảnh giữa các trang</h3>
<p>Các trang được dịch lần lượt và mang theo bảng thuật ngữ (tên nhân vật, chức danh) từ trang trước, nên nhất quán hơn nhưng chậm hơn dịch song song.</p>
"""),
    ("Phát hiện bóng thoại & OCR", """
<h2>Phát hiện bóng thoại &amp; OCR</h2>
<p>Nằm ngay trong cửa sổ Thiết lập dịch thuật.</p>
<ul>
<li><b>Dùng koharu-yolo26s:</b> model tìm bóng thoại, vùng chữ và khung tranh. Nên bật.</li>
<li><b>Kết hợp thêm YOLO:</b> chạy thêm model YOLO thứ hai (file .pt bạn chọn) để bắt thêm bóng thoại. Chính xác hơn nhưng chậm hơn, vì mỗi trang chạy hai lượt model.</li>
<li><b>File YOLO (.pt):</b> chọn bằng nút Duyệt.</li>
<li><b>Model Manga-OCR:</b> tên repo Hugging Face (mặc định <code>mayocream/manga-ocr</code>) <i>hoặc đường dẫn thư mục model đã tải về máy</i>. Dùng thư mục local giúp khởi động nhanh và chạy được khi mất mạng.</li>
<li><b>Câu OCR đọc chưa chắc: gửi ảnh crop cho Gemini đối chiếu:</b> tăng độ chính xác với chữ khó đọc, đổi lại tốn thêm lượt gọi API.</li>
<li><b>Lọc chữ lạc từ bóng thoại khác (thử nghiệm):</b> loại các ký tự bị nhận nhầm từ bóng bên cạnh.</li>
</ul>
"""),
    ("Cài đặt Redraw", """
<h2>Cài đặt Redraw</h2>
<p>Mở bằng [[menu]] → <b>Thiết lập…</b>, tab <b>Redraw</b> (hoặc nút Cài đặt Redraw ở tab Redraw bên phải).</p>
<p>Redraw là bước <b>xoá chữ gốc</b> trong bóng thoại rồi <b>chèn chữ dịch</b> lên ảnh.</p>
<ul>
<li><b>Bật redraw tự động:</b> tự chạy khi dịch và khi thêm / sửa / xoá khung thoại. <b>Vẽ và Fill không nằm trong số này</b>: chúng không tự redraw lại trang hay xếp lại khung chữ (xem mục Redraw).</li>
<li><b>Model lama-manga:</b> chọn file model (.safetensors hoặc .onnx) để xoá chữ trên nền phức tạp; nền trắng đơn giản luôn dùng OpenCV (nhanh).</li>
<li><b>Thiết bị chạy LaMa / Số luồng CPU (torch):</b> chạy CPU hay GPU và số luồng dành cho torch. Để mặc định thường là ổn.</li>
<li><b>Font:</b> <i>Chỉ dùng font trong thư mục</i> và <i>Font thường</i> (xem mục Font chữ).</li>
<li><b>Màu chữ tối (R,G,B)</b> và <b>Giãn cách dòng:</b> giá trị chung cho cả truyện. Từng câu vẫn chỉnh riêng được ở thanh định dạng chữ.</li>
<li><b>Koharu:</b> dùng koharu-yolo26s để xác định bóng thoại và vùng chữ cần xoá, thư mục model, <i>Kích thước suy luận</i> (nhỏ hơn thì nhanh hơn), và tuỳ chọn tạo vùng "chỉ redraw" để xoá cả SFX ngoài bóng thoại khi dịch.</li>
</ul>
<p>Sau khi lưu, các trang đã redraw tự xếp chữ lại với font / màu mới. Muốn xoá lại chữ gốc bằng model mới, dùng <b>Redraw lại trang này</b> (tab Redraw) hoặc <b>Thêm ▾ → Chỉ redraw toàn bộ</b>.</p>
"""),
    ("Font chữ", """
<h2>Font chữ</h2>
<ul>
<li>Danh sách font hiển thị <b>mỗi tên bằng chính font đó</b> để xem trước. Chỉ font có đủ dấu tiếng Việt mới xuất hiện.</li>
<li><b>Chỉ dùng font trong thư mục</b> (Thiết lập → Redraw): nếu điền, danh sách font, font mặc định và font tự tìm chỉ lấy từ thư mục đó (gồm thư mục con). Để trống = dùng thư mục <code>fonts</code> của app và font hệ thống. Mục "Chọn font từ file…" sẽ ẩn khi đã giới hạn thư mục.</li>
<li><b>Font từng câu:</b> chọn ở thanh định dạng chữ, hoặc chuột phải câu trong bảng → <b>Tuỳ chỉnh font…</b>.</li>
</ul>
"""),
    ("Bộ thuật ngữ", """
<h2>Bộ thuật ngữ</h2>
<p>[[menu]] → <b>Thiết lập…</b>, tab <b>Bộ thuật ngữ</b>. Soạn sẵn các cặp thuật ngữ (tên riêng, chức danh, kỹ năng…) hay bị dịch sai để kết hợp với Gemini.</p>
<ul>
<li><b>Thêm dòng / Xoá dòng đã chọn</b> để sửa bảng; <b>Lưu</b> để lưu (cửa sổ Thiết lập vẫn mở).</li>
<li>Bộ thuật ngữ lưu thành một file JSON riêng, dùng lại cho nhiều bộ truyện bằng <b>Mở file khác…</b>.</li>
<li>Có tuỳ chọn riêng về cách xử lý kính ngữ (honorifics).</li>
</ul>
"""),
    ("Giao diện & tuỳ chọn PSD", """
<h2>Giao diện và tuỳ chọn PSD</h2>
<ul>
<li><b>Giao diện:</b> [[menu]] → Giao diện để đổi giữa các kiểu sáng / tối.</li>
<li><b>Tuỳ chọn PSD:</b> [[menu]] → Xuất → Tuỳ chọn PSD:
  <ul>
  <li><b>Chữ dịch là lớp chữ sửa được:</b> mỗi câu là lớp Type của Photoshop. Nếu Photoshop báo lỗi khi mở file, hãy tắt mục này.</li>
  <li><b>Tách lớp Fill / nét vẽ:</b> mỗi vùng Fill và mỗi nét vẽ là một lớp riêng. Tắt để gộp lại cho file gọn.</li>
  </ul></li>
</ul>
"""),
]

TOOLS_ITEMS: List[Tuple[str, str]] = [
    ("Chế độ xem: Gốc / Bản dịch / Ảnh sạch", """
<h2>Ba chế độ xem</h2>
<p>Ba nút xếp dọc ở thanh bên trái ảnh:</p>
<ul>
<li><b>[[image]] Gốc:</b> ảnh gốc kèm khung nhận diện bóng thoại.</li>
<li><b>[[languages]] Bản dịch:</b> ảnh đã xoá chữ gốc và chèn chữ dịch.</li>
<li><b>[[brush-cleaning]] Ảnh sạch:</b> ảnh đã xoá chữ gốc nhưng chưa có chữ dịch. Dùng để kiểm tra phần xoá chữ.</li>
</ul>
<p><b>Ctrl+T</b>: đổi nhanh qua lại giữa ảnh gốc và bản dịch để so sánh.</p>
"""),
    ("[[square-dashed]] Khung (Ctrl+E)", """
<h2>[[square-dashed]] Khung</h2>
<p>Bật bằng nút Khung hoặc <b>Ctrl+E</b>. Nút Khung, Vẽ, Fill và Đũa thần dùng chung chuột trái nên mỗi lúc chỉ bật một.</p>
<ul>
<li><b>Đang xem Bản dịch:</b> kéo thả khung chữ như Canva (di chuyển, đổi cỡ, xoay); chữ tự xếp lại vừa khung. Khung chữ có thể kéo ra <b>ngoài trang</b>; phần ngoài trang bị cắt khi xuất ảnh.</li>
<li><b>Đang xem Gốc:</b> chỉnh khung nhận diện bóng thoại. Ảnh redraw không đổi cho đến khi bạn chuột phải câu thoại trong bảng → <b>🎯 Áp dụng khung mới vào redraw</b>.</li>
<li><b>Ctrl + bấm khung:</b> chọn nhiều câu cùng lúc để chỉnh định dạng chung.</li>
<li><b>Dời khung chữ không đổi cách trình bày:</b> chỉ kéo (không đổi cỡ khung) thì app dời nguyên khối chữ, giữ đúng cỡ chữ và cách ngắt dòng cũ. Đổi kích thước khung thì chữ được xếp lại.</li>
<li><b>Chữ hình oval:</b> trong khung do bạn kéo / đổi cỡ, chữ tự xếp thành khối tròn kiểu bóng thoại (dòng giữa dài nhất, dòng đầu và cuối ngắn dần) và dùng cỡ chữ lớn nhất còn vừa khung. Cách xếp chỉ phụ thuộc kích thước khung và nội dung, không phụ thuộc vị trí.</li>
<li><b>Nhấn đúp vào khung chữ</b> (đang xem Bản dịch, có hay không bật nút Khung đều được) để <b>gõ trực tiếp</b> câu dịch ngay trên ảnh: <b>Enter</b> lưu, <b>Shift+Enter</b> xuống dòng, <b>Esc</b> huỷ; bấm ra ngoài cũng là lưu. Giống ô sửa câu dịch ở bảng thoại; lưu xong chữ tự xếp lại.</li>
</ul>
"""),
    ("Thanh định dạng chữ", """
<h2>Thanh định dạng chữ</h2>
<p>Hiện ở thanh trên cùng khi bạn chọn một hay nhiều câu thoại (và đang xem Bản dịch).</p>
<ul>
<li>Font (xem trước bằng chính font), cỡ chữ, màu chữ.</li>
<li>Căn lề, <b>viền chữ</b> (bật rồi chỉnh độ dày và màu), giãn dòng.</li>
<li><b>[[rotate-ccw]] Đặt lại:</b> đưa font, cỡ, giãn dòng, căn lề, màu của câu về cài đặt chung.</li>
<li>Chọn nhiều câu (Ctrl + bấm khung hoặc chọn nhiều dòng trong bảng) để chỉnh cùng lúc.</li>
</ul>
"""),
    ("[[brush]] Vẽ (Ctrl+D)", """
<h2>[[brush]] Vẽ</h2>
<p>Vẽ tay lên trang: <b>bút, bút dạ, dạ quang, tẩy</b>. Dùng bút dạ trắng để che chữ sót. Chữ dịch luôn nằm trên nét vẽ.</p>
<ul>
<li>Mỗi nét vẽ là một lớp, có thể ẩn / hiện / xoá ở tab Redraw; được lưu cùng project và in vào ảnh khi xuất.</li>
<li><b>Ctrl+Z</b> hoàn tác, <b>Ctrl+Y</b> hoặc <b>Ctrl+Shift+Z</b> làm lại.</li>
<li>Vẽ <b>không làm app tự redraw</b> lại trang và không đổi khung chữ. Ở chế độ vẽ thẳng lên ảnh, nếu khi xem <i>Bản dịch</i> thấy nét vẽ chưa hiện thì bấm <b>Redraw lại trang này</b>.</li>
<li><b>[[brush]] Vẽ thẳng lên ảnh:</b> bật nút này trên thanh Vẽ để in nét vẽ thẳng vào ảnh đang xem, không tạo lớp. App chỉnh trên một <b>bản làm việc</b> riêng, <b>file gốc của bạn không bao giờ bị sửa</b>. Muốn bỏ hết chỉnh sửa: ☰ → Project → <b>Bỏ chỉnh sửa trực tiếp của trang</b>. Hoàn tác bằng ↶ (giữ 15 bước gần nhất).</li>
</ul>
"""),
    ("[[wand-sparkles]] Fill (Ctrl+G)", """
<h2>[[wand-sparkles]] Fill</h2>
<p>Vá nền theo kiểu Content-Aware Fill: bôi lên vùng cần vá rồi bấm <b>✅ Áp dụng</b>.</p>
<ul>
<li><b>Cách vá</b> (ô chọn trên thanh Fill): <i>Tự động</i> (khuyên dùng: LaMa nếu có model, không thì Patch-match cho nền họa tiết và OpenCV cho nền phẳng), <i>LaMa</i> (AI, tốt cho nền phức tạp; cần bật model ở Thiết lập → Redraw), <i>Patch-match</i> (giữ screentone / nét kẻ), <i>OpenCV</i> (nhanh, hợp nền trắng hoặc đơn sắc).</li>
<li>Mỗi vùng Fill là một lớp, ẩn / hiện / xoá được ở tab Redraw.</li>
<li><b>Fill không tự redraw:</b> thêm, hoàn tác hay xoá vùng Fill không tự chạy lại redraw và không xếp lại khung chữ. Vùng Fill dạng lớp được vẽ bởi bộ máy Redraw nên sẽ <b>hiện sau khi bạn bấm Redraw lại trang này</b> (tab Redraw bên phải).</li>
<li><b>[[brush]] Fill thẳng lên ảnh:</b> giống chế độ vẽ thẳng: kết quả vá được in vào ảnh đang xem (bản làm việc), không tạo lớp và không sửa file gốc.</li>
</ul>
"""),
    ("[[wand]] Đũa thần (Ctrl+Shift+W)", """
<h2>[[wand]] Đũa thần</h2>
<p>Chọn nhanh bóng thoại giống Magic Wand của Photoshop, rồi tạo câu thoại cho <b>một hay nhiều bóng cùng lúc</b>. <b>Hỗ trợ bóng thoại có viền liền</b>, kể cả bóng gai nhọn, bóng bầu dục và bóng góc cạnh.</p>
<ol>
<li>Bật bằng nút <b>Đũa thần</b> hoặc <b>Ctrl+Shift+W</b>.</li>
<li><b>Bấm vào trong bóng thoại</b> (bấm trúng chữ cũng được). Vùng chọn hiện bằng viền nét đứt chuyển động và lớp tô xanh mờ; chữ bên trong bóng nằm gọn trong vùng chọn.</li>
<li><b>Chọn nhiều bóng:</b> cứ bấm tiếp các bóng khác, không cần giữ phím. <b>Bấm lại bóng đã chọn</b> (hoặc <b>Alt + bấm</b>) để bỏ bóng đó. <b>Ctrl + bấm</b> để chọn lại từ đầu. Kéo chuột để cuộn ảnh như bình thường.</li>
<li>Chỉnh <b>Dung sai</b> ở thanh trên: tăng nếu vùng chọn thiếu (nền bóng xám nhạt), giảm nếu lan quá tay. Mọi bóng đã chọn được tính lại ngay khi kéo thanh trượt.</li>
<li><b>Enter</b> (hoặc nút ✓): tạo câu thoại. Một bóng: mở hộp thoại "Thêm câu thoại mới" như khoanh vùng thường. Nhiều bóng: hộp thoại hỏi <b>một lần</b> (Quét OCR tự động hoặc Chỉ redraw) rồi áp dụng cho tất cả, các câu được thêm theo thứ tự đọc (trên xuống dưới, phải sang trái với truyện Nhật). <b>Esc</b>: bỏ chọn tất cả.</li>
</ol>
<p>Khi báo "Viền bóng không liền" hoặc "Vùng chọn tràn ra ngoài": bóng không có viền khép kín (nét đứt, hở lớn, hoặc hòa vào nền) nên đũa thần từ chối để không chọn nhầm cả nền; dùng khoanh vùng bằng Shift + kéo cho các bóng đó. Chữ nằm ngoài bóng (SFX, chữ tự do) và khung chữ không có viền (lời dẫn) cũng không hợp với đũa thần.</p>
"""),
    ("Thu phóng & bố cục", """
<h2>Thu phóng và bố cục</h2>
<ul>
<li><b>[[zoom-in]] / [[zoom-out]]:</b> phóng to, thu nhỏ. <b>[[maximize]] Vừa:</b> vừa khung nhìn. <b>1:1:</b> kích thước thật.</li>
<li><b>Ctrl+[</b> và <b>Ctrl+]</b>: thu gọn / mở danh sách trang (trái) và bảng thoại (phải). Kéo thanh chia để đổi bề ngang.</li>
<li><b>[[chevron-left]] [[chevron-right]]</b> ở thanh trên: chuyển trang. Có thể chọn trang trong danh sách.</li>
</ul>
"""),
]

FEATURE_ITEMS: List[Tuple[str, str]] = [
    ("Trang chủ (nạp ảnh)", """
<h2>Trang chủ (nạp ảnh)</h2>
<p>Màn hình đầu tiên chỉ để đưa ảnh vào app.</p>
<ul>
<li><b>Kéo thả</b> tệp ảnh hoặc cả thư mục vào khung nét đứt (có thể thả vào bất kỳ chỗ nào của cửa sổ).</li>
<li>Hoặc <b>nhấp vào khung</b> (cũng dùng được Enter / Space khi khung đang được chọn): hiện menu <i>Chọn tệp ảnh…</i> / <i>Chọn thư mục…</i>.</li>
<li>Định dạng hỗ trợ: BMP, JPEG, JPG, PNG, WEBP. Ảnh được sắp theo thứ tự tự nhiên (trang 2 trước trang 10).</li>
<li>Nạp xong, app chuyển sang Workspace và <b>chưa dịch / redraw gì</b> cho đến khi bạn bấm dịch.</li>
<li>Nút <b>[[menu]]</b> ở góc trên bên trái luôn có ở mọi màn hình; chọn <b>Trang chủ</b> trong menu để quay lại đây. Chatbot không hiện ở Trang chủ.</li>
</ul>
"""),
    ("Dịch trang", """
<h2>Dịch trang</h2>
<ul>
<li><b>Dịch trang này:</b> phát hiện bóng thoại, OCR và dịch trang đang xem.</li>
<li><b>Dịch toàn bộ:</b> dịch mọi trang trong danh sách.</li>
<li><b>Thêm ▾:</b> dịch hoặc chỉ redraw (không dịch) các trang đã chọn hay toàn bộ.</li>
<li><b>Chuột phải trong danh sách trang:</b> Dịch lại trang này, Dịch / Chỉ redraw các trang đã chọn (chọn nhiều bằng Ctrl hoặc Shift), Xoá ảnh, Nạp thêm ảnh / thư mục.</li>
</ul>
<p>Trang dịch xong sẽ tự redraw ở nền nếu đã bật Redraw, nên bạn có thể xem trang trước trong lúc trang sau đang xử lý.</p>
"""),
    ("Dịch lại trang", """
<h2>Dịch lại trang</h2>
<p>[[menu]] → Project → <b>Dịch lại trang hiện tại…</b> (hoặc chuột phải trang trong danh sách).</p>
<ul>
<li>Phát hiện + OCR + dịch lại từ đầu bằng cài đặt hiện tại. Các câu tự động bị <b>thay thế</b> và không hoàn tác được.</li>
<li><b>Giữ lại câu đã gõ / sửa tay:</b> bật để không mất phần bạn đã chỉnh.</li>
<li><b>Không redraw các khung mới:</b> giữ chữ gốc, chỉ chèn lớp chữ dịch.</li>
<li>Nét vẽ tay và vùng Fill được giữ nguyên.</li>
</ul>
"""),
    ("Bảng thoại", """
<h2>Bảng thoại</h2>
<p>Tab <b>Thoại</b> bên phải liệt kê các câu của trang: số thứ tự, câu gốc, câu dịch, và hai cột tick.</p>
<ul>
<li><b>Sửa câu dịch:</b> bấm đúp vào ô. <b>Shift+Enter</b> để xuống dòng.</li>
<li><b>Cột Câu gốc</b> hiện "câu gốc" và dòng "phiên âm" (Romaji, với truyện tiếng Nhật) bên dưới. Hai phần được lưu riêng nhau.</li>
<li><b>Cột 👁:</b> hiện / ẩn chữ dịch của câu. <b>Cột 🎨:</b> bật / tắt vùng xoá chữ gốc của câu.</li>
<li><b>Chuột phải một câu</b> (hoặc nhiều câu đã chọn):
  <ul>
  <li><b>👁 Ẩn / hiện:</b> Hiện chữ, Ẩn chữ, Bật vùng xoá chữ gốc, Tắt vùng xoá chữ gốc cho các câu đang chọn; kèm "Chọn tất cả" (Ctrl+A).</li>
  <li><b>[[file-text]] Sao chép câu gốc / [[a-large-small]] Sao chép phiên âm:</b> chép riêng từng phần vào clipboard (chỉ hiện mục nào có nội dung). Chọn nhiều câu thì chép mỗi câu một dòng.</li>
  <li><b>↩️ Đặt lại khung chữ (tự động):</b> bỏ khung bạn đã kéo.</li>
  <li><b>🎯 Áp dụng khung mới vào redraw:</b> dùng khung nhận diện đã chỉnh để xoá chữ gốc lại.</li>
  <li><b>🔤 Tuỳ chỉnh font…:</b> font riêng cho câu.</li>
  <li><b>[[message-square-text]] Hỏi chatbot về câu này:</b> mở hộp chat, tự phân tích từ vựng + ngữ pháp của câu rồi bạn hỏi tiếp (xem mục Chatbot trợ lý).</li>
  <li><b>🚫 Không redraw câu này / 🎨 Bật redraw:</b> giữ chữ gốc và chỉ chèn chữ dịch, hoặc ngược lại.</li>
  <li><b>❌ Xoá thoại này.</b></li>
  </ul></li>
</ul>
"""),
    ("Thêm câu thoại mới", """
<h2>Thêm câu thoại mới</h2>
<p>Dùng khi app bỏ sót một bóng thoại hoặc bạn muốn dọn chữ sót.</p>
<ol>
<li>Trên ảnh, chọn vùng: <b>giữ Shift và kéo chuột trái</b>, <b>kéo chuột phải</b>, hoặc dùng <b>[[wand]] Đũa thần</b> bấm vào một hay nhiều bóng thoại có viền liền rồi nhấn Enter.</li>
<li>Chọn cách thêm:
  <ul>
  <li><b>📷 Quét OCR tự động:</b> OCR và dịch vùng đó.</li>
  <li><b>✏️ Tự nhập text:</b> gõ câu gốc và câu dịch, không cần OCR hay API.</li>
  <li><b>🎨 Chỉ redraw (xoá chữ gốc):</b> chỉ xoá chữ trong vùng, không dịch; dùng cho chữ sót hoặc SFX.</li>
  </ul></li>
<li>Tick <b>Không redraw khung này</b> nếu chỉ muốn chèn chữ dịch mà giữ chữ gốc.</li>
</ol>
"""),
    ("Redraw (xoá chữ gốc)", """
<h2>Redraw</h2>
<ul>
<li>Chạy tự động ở nền sau khi dịch và khi sửa khung thoại (nếu bật). Thanh trạng thái ở tab Redraw báo "Redraw xong" khi hoàn tất.</li>
<li><b>Vẽ / Fill không kích hoạt redraw tự động</b> và không làm khung chữ bị xếp lại. Khi cần áp dụng (ví dụ hiện vùng Fill dạng lớp), bấm <b>Redraw lại trang này</b>.</li>
<li><b>Redraw lại trang này</b> (nút ở tab Redraw): làm lại trang đang xem. <b>Thêm ▾ → Chỉ redraw toàn bộ:</b> làm lại mọi trang, không dịch. Dùng sau khi đổi model xoá chữ.</li>
<li><b>Chuột phải trang → Chỉ redraw (không dịch):</b> xoá chữ lại mà không gọi API dịch.</li>
<li>Tab <b>Redraw</b> bên phải liệt kê nét vẽ và vùng Fill của trang: tick để hiện / ẩn, nút <b>Hiện / Ẩn mục đã chọn</b>, <b>Xoá mục đang chọn</b>.</li>
</ul>
"""),
    ("Project & lưu", """
<h2>Project và lưu</h2>
<ul>
<li>[[menu]] → Project → <b>Lưu JSON:</b> lưu toàn bộ (trang, câu thoại, khung, font, nét vẽ, Fill).</li>
<li>[[menu]] → Project → <b>Mở project (.json):</b> mở lại bộ truyện đã lưu. Nên mở khi ảnh vẫn còn ở vị trí cũ.</li>
<li><b>Nạp thêm ảnh / thư mục</b> để thêm trang vào project hiện tại.</li>
<li><b>Xoá trang hiện tại</b> chỉ bỏ trang khỏi project, không xoá file ảnh trên ổ đĩa. Chọn nhiều trang rồi chuột phải để xoá nhiều trang.</li>
<li>Vẽ / Fill thẳng lên ảnh chỉ sửa <b>bản làm việc</b> (lưu trong thư mục dữ liệu của app), file gốc luôn nguyên vẹn. Lưu JSON sẽ ghi nhớ trang nào đã chỉnh, và khi mở lại project trên cùng máy app tự dùng lại bản đã chỉnh. Xuất ảnh / PSD dùng ảnh đã chỉnh.</li>
</ul>
"""),
    ("Xuất ảnh / PSD / TXT", """
<h2>Xuất ảnh, PSD, TXT</h2>
<p>[[menu]] → <b>Xuất</b>:</p>
<ul>
<li><b>Ảnh — trang hiện tại</b> (<b>Ctrl+Shift+E</b>): một ảnh PNG / JPG / WebP.</li>
<li><b>Ảnh — tất cả trang:</b> xuất hàng loạt vào thư mục bạn chọn.</li>
<li><b>PSD — trang hiện tại / tất cả:</b> file Photoshop nhiều lớp (ảnh gốc, vùng xoá chữ, nét vẽ, từng câu chữ dịch). Lớp bị ẩn trong app vẫn ẩn trong PSD.</li>
<li><b>TXT (bản dịch):</b> xuất toàn bộ câu dịch ra file văn bản.</li>
<li><b>Mọi trang đều được xuất</b>, kể cả trang chưa redraw hay chưa chỉnh sửa gì: những trang đó ra là ảnh gốc (PSD chỉ có lớp ảnh gốc), nên bộ ảnh xuất ra luôn đủ số trang, không bị thiếu trang.</li>
<li>Hai trang trùng tên gốc không ghi đè nhau (thêm _2, _3…).</li>
<li>Phần chữ nằm ngoài trang (kéo khung ra ngoài) bị cắt khi xuất.</li>
</ul>
"""),
    ("Chatbot trợ lý", """
<h2>Chatbot trợ lý</h2>
<p>Bấm <b>bong bóng tròn</b> ở góc dưới bên phải để mở hộp chat (bấm lại hoặc bấm ✕ để đóng; lịch sử vẫn giữ trong phiên). Bong bóng chỉ hiện khi bạn đã vào <b>Workspace</b>, <b>không hiện ở Trang chủ</b> (màn nạp ảnh). Kéo bong bóng để dời sang chỗ khác nếu nó che thứ bạn cần. Chấm đỏ trên bong bóng báo có câu trả lời mới khi hộp chat đang đóng. Chatbot cần <b>API key Gemini</b> (xem mục Tạo API key Gemini).</p>
<h3>Chatbot trả lời được gì</h3>
<ul>
<li><b>Cách dùng app:</b> nạp ảnh, dịch, redraw, xuất file, phím tắt, xử lý lỗi. Ví dụ: "Làm sao xuất PSD?".</li>
<li><b>Nội dung trang đang xem:</b> tóm tắt, ai nói gì, câu dịch nào chưa sát. Ví dụ: "Trang này nói gì?". Chatbot đọc chữ gốc và bản dịch trong bảng thoại; hỏi về trang khác cũng được (ví dụ "trang trước nói gì?", "trang 12", "tóm tắt từ đầu tới giờ"). Trang chưa dịch hoặc khi bạn hỏi về hình ảnh / biểu cảm / cảnh, chatbot sẽ nhìn ảnh trang (tối đa 3 ảnh mỗi lượt).</li>
<li><b>Tác phẩm gốc:</b> giới thiệu bộ truyện, tác giả, nhân vật (có tra cứu web). Chatbot đoán tên bộ truyện từ tên thư mục ảnh; nếu đoán sai, cứ nhắn tên bộ truyện trong khung chat. Mặc định chatbot tránh tiết lộ cốt truyện.</li>
<li><b>Ngôn ngữ:</b> ngữ pháp, từ vựng, cách đọc, sắc thái, cách dịch. Dán thẳng câu cần hỏi, hoặc chuột phải một câu trong bảng thoại → <b>Hỏi chatbot về câu này</b> (chatbot tự mở đầu bằng phân tích từ vựng + ngữ pháp, mỗi câu một cuộc trò chuyện riêng).</li>
</ul>
<h3>Chọn nhân vật để trò chuyện</h3>
<p>Bấm <b>tên ở đầu hộp chat</b> (có mũi tên ⌄) để chọn nhân vật. Mặc định là <b>Trợ lý dịch truyện</b> (trả lời bốn nhóm trên). App còn kèm sẵn nhân vật nhập vai (ví dụ Sengoku Yuno) với giọng điệu riêng; mỗi nhân vật có cuộc trò chuyện riêng và avatar hiện trên bong bóng. Nhân vật là các thư mục chứa file <code>profile.json</code> trong thư mục <code>characters</code> cạnh app, hoặc trong thư mục dữ liệu của bạn.</p>
<h3>Ngoài phạm vi</h3>
<p>Chatbot trợ lý chỉ trả lời bốn nhóm trên. Các chủ đề khác (thời tiết, tài chính, lập trình, viết bài…) sẽ được từ chối lịch sự.</p>
<h3>Mẹo</h3>
<ul>
<li>Nút <b>☰</b> trong hộp chat: chuyển giữa "Trò chuyện chung" và các câu đã hỏi, hoặc xoá cuộc trò chuyện hiện tại. Nút <b>–</b> thu nhỏ hộp chat thành thanh tiêu đề.</li>
<li><b>Enter</b> gửi, <b>Shift+Enter</b> xuống dòng, <b>Esc</b> đóng hộp chat.</li>
</ul>
"""),
]

EXTRA: List[Tuple[str, str, str]] = [
    ("⌨️ Phím tắt", "Phím tắt", """
<h2>Phím tắt</h2>
<table border="0" cellpadding="4">
<tr><td><b>Ctrl+T</b></td><td>Đổi nhanh giữa ảnh gốc và bản dịch</td></tr>
<tr><td><b>Ctrl+F</b></td><td>Tìm chữ trong bản dịch / câu gốc của mọi trang (Enter: kết quả kế, Shift+Enter: kết quả trước)</td></tr>
<tr><td><b>Ctrl+H</b></td><td>Tìm và thay thế trong bản dịch (Thay câu này / Thay tất cả, theo mọi trang hoặc trang hiện tại)</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Bật / tắt [[square-dashed]] Khung</td></tr>
<tr><td><b>Ctrl+D</b></td><td>Bật / tắt [[brush]] Vẽ</td></tr>
<tr><td><b>Ctrl+G</b></td><td>Bật / tắt [[wand-sparkles]] Fill</td></tr>
<tr><td><b>Ctrl+Shift+W</b></td><td>Bật / tắt [[wand]] Đũa thần (bấm nhiều bóng để chọn nhiều; bấm lại hoặc Alt + bấm: bỏ bóng; Ctrl + bấm: chọn lại; Enter: tạo câu thoại; Esc: bỏ chọn)</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Hoàn tác (nét vẽ / Fill / vẽ thẳng)</td></tr>
<tr><td><b>Ctrl+Y</b>, <b>Ctrl+Shift+Z</b></td><td>Làm lại</td></tr>
<tr><td><b>Ctrl+[</b> / <b>Ctrl+]</b></td><td>Thu gọn / mở danh sách trang / bảng thoại</td></tr>
<tr><td><b>Ctrl+Shift+E</b></td><td>Xuất ảnh trang hiện tại</td></tr>
<tr><td><b>Ctrl + bấm khung</b></td><td>Chọn thêm / bỏ một câu trong tập đang chọn</td></tr>
<tr><td><b>Shift + kéo chuột trái</b> hoặc <b>kéo chuột phải</b></td><td>Khoanh vùng để thêm câu thoại mới</td></tr>
<tr><td><b>Shift+Enter</b></td><td>Xuống dòng khi sửa câu dịch</td></tr>
<tr><td><b>Delete</b></td><td>Xoá trang đang chọn trong danh sách trang</td></tr>
</table>
"""),
    ("🩺 Xử lý sự cố", "Xử lý sự cố", """
<h2>Xử lý sự cố</h2>
<h3>Nơi lưu log</h3>
<p>Thư mục <code>logs</code> trong thư mục dữ liệu của app (AppData): <code>app.log</code> (mọi log và lỗi), <code>crash.log</code> (dấu vết khi app bị sập hoặc treo), <code>stalls.log</code> (các lần giao diện bị khựng). Gửi các file này khi cần báo lỗi.</p>
<h3>Một số tình huống thường gặp</h3>
<ul>
<li><b>Báo lỗi API key / không tìm thấy model:</b> mở [[menu]] → Thiết lập → Dịch thuật, kiểm tra key (không thừa khoảng trắng) và chọn model khác. Chưa có key hoặc key cũ bị từ chối (403): tạo key mới theo mục <i>Thiết lập → Tạo API key Gemini</i>.</li>
<li><b>"Không nạp được Manga-OCR … Unable to load vocabulary from file":</b> thường do môi trường Python, không phải do app. (1) Nếu <code>pip show transformers</code> ra bản 5.x: chạy <code>pip install "transformers&lt;5"</code> rồi mở lại app. (2) Xoá thư mục <code>models--mayocream--manga-ocr</code> và <code>models--kha-white--manga-ocr-base</code> trong thư mục cache Hugging Face (<code>%USERPROFILE%/.cache/huggingface/hub</code>) để tải lại. (3) Tên người dùng Windows có dấu: đặt biến môi trường <code>HF_HOME</code> trỏ tới thư mục không dấu. Có thể trỏ ô <i>Model Manga-OCR</i> trong Thiết lập tới thư mục model đã tải sẵn.</li>
<li><b>Báo hết hạn mức / quá nhiều yêu cầu (429):</b> bản miễn phí có giới hạn lượt dùng. Đợi một lúc rồi dịch tiếp, giảm số trang dịch cùng lúc, hoặc đổi model.</li>
<li><b>Vẽ / Fill xong mà chưa thấy trên bản dịch:</b> Vẽ và Fill không tự redraw. Bấm <b>Redraw lại trang này</b> ở tab Redraw bên phải.</li>
<li><b>Không thấy bong bóng chatbot:</b> bong bóng chỉ hiện trong Workspace, không hiện ở Trang chủ. Nạp ảnh để vào Workspace.</li>
<li><b>Trang đầu tiên dịch rất chậm:</b> app đang nạp model lần đầu (Manga-OCR, YOLO). Các trang sau nhanh hơn. Tải model về thư mục local để khởi động nhanh hơn.</li>
<li><b>Chữ dịch không có dấu hoặc sai font:</b> chọn font có hỗ trợ tiếng Việt; font không đủ dấu sẽ bị ẩn khỏi danh sách.</li>
<li><b>Chữ thò ra ngoài bóng thoại:</b> kéo khung chữ ([[square-dashed]] Khung) hoặc giảm cỡ chữ ở thanh định dạng.</li>
<li><b>Còn sót chữ gốc:</b> dùng bút dạ trắng, hoặc [[wand-sparkles]] Fill, hoặc khoanh vùng rồi chọn "Chỉ redraw".</li>
<li><b>Muốn bỏ nét vẽ / Fill đã in thẳng vào ảnh:</b> Ctrl+Z hoặc ☰ → Project → Bỏ chỉnh sửa trực tiếp của trang.</li>
</ul>
"""),
]
