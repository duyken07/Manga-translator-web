"""
AppState = "Global State" của ứng dụng (tương đương Redux/Zustand phía Web).

Toàn bộ UI (Dashboard, Workspace) chỉ đọc / phát lệnh thay đổi thông qua
đối tượng AppState duy nhất này, đảm bảo dữ liệu không bị mất khi chuyển
trang hay chuyển màn hình (Auto-save in-memory).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

from app.qt_compat import QObject, Signal

from app.characters.character_model import DEFAULT_CHARACTER_ID, Character
from app.models import Project, Page, Dialogue
from app.services.chat_service import ChatSessionStore


class AppState(QObject):
    # --- Signals để UI tự cập nhật khi state thay đổi ---
    project_changed = Signal()
    current_page_changed = Signal(int)  # page_index mới
    page_updated = Signal(int)  # page_index đã có dữ liệu dịch/chỉnh sửa mới
    dirty_changed = Signal(bool)
    # Nội dung thoại của 1 trang thay đổi (dịch xong, sửa chữ dịch, thêm, xoá) -> cần redraw lại trang đó
    dialogues_changed = Signal(int)
    # Trạng thái / ảnh redraw của 1 trang thay đổi
    redraw_updated = Signal(int)
    # Khung thoại (box_location) bị kéo thả / đổi cỡ: KHÔNG kéo theo redraw và không nạp lại trang
    dialogue_box_changed = Signal(int, int)  # page_index, dialogue_id
    # Ẩn / hiện chữ hoặc vùng redraw của 1 câu (dialogue_id = 0 nghĩa là nhiều câu): không chạy lại redraw
    visibility_changed = Signal(int, int)  # page_index, dialogue_id
    # Metadata của 1 câu đổi (font riêng, giãn dòng...) mà không cần nạp lại trang / ảnh
    dialogue_meta_changed = Signal(int, int)  # page_index, dialogue_id
    # Như trên nhưng cho NHIỀU câu cùng lúc (chỉnh định dạng hàng loạt): page_index, [dialogue_id...]
    dialogues_meta_changed = Signal(int, list)
    # Nét vẽ tay của 1 trang đổi (thêm / xoá / hoàn tác / làm lại)
    strokes_changed = Signal(int)  # page_index
    # Danh sách vùng Content-Aware Fill của 1 trang đổi (thêm / xoá / ẩn)
    fills_changed = Signal(int)  # page_index
    # Nhân vật chatbot đang chọn đổi (id mới). Cài đặt toàn cục của app, KHÔNG lưu trong project
    character_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.current_page_index: int = -1
        self._dirty: bool = False
        # Lịch sử chatbot ngôn ngữ của PHIÊN này: chỉ trong bộ nhớ, không lưu vào project
        self.chat_sessions = ChatSessionStore()
        # Nhân vật chatbot đang chọn (mặc định 'assistant' = trợ lý cũ nên hành vi không đổi cho tới khi giao diện chọn nhân vật khác)
        self.active_character_id: str = DEFAULT_CHARACTER_ID
        self.active_character: Optional[Character] = None

    # -- Nhân vật chatbot ------------------------------------------------------
    def set_active_character(self, profile: Character) -> None:
        """Đổi nhân vật đang chọn: cập nhật state + kho hội thoại (mỗi nhân vật có lịch sử riêng) rồi báo cho giao diện.
        Việc kiểm tra tồn tại / đang bật là của CharacterManager.get_enabled_character()."""
        self.active_character = profile
        self.active_character_id = profile.id
        self.chat_sessions.active_character_id = profile.id
        self.character_changed.emit(profile.id)

    # -- Project lifecycle ---------------------------------------------------
    def new_project(self, source_lang: str, target_lang: str) -> None:
        self.project = Project(source_lang=source_lang, target_lang=target_lang)
        self.current_page_index = -1
        self._set_dirty(False)
        self.project_changed.emit()

    def load_project(self, project: Project) -> None:
        self.project = project
        self.current_page_index = 0 if project.pages else -1
        self._set_dirty(False)
        self.project_changed.emit()

    def add_images(self, image_paths: List[str], status: Optional[str] = None) -> None:
        """status: đặt luôn trạng thái cho các trang vừa thêm (VD "done" khi chỉ nạp ảnh để tự làm thủ công, không dịch)."""
        n_before = len(self.project.pages)
        self.project.add_images(image_paths)
        if status:
            for pg in self.project.pages[n_before:]:
                pg.status = status
        if self.current_page_index == -1 and self.project.pages:
            self.current_page_index = 0
        self._set_dirty(True)
        self.project_changed.emit()

    def add_more_images(self, image_paths: List[str]) -> tuple[int, int]:
        """Nạp THÊM ảnh vào project đang mở. Trang mới xen vào đúng chỗ theo Natural Sort của tên file (như mọi project);
        trang đang xem vẫn là trang đó. Ảnh trùng tên file với trang đã có bị bỏ qua. Trả (số trang thêm, số ảnh bị bỏ qua).
        Chỉ số trang có thể dời -> người gọi phải chắc không còn việc dịch / redraw nào đang chạy theo chỉ số."""
        current = self.current_page()
        before = len(self.project.pages)
        self.project.add_images(image_paths)
        added = len(self.project.pages) - before
        if added <= 0:
            return 0, len(image_paths)
        pages = self.project.pages
        self.current_page_index = next((i for i, p in enumerate(pages) if p is current), 0)
        self._set_dirty(True)
        self.project_changed.emit()
        return added, len(image_paths) - added

    def remove_pages(self, indices: List[int]) -> List[Page]:
        """Xoá các trang khỏi project (không đụng tới file ảnh trên ổ đĩa). Trang đang xem bị xoá thì chuyển sang trang gần nhất
        còn lại. Trả danh sách Page đã xoá (để dọn cache redraw)."""
        n = len(self.project.pages)
        drop = sorted({i for i in indices if 0 <= i < n})
        if not drop:
            return []
        drop_set = set(drop)
        current = self.current_page()
        removed = [self.project.pages[i] for i in drop]
        kept = [p for i, p in enumerate(self.project.pages) if i not in drop_set]
        for p in removed:
            self.project.removed_names.add(p.page_name)
        self.project.pages = kept
        current_removed = current is not None and any(p is current for p in removed)
        if not kept:
            self.current_page_index = -1
        elif current_removed or current is None:
            self.current_page_index = min(drop[0], len(kept) - 1)
        else:
            self.current_page_index = next(i for i, p in enumerate(kept) if p is current)
        self._set_dirty(True)
        self.project_changed.emit()
        if current_removed and self.current_page_index >= 0:
            self.current_page_changed.emit(self.current_page_index)
        return removed

    # -- Page navigation -------------------------------------------------------
    def go_to_page(self, index: int) -> None:
        if 0 <= index < len(self.project.pages):
            self.current_page_index = index
            self.current_page_changed.emit(index)

    def next_page(self) -> None:
        self.go_to_page(self.current_page_index + 1)

    def prev_page(self) -> None:
        self.go_to_page(self.current_page_index - 1)

    def current_page(self) -> Optional[Page]:
        if 0 <= self.current_page_index < len(self.project.pages):
            return self.project.pages[self.current_page_index]
        return None

    # -- Update kết quả dịch từ Queue -----------------------------------------
    def apply_translated_page(self, page_index: int, translated_page: Page) -> None:
        if 0 <= page_index < len(self.project.pages):
            existing = self.project.pages[page_index]
            existing.dialogues = translated_page.dialogues
            existing.status = "done"
            self._set_dirty(True)
            if translated_page.glossary_updates:
                self.update_glossary(translated_page.glossary_updates)
            self.page_updated.emit(page_index)
            self.dialogues_changed.emit(page_index)

    def apply_retranslated_page(self, page_index: int, translated_page: Page, keep_manual: bool = True,
                                no_redraw: bool = False) -> int:
        """Kết quả DỊCH LẠI một trang: thay các câu tự động bằng bản dịch mới. keep_manual: giữ lại các câu người dùng đã gõ / sửa tay
        (edited=True) và vùng chỉ redraw, xếp sau các câu mới. no_redraw: các khung MỚI chỉ tạo khung + bản dịch, không redraw. Đánh số lại
        dialogue_id từ 1. Nét vẽ tay và vùng Fill của trang được giữ nguyên. Trả số câu mới."""
        if not (0 <= page_index < len(self.project.pages)):
            return 0
        page = self.project.pages[page_index]
        kept = [d for d in page.dialogues if keep_manual and (d.edited or d.redraw_only)]
        fresh = list(translated_page.dialogues)
        if no_redraw:
            for d in fresh:
                d.no_redraw = True
        combined = fresh + kept
        for i, d in enumerate(combined, 1):
            d.dialogue_id = i
        page.dialogues = combined
        page.status = "done"
        page.error_message = None
        self._set_dirty(True)
        if translated_page.glossary_updates:
            self.update_glossary(translated_page.glossary_updates)
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)
        return len(fresh)

    def set_dialogue_no_redraw(self, page_index: int, dialogue_id: int, on: bool) -> None:
        """Bật / tắt "khung chỉ để dịch" (không redraw) cho một câu."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id and d.no_redraw != on:
                d.no_redraw = on
                self._set_dirty(True)
                self.dialogue_meta_changed.emit(page_index, dialogue_id)   # bảng vẽ lại đúng dòng, không nạp lại trang
                self.dialogues_changed.emit(page_index)                    # redraw đồng bộ (xoá / thêm phần chênh lệch)
                return

    def update_glossary(self, updates: dict) -> None:
        """Gộp thuật ngữ mới phát hiện (từ 1 trang vừa dịch) vào glossary chung của Project."""
        if not updates:
            return
        before = dict(self.project.glossary)
        self.project.update_glossary(updates)
        if self.project.glossary != before:
            self._set_dirty(True)

    def cache_grammar_result(self, sentence: str, data: dict) -> None:
        """Lưu kết quả phân tích ngữ pháp của 1 câu vào cache của Project, đánh dấu dirty để nhắc Lưu JSON."""
        self.project.cache_grammar_result(sentence, data)
        self._set_dirty(True)

    def get_cached_grammar_result(self, sentence: str) -> Optional[dict]:
        return self.project.grammar_cache.get(sentence)

    def set_work_title(self, title: str) -> None:
        """Tên tác phẩm gốc (cho chatbot). Lưu cùng project nên đánh dấu có thay đổi chưa lưu."""
        title = " ".join((title or "").split())
        if title != self.project.work_title:
            self.project.work_title = title
            self._set_dirty(True)

    def mark_page_error(self, page_index: int, message: str) -> None:
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].status = "error"
            self.project.pages[page_index].error_message = message
            self.page_updated.emit(page_index)

    def restore_page_status(self, page_index: int, status: str) -> None:
        """Trả trạng thái cũ cho trang (VD dịch lại thất bại thì trang vẫn dùng được với các câu cũ)."""
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].status = status
            self.project.pages[page_index].error_message = None
            self.page_updated.emit(page_index)

    def mark_page_translating(self, page_index: int) -> None:
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].status = "translating"
            self.page_updated.emit(page_index)

    # -- Editor: chỉnh sửa thủ công câu dịch, kéo thả thứ tự & xóa thoại ------
    def update_dialogue_translation(self, page_index: int, dialogue_id: int, new_text: str) -> None:
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                d.translated_text = new_text
                d.edited = True
                self._set_dirty(True)
                # KHÔNG phát page_updated: ô bảng đã hiện chữ mới, nạp lại cả trang sẽ làm mất zoom / lựa chọn
                self.dialogue_meta_changed.emit(page_index, dialogue_id)
                self.dialogues_changed.emit(page_index)
                return

    def replace_translations(self, mapping: Dict[int, Dict[int, str]]) -> int:
        """Đặt lại bản dịch cho NHIỀU câu ở NHIỀU trang trong một lần (Tìm / Thay thế): {page_index: {dialogue_id: văn bản mới}}.
        Mỗi trang chỉ phát tín hiệu một lần (một đợt xếp lại chữ cho trang đó). Trả về số câu thật sự đổi."""
        total = 0
        for page_index, items in mapping.items():
            page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
            if not page or not items:
                continue
            changed: List[int] = []
            for d in page.dialogues:
                new = items.get(d.dialogue_id)
                if new is None or new == d.translated_text:
                    continue
                d.translated_text = new
                d.edited = True
                changed.append(d.dialogue_id)
            if changed:
                total += len(changed)
                self._set_dirty(True)
                self.dialogues_meta_changed.emit(page_index, changed)  # vẽ lại dòng trong bảng, không nạp lại trang
                self.dialogues_changed.emit(page_index)                # xếp lại chữ trên ảnh
        return total

    def flatten_dialogue_lines(self, page_indices=None) -> int:
        """Đưa câu gốc / phiên âm / bản dịch của các trang về MỘT dòng (nối các chỗ ngắt hàng; kể cả xuống dòng tự gõ bằng Shift+Enter).
        page_indices=None: mọi trang. Trả về số câu thật sự đổi. Dùng để dọn project cũ; câu mới từ Gemini đã tự được nối khi nhận về."""
        wanted = None if page_indices is None else set(page_indices)
        total = 0
        for pi, page in enumerate(self.project.pages):
            if wanted is not None and pi not in wanted:
                continue
            changed: List[int] = []
            for d in page.dialogues:
                before = (d.original_text, d.romanization, d.translated_text)
                d.flatten_lines()
                if (d.original_text, d.romanization, d.translated_text) != before:
                    changed.append(d.dialogue_id)
            if not changed:
                continue
            total += len(changed)
            self._set_dirty(True)
            if pi == self.current_page_index:
                self.page_updated.emit(pi)                              # nạp lại trang đang xem (bảng thoại đổi chiều cao dòng)
            else:
                self.dialogues_meta_changed.emit(pi, changed)
            self.dialogues_changed.emit(pi)                             # xếp lại chữ trên ảnh theo bản dịch mới
        return total

    def update_dialogue_overrides(self, page_index: int, dialogue_id: int, overrides: dict) -> None:
        """Đặt tuỳ chỉnh redraw (font, cỡ chữ, kiểu, màu...) riêng cho 1 câu thoại; dùng chung bộ khoá với JSON."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                if d.overrides == overrides:
                    return
                d.overrides = dict(overrides)
                self._set_dirty(True)
                self.dialogue_meta_changed.emit(page_index, dialogue_id)  # không nạp lại trang: giữ zoom / lựa chọn
                self.dialogues_changed.emit(page_index)  # chỉ xếp lại chữ, không xoá lại chữ gốc
                return

    def update_dialogues_overrides(self, page_index: int, mapping: Dict[int, dict]) -> int:
        """Đặt tuỳ chỉnh (font, cỡ chữ, viền...) cho NHIỀU câu trong MỘT lần: {dialogue_id: overrides đầy đủ}.
        Chỉ phát tín hiệu một lần (không phải mỗi câu một lần) nên chỉ có một đợt xếp lại chữ. Trả về số câu thật sự đổi."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page or not mapping:
            return 0
        changed: List[int] = []
        for d in page.dialogues:
            ov = mapping.get(d.dialogue_id)
            if ov is None or d.overrides == ov:
                continue
            d.overrides = dict(ov)
            changed.append(d.dialogue_id)
        if not changed:
            return 0
        self._set_dirty(True)
        self.dialogues_meta_changed.emit(page_index, changed)  # không nạp lại trang: giữ zoom / lựa chọn
        self.dialogues_changed.emit(page_index)  # chỉ xếp lại chữ, không xoá lại chữ gốc
        return len(changed)

    # -- Khung thoại: kéo thả / đổi cỡ độc lập với ảnh redraw -------------------------------
    def update_dialogue_box(self, page_index: int, dialogue_id: int, box: List[int]) -> None:
        """Đổi vị trí / kích thước khung [x, y, w, h]. Ảnh redraw giữ nguyên (bám khung neo `redraw_box`)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page or len(box) != 4:
            return
        box = [int(round(v)) for v in box]
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                if list(d.box_location) == box:
                    return
                if d.redraw_box is None and page.redraw_path and d.translated_text.strip():
                    d.redraw_box = list(d.box_location)  # đóng băng khung đã dùng cho redraw trước khi đổi
                d.box_location = box
                self._set_dirty(True)
                self.dialogue_box_changed.emit(page_index, dialogue_id)  # cố ý không phát page_updated / dialogues_changed
                return

    def freeze_redraw_boxes(self, page_index: int) -> None:
        """Gọi ngay trước khi redraw trang: ghi nhớ khung hiện tại làm khung neo cho các câu có bản dịch."""
        if 0 <= page_index < len(self.project.pages):
            for d in self.project.pages[page_index].dialogues:
                if d.redraw_box is None and d.translated_text.strip():
                    d.redraw_box = list(d.box_location)

    def apply_dialogue_box_to_redraw(self, page_index: int, dialogue_id: int) -> bool:
        """Cho redraw dùng khung hiện tại của câu này (xoá chữ gốc ở vị trí mới, trả lại ảnh gốc ở vị trí cũ)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return False
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                if d.redraw_box is None or list(d.redraw_box) == list(d.box_location):
                    return False
                d.redraw_box = None
                self._set_dirty(True)
                self.dialogues_changed.emit(page_index)
                return True
        return False

    def reset_redraw_boxes(self, page_index: int) -> None:
        """Bỏ mọi khung neo của trang (redraw lại từ đầu sẽ dùng các khung hiện tại)."""
        if 0 <= page_index < len(self.project.pages):
            for d in self.project.pages[page_index].dialogues:
                d.redraw_box = None

    def reorder_dialogues(self, page_index: int, new_order_ids: List[int]) -> None:
        """Sắp xếp lại thứ tự dialogues của một trang và đánh lại STT (1..N)."""
        if not (0 <= page_index < len(self.project.pages)):
            return

        page = self.project.pages[page_index]
        if not page or not page.dialogues:
            return

        dialogue_map = {d.dialogue_id: d for d in page.dialogues}
        reordered_dialogues = []

        for d_id in new_order_ids:
            if d_id in dialogue_map:
                reordered_dialogues.append(dialogue_map[d_id])

        # Đảm bảo không bỏ sót bất kỳ thoại nào nếu mảng ID gửi lên bị thiếu
        for d in page.dialogues:
            if d not in reordered_dialogues:
                reordered_dialogues.append(d)

        # Cập nhật lại dialogue_id nối tiếp 1, 2, 3...
        for idx, d in enumerate(reordered_dialogues, start=1):
            d.dialogue_id = idx

        page.dialogues = reordered_dialogues
        self._set_dirty(True)
        self.page_updated.emit(page_index)

    # Khoá override là ĐỊNH DẠNG chữ (font, cỡ, kiểu, màu, viền, căn lề, giãn dòng). Các khoá còn lại (type, fill, region, keep,
    # polarity, cleanup, engine) chỉ phục vụ việc xoá chữ gốc nên KHÔNG chép sang khung nhân bản.
    FORMAT_OVERRIDE_KEYS = ("style", "font_size", "font_path", "text_color", "line_spacing",
                            "outline", "outline_width", "outline_color", "align")

    def duplicate_dialogues(self, page_index: int, dialogue_ids: List[int], offset=(36, 36)) -> List[int]:
        """Nhân bản khung chữ: chép khung chữ + góc xoay + bản dịch + định dạng sang một câu mới, nhưng KHÔNG chép vùng xoá chữ gốc
        (câu mới là khung "chỉ chèn chữ": no_redraw). Khung mới lệch `offset` px để nhìn thấy ngay. Trả về id các câu mới."""
        if not (0 <= page_index < len(self.project.pages)):
            return []
        page = self.project.pages[page_index]
        by_id = {d.dialogue_id: d for d in page.dialogues}
        dx, dy = offset
        new_ids: List[int] = []
        for did in dialogue_ids:
            src = by_id.get(did)
            if src is None or src.redraw_only:      # vùng "chỉ redraw" không có chữ để nhân bản
                continue
            x, y, w, h = [int(round(v)) for v in (src.text_frame or src.redraw_box or src.box_location)]
            frame = [x + dx, y + dy, w, h]
            new = Dialogue(
                dialogue_id=0,
                box_location=list(frame),
                original_text=src.original_text,
                romanization=src.romanization,
                translated_text=src.translated_text,
                edited=True,
                overrides={k: copy.deepcopy(v) for k, v in src.overrides.items() if k in self.FORMAT_OVERRIDE_KEYS},   # sao chép sâu: màu là list
                text_frame=list(frame),
                text_rotation=src.text_rotation,
                hide_text=False,
                no_redraw=True,                       # không xoá chữ gốc ở vị trí mới
            )
            new.dialogue_id = max((d.dialogue_id for d in page.dialogues), default=0) + 1
            page.dialogues.append(new)
            new_ids.append(new.dialogue_id)
        if new_ids:
            self._set_dirty(True)
            self.page_updated.emit(page_index)
            self.dialogues_changed.emit(page_index)
        return new_ids

    def delete_dialogues(self, page_index: int, dialogue_ids: List[int]) -> None:
        """Xoá nhiều thoại cùng lúc (một lần đánh lại STT, một lần phát tín hiệu cập nhật + redraw)."""
        if not (0 <= page_index < len(self.project.pages)):
            return
        page = self.project.pages[page_index]
        ids = set(dialogue_ids)
        if not page or not ids or not any(d.dialogue_id in ids for d in page.dialogues):
            return
        page.dialogues = [d for d in page.dialogues if d.dialogue_id not in ids]
        for idx, d in enumerate(page.dialogues, start=1):
            d.dialogue_id = idx
        self._set_dirty(True)
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)

    def delete_dialogue(self, page_index: int, dialogue_id: int) -> None:
        """Xóa một thoại khỏi trang hiện tại và cập nhật lại STT nối tiếp."""
        if not (0 <= page_index < len(self.project.pages)):
            return

        page = self.project.pages[page_index]
        if not page or not page.dialogues:
            return

        page.dialogues = [d for d in page.dialogues if d.dialogue_id != dialogue_id]

        # Đánh lại ID theo thứ tự nối tiếp 1..N
        for idx, d in enumerate(page.dialogues, start=1):
            d.dialogue_id = idx

        self._set_dirty(True)
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)

    def add_dialogues(self, page_index: int, dialogues: List[Dialogue]) -> List[int]:
        """Thêm nhiều thoại cùng lúc theo thứ tự đã cho (một lần phát tín hiệu cập nhật + redraw). Trả về các dialogue_id mới."""
        if not (0 <= page_index < len(self.project.pages)) or not dialogues:
            return []
        page = self.project.pages[page_index]
        new_ids: List[int] = []
        for d in dialogues:
            d.dialogue_id = max((x.dialogue_id for x in page.dialogues), default=0) + 1
            page.dialogues.append(d)
            new_ids.append(d.dialogue_id)
        self._set_dirty(True)
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)
        return new_ids

    def add_dialogue(self, page_index: int, dialogue: Dialogue) -> None:
        """
        Thêm 1 Dialogue mới (thường đến từ kết quả dịch 1 vùng ảnh người dùng
        tự khoanh) vào cuối danh sách thoại của trang. Tự cấp dialogue_id tiếp
        theo, bỏ qua giá trị dialogue_id tạm (0) mà translate_region() trả về.
        """
        if not (0 <= page_index < len(self.project.pages)):
            return

        page = self.project.pages[page_index]
        if page is None:
            return

        new_id = (max((d.dialogue_id for d in page.dialogues), default=0)) + 1
        dialogue.dialogue_id = new_id
        # Giữ nguyên dialogue.edited do nơi gọi quyết định: False cho kết quả
        # OCR/API mới (chưa qua chỉnh sửa), True cho câu nhập tay thủ công
        # (để bảng thoại highlight vàng, phân biệt rõ với câu tự động).

        page.dialogues.append(dialogue)
        self._set_dirty(True)

        # Phát signal để UI đồng bộ nạp lại dòng mới
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)  # redraw đúng vị trí khung thoại mới

    # -- Redraw -----------------------------------------------------------------
    # -- Nét vẽ tay (bút, bút dạ, dạ quang, tẩy) ------------------------------------------------
    def _page(self, page_index: int):
        return self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None

    def _strokes_done(self, page_index: int) -> None:
        self._set_dirty(True)
        self.strokes_changed.emit(page_index)

    def add_stroke(self, page_index: int, stroke: dict) -> None:
        page = self._page(page_index)
        if page is None or not stroke.get("pts"):
            return
        stroke = dict(stroke, id=max((s["id"] for s in page.strokes), default=0) + 1)
        page.strokes.append(stroke)
        page.stroke_hist.append(("add", [stroke]))
        page.stroke_redo.clear()
        self._strokes_done(page_index)

    def remove_strokes(self, page_index: int, ids) -> None:
        """Xoá các nét theo id (công cụ tẩy / xoá hết). Hoàn tác được."""
        page = self._page(page_index)
        if page is None:
            return
        ids = set(ids)
        removed = [s for s in page.strokes if s["id"] in ids]
        if not removed:
            return
        page.strokes = [s for s in page.strokes if s["id"] not in ids]
        page.stroke_hist.append(("del", removed))
        page.stroke_redo.clear()
        self._strokes_done(page_index)

    def set_stroke_hidden(self, page_index: int, stroke_id: int, hidden: bool) -> None:
        """Ẩn / hiện 1 nét vẽ (chỉ đổi hiển thị và xuất; nét vẫn nằm trong project). Không ghi vào lịch sử hoàn tác."""
        page = self._page(page_index)
        if page is None:
            return
        for s in page.strokes:
            if s["id"] == stroke_id and bool(s.get("hidden")) != hidden:
                if hidden:
                    s["hidden"] = True
                else:
                    s.pop("hidden", None)
                self._strokes_done(page_index)
                return

    def clear_strokes(self, page_index: int) -> None:
        page = self._page(page_index)
        if page is not None:
            self.remove_strokes(page_index, [s["id"] for s in page.strokes])

    def can_undo_stroke(self, page_index: int) -> bool:
        page = self._page(page_index)
        return bool(page and page.stroke_hist)

    def can_redo_stroke(self, page_index: int) -> bool:
        page = self._page(page_index)
        return bool(page and page.stroke_redo)

    def undo_stroke(self, page_index: int) -> bool:
        page = self._page(page_index)
        if page is None or not page.stroke_hist:
            return False
        op, items = page.stroke_hist.pop()
        self._apply_stroke_op(page, op, items, inverse=True)
        page.stroke_redo.append((op, items))
        self._strokes_done(page_index)
        return True

    def redo_stroke(self, page_index: int) -> bool:
        page = self._page(page_index)
        if page is None or not page.stroke_redo:
            return False
        op, items = page.stroke_redo.pop()
        self._apply_stroke_op(page, op, items, inverse=False)
        page.stroke_hist.append((op, items))
        self._strokes_done(page_index)
        return True

    @staticmethod
    def _apply_stroke_op(page, op: str, items: list, inverse: bool) -> None:
        add = (op == "add") != inverse  # "add" làm lại / "del" hoàn tác -> thêm nét; ngược lại -> bỏ nét
        ids = {s["id"] for s in items}
        if add:
            page.strokes = sorted([s for s in page.strokes if s["id"] not in ids] + [dict(s) for s in items],
                                  key=lambda s: s["id"])
        else:
            page.strokes = [s for s in page.strokes if s["id"] not in ids]

    # -- Content-Aware Fill ------------------------------------------------------------------------
    def add_fill(self, page_index: int, strokes: list, engine: str = "auto") -> int:
        """Thêm vùng fill từ các nét bôi [{pts, width}]; trả về id (0 nếu rỗng). Kích hoạt redraw nền."""
        page = self._page(page_index)
        strokes = [dict(pts=[list(map(int, p)) for p in s["pts"]], width=float(s["width"])) for s in strokes if s.get("pts")]
        if page is None or not strokes:
            return 0
        fid = max((f["id"] for f in page.fills), default=0) + 1
        page.fills.append(dict(id=fid, strokes=strokes, engine=engine, hidden=False))
        self._set_dirty(True)
        self.fills_changed.emit(page_index)
        self.dialogues_changed.emit(page_index)  # -> redraw nền vá vùng mới
        return fid

    def remove_fill(self, page_index: int, fill_id: int) -> bool:
        page = self._page(page_index)
        if page is None or not any(f["id"] == fill_id for f in page.fills):
            return False
        page.fills = [f for f in page.fills if f["id"] != fill_id]
        self._set_dirty(True)
        self.fills_changed.emit(page_index)
        self.dialogues_changed.emit(page_index)
        return True

    def set_fill_hidden(self, page_index: int, fill_id: int, hidden: bool) -> None:
        """Ẩn / hiện 1 vùng fill (chỉ đổi lớp hiển thị, không tính lại). Lớp của vùng fill có id = -fill_id."""
        page = self._page(page_index)
        if page is None:
            return
        for f in page.fills:
            if f["id"] == fill_id and bool(f.get("hidden")) != hidden:
                f["hidden"] = hidden
                self._set_dirty(True)
                self.fills_changed.emit(page_index)
                self.visibility_changed.emit(page_index, -fill_id)
                return

    def set_layout_info(self, page_index: int, info: dict) -> None:
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].layout_info = {int(k): dict(v) for k, v in (info or {}).items()}

    def set_layers_meta(self, page_index: int, layers_path: str) -> None:
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].layers_path = layers_path or None

    # -- Ẩn / hiện lẻ chữ & vùng redraw ----------------------------------------------------
    def set_dialogue_visibility(self, page_index: int, dialogue_id: int, text: Optional[bool] = None,
                                redraw: Optional[bool] = None) -> None:
        """text / redraw: True = hiện, False = ẩn, None = giữ nguyên. Không chạy lại redraw (chỉ đổi lớp hiển thị)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                changed = False
                if text is not None and d.hide_text == text:
                    d.hide_text = not text
                    changed = True
                if redraw is not None and d.hide_redraw == redraw:
                    d.hide_redraw = not redraw
                    changed = True
                if changed:
                    self._set_dirty(True)
                    self.visibility_changed.emit(page_index, dialogue_id)
                return

    def set_all_visibility(self, page_index: int, text: Optional[bool] = None, redraw: Optional[bool] = None) -> None:
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        changed = False
        for d in page.dialogues:
            if text is not None and d.hide_text == text:
                d.hide_text = not text
                changed = True
            if redraw is not None and d.hide_redraw == redraw:
                d.hide_redraw = not redraw
                changed = True
        if changed:
            self._set_dirty(True)
            self.visibility_changed.emit(page_index, 0)

    def add_redraw_only_regions(self, page_index: int, dialogues) -> int:
        """Thêm HÀNG LOẠT vùng 'chỉ redraw' (Dialogue redraw_only=True, chưa có id) vào cuối danh sách thoại của trang; phát tín hiệu MỘT lần.
        Trả về số vùng đã thêm."""
        if not (0 <= page_index < len(self.project.pages)) or not dialogues:
            return 0
        page = self.project.pages[page_index]
        next_id = max((d.dialogue_id for d in page.dialogues), default=0) + 1
        for d in dialogues:
            d.dialogue_id = next_id
            next_id += 1
            page.dialogues.append(d)
        self._set_dirty(True)
        self.page_updated.emit(page_index)
        self.dialogues_changed.emit(page_index)
        return len(dialogues)

    def set_dialogues_visibility(self, page_index: int, dialogue_ids, text: Optional[bool] = None,
                                 redraw: Optional[bool] = None) -> None:
        """Như set_all_visibility nhưng chỉ cho các câu có dialogue_id trong `dialogue_ids` (các câu đang chọn)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        wanted = {int(i) for i in dialogue_ids}
        changed = False
        for d in page.dialogues:
            if d.dialogue_id not in wanted:
                continue
            if text is not None and d.hide_text == text:
                d.hide_text = not text
                changed = True
            if redraw is not None and d.hide_redraw == redraw:
                d.hide_redraw = not redraw
                changed = True
        if changed:
            self._set_dirty(True)
            self.visibility_changed.emit(page_index, 0)

    def set_layout_boxes(self, page_index: int, layout: dict) -> None:
        if 0 <= page_index < len(self.project.pages):
            self.project.pages[page_index].layout_boxes = {int(k): list(v) for k, v in (layout or {}).items()}

    # -- Khung chữ (kéo thả kiểu Canva trong chế độ xem bản dịch) ---------------------------
    def update_text_frame(self, page_index: int, dialogue_id: int, frame: List[int]) -> None:
        """Đặt khung chữ [x, y, w, h]. Chỉ xếp lại chữ trên ảnh đã xoá (không xoá lại chữ gốc, không nạp lại trang)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page or len(frame) != 4:
            return
        frame = [int(round(v)) for v in frame]
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                if d.text_frame == frame:
                    return
                d.text_frame = frame
                self._set_dirty(True)
                self.dialogues_changed.emit(page_index)  # -> redraw nền (chỉ typeset)
                return

    def update_text_rotation(self, page_index: int, dialogue_id: int, angle: float) -> None:
        """Xoay khối chữ (độ, dương = theo chiều kim đồng hồ). Chỉ xếp lại chữ, không xoá lại chữ gốc."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return
        angle = round(((float(angle) + 180.0) % 360.0) - 180.0, 1)  # về khoảng [-180, 180)
        if abs(angle) < 0.2:
            angle = 0.0
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                if abs(d.text_rotation - angle) < 0.05:
                    return
                d.text_rotation = angle
                self._set_dirty(True)
                self.dialogues_changed.emit(page_index)
                return

    def redraw_dialogue_region(self, page_index: int, dialogue_id: int) -> bool:
        """Redraw lại vùng của 1 câu theo KHUNG HIỆN TẠI: bỏ khung neo cũ và buộc engine xoá lại chữ gốc từ đầu
        (trả lại điểm ảnh gốc ở vị trí cũ, xoá ở vị trí mới)."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return False
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id:
                d.redraw_box = None
                d.redraw_gen += 1
                self._set_dirty(True)
                self.dialogues_changed.emit(page_index)
                return True
        return False

    def reset_text_frame(self, page_index: int, dialogue_id: int) -> bool:
        """Bỏ khung chữ và góc xoay do người dùng đặt: chữ trở lại xếp tự động theo hình bóng thoại."""
        page = self.project.pages[page_index] if 0 <= page_index < len(self.project.pages) else None
        if not page:
            return False
        for d in page.dialogues:
            if d.dialogue_id == dialogue_id and (d.text_frame or d.text_rotation):
                d.text_frame = None
                d.text_rotation = 0.0
                self._set_dirty(True)
                self.dialogues_changed.emit(page_index)
                return True
        return False

    def mark_redraw(self, page_index: int, status: str, path: Optional[str] = None, message: str = "") -> None:
        """Cập nhật trạng thái redraw của trang (không đánh dấu dirty: ảnh redraw nằm trong cache, không nằm trong JSON)."""
        if 0 <= page_index < len(self.project.pages):
            page = self.project.pages[page_index]
            page.redraw_status = status
            page.redraw_message = message
            if path is not None:
                page.redraw_path = path
            self.redraw_updated.emit(page_index)

    # -- Dirty flag (đã có thay đổi chưa lưu) ----------------------------------
    def _set_dirty(self, value: bool) -> None:
        if value != self._dirty:
            self._dirty = value
            self.dirty_changed.emit(value)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_saved(self) -> None:
        self._set_dirty(False)