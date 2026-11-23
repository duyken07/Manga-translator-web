"""
Hàng đợi xử lý dịch ảnh với Concurrency Limit, tránh vượt Rate Limit (RPM)
của Gemini API. Dùng QThreadPool có sẵn của Qt - không cần thêm dependency.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.qt_compat import QObject, QRunnable, QThreadPool, Signal, Slot

from app.models import Page
from app.services.translation_service import (
    TranslationService,
    TranslationServiceError,
    TranslationRateLimitError,
)

_WORKER_STACK_BYTES = 32 * 1024 * 1024  # stack luồng nền (xem TranslationQueueManager.__init__)
log = logging.getLogger("queue_manager")

# Số lần thử lại tối đa khi gặp lỗi Rate Limit (429), với backoff tăng dần.
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 5


class _PageTranslateSignals(QObject):
    started = Signal(int)  # page_index: luồng nền BẮT ĐẦU dịch trang này (khác với lúc xếp hàng)
    finished = Signal(int, object)  # page_index, Page
    error = Signal(int, str)  # page_index, error message


class _PageTranslateTask(QRunnable):
    def __init__(
        self,
        page_index: int,
        image_path: str,
        source_lang: str,
        target_lang: str,
        service: TranslationService,
        context: Optional[dict] = None,
    ):
        super().__init__()
        self.page_index = page_index
        self.image_path = image_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.service = service
        self.context = context
        self.signals = _PageTranslateSignals()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit(self.page_index)
        attempt = 0
        while True:
            try:
                page = self.service.translate_page(
                    self.image_path, self.source_lang, self.target_lang, context=self.context
                )
                self.signals.finished.emit(self.page_index, page)
                return
            except TranslationRateLimitError as exc:
                attempt += 1
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    log.error("Trang idx=%s: vượt Rate Limit sau %d lần thử: %s", self.page_index, attempt - 1, exc)
                    self.signals.error.emit(self.page_index, f"Vượt Rate Limit, đã thử lại {attempt - 1} lần: {exc}")
                    return
                log.warning(
                    "Trang idx=%s: gặp Rate Limit, thử lại lần %d/%d sau %ds...",
                    self.page_index, attempt, MAX_RATE_LIMIT_RETRIES, RATE_LIMIT_BACKOFF_SECONDS * attempt,
                )
                # Backoff tăng dần: 5s, 10s, 15s... để tránh dội thêm request vào lúc đang bị giới hạn
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS * attempt)
            except TranslationServiceError as exc:
                log.error("Trang idx=%s lỗi dịch thuật: %s", self.page_index, exc)
                self.signals.error.emit(self.page_index, str(exc))
                return
            except Exception as exc:  # bảo vệ: không để 1 lỗi lạ làm crash worker pool
                log.exception("Trang idx=%s gặp lỗi không xác định", self.page_index)
                self.signals.error.emit(self.page_index, f"Lỗi không xác định: {exc}")
                return


class TranslationQueueManager(QObject):
    """
    Điều phối dịch song song nhiều trang, giới hạn số request đồng thời.

    Signals:
        progress(done, total)        - cập nhật thanh tiến trình
        page_done(page_index, Page)  - 1 trang dịch xong thành công
        page_failed(page_index, msg) - 1 trang dịch lỗi
        page_started(page_index)     - 1 trang vừa bắt đầu được dịch
        all_finished()               - toàn bộ hàng đợi đã xử lý xong
    """

    progress = Signal(int, int)
    page_done = Signal(int, object)
    page_failed = Signal(int, str)
    page_started = Signal(int)   # page_index: trang vừa BẮT ĐẦU được dịch (để báo "đang dịch trang nào")
    all_finished = Signal()

    def __init__(self, service: TranslationService, concurrency_limit: int = 4, parent=None):
        super().__init__(parent)
        self.service = service
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(1, concurrency_limit))
        # Mỗi trang chạy YOLO + Manga-OCR (torch) trong luồng của pool. Luồng QThreadPool mặc định chỉ có ~1 MB stack trên
        # Windows; code native sâu của torch / OpenCV dễ tràn stack -> tiến trình chết đột ngột, không có traceback.
        self.pool.setStackSize(_WORKER_STACK_BYTES)
        self._total = 0
        self._done = 0
        self._cancelled = False
        self._running: set = set()   # các trang đang được dịch thật sự (đã bắt đầu, chưa xong)
        self._live: dict = {}  # page_index -> task đang chạy: giữ tham chiếu Python (kèm signals) tới khi có kết quả

    def cancel(self) -> None:
        """Huỷ hàng đợi: bỏ các trang chưa chạy; kết quả trang đang dở (không dừng giữa chừng được) bị bỏ qua
        không còn đổ vào project hiện tại. Dùng khi bắt đầu dịch project khác / đóng app."""
        self._cancelled = True
        self._seq_remaining = []
        self.pool.clear()
        log.info("Đã huỷ hàng đợi dịch (%d trang đang chạy dở sẽ bị bỏ kết quả).", self.pool.activeThreadCount())

    def running_pages(self) -> list:
        """Chỉ số các trang đang được dịch (đã bắt đầu, chưa xong), theo thứ tự."""
        return sorted(self._running)

    def _on_task_started(self, page_index: int) -> None:
        if self._cancelled:
            return
        self._running.add(page_index)
        self.page_started.emit(page_index)

    def is_busy(self) -> bool:
        return self.pool.activeThreadCount() > 0

    def set_concurrency_limit(self, limit: int) -> None:
        self.pool.setMaxThreadCount(max(1, limit))

    def enqueue_pages(
        self,
        pages: List[Page],
        source_lang: str,
        target_lang: str,
        indices: List[int],
        base_context: Optional[dict] = None,
    ) -> None:
        """
        pages: danh sách Page tương ứng 1-1 với indices (vị trí trong Project.pages)
        indices: chỉ số thật trong Project.pages, dùng để map kết quả trả về đúng vị trí
        base_context: ngữ cảnh TĨNH áp dụng CHO MỌI trang như nhau (VD: bộ
        thuật ngữ tùy chỉnh người dùng tự soạn, tùy chọn giữ nguyên hậu tố) -
        khác với glossary tích luỹ dần theo thứ tự của enqueue_pages_sequential(),
        base_context ở đây giống nhau cho tất cả các trang vì chạy song song
        không đảm bảo thứ tự hoàn thành.
        """
        self._total = len(pages)
        self._done = 0
        log.info("Bắt đầu dịch %d trang (concurrency=%d)...", self._total, self.pool.maxThreadCount())
        if self._total == 0:
            self.all_finished.emit()
            return

        for page, idx in zip(pages, indices):
            task = _PageTranslateTask(idx, page.image_path, source_lang, target_lang, self.service, context=base_context)
            task.signals.started.connect(self._on_task_started)
            task.signals.finished.connect(self._on_task_finished)
            task.signals.error.connect(self._on_task_error)
            self._live[idx] = task
            self.pool.start(task)

    def _on_task_finished(self, page_index: int, page: Page) -> None:
        self._live.pop(page_index, None)
        self._running.discard(page_index)
        if self._cancelled:
            return
        self._done += 1
        self.page_done.emit(page_index, page)
        self.progress.emit(self._done, self._total)
        if self._done >= self._total:
            self.all_finished.emit()

    def _on_task_error(self, page_index: int, message: str) -> None:
        self._live.pop(page_index, None)
        self._running.discard(page_index)
        if self._cancelled:
            return
        self._done += 1
        self.page_failed.emit(page_index, message)
        self.progress.emit(self._done, self._total)
        if self._done >= self._total:
            self.all_finished.emit()

    def wait_for_done(self, timeout_ms: int = -1) -> None:
        self.pool.waitForDone(timeout_ms)

    # -- Chế độ dịch TUẦN TỰ có ngữ cảnh liên trang -----------------------------
    def enqueue_pages_sequential(
        self,
        pages: List[Page],
        source_lang: str,
        target_lang: str,
        indices: List[int],
        initial_glossary: Optional[dict] = None,
        keep_honorifics: bool = False,
    ) -> None:
        """
        Dịch TUẦN TỰ từng trang một theo đúng thứ tự (KHÔNG song song), mang
        glossary tích luỹ + vài câu thoại cuối trang trước làm ngữ cảnh cho
        trang kế tiếp (xem TranslationService.translate_page). Tăng độ nhất
        quán tên riêng/thuật ngữ xuyên suốt truyện, đổi lại chậm hơn hẳn so
        với enqueue_pages() vì chỉ có 1 request chạy tại một thời điểm.

        initial_glossary: glossary đã tích luỹ từ trước (VD: mở lại project
        đã dịch một phần), dùng làm điểm khởi đầu thay vì bắt đầu từ rỗng.
        """
        self._total = len(pages)
        self._done = 0
        self._seq_remaining = list(zip(pages, indices))
        self._seq_source_lang = source_lang
        self._seq_target_lang = target_lang
        self._seq_running_glossary = dict(initial_glossary or {})
        self._seq_recent_dialogues: list = []
        self._seq_keep_honorifics = keep_honorifics
        log.info("Bắt đầu dịch TUẦN TỰ %d trang (giữ ngữ cảnh liên trang)...", self._total)

        if self._total == 0:
            self.all_finished.emit()
            return

        self._start_next_sequential_task()

    def _start_next_sequential_task(self) -> None:
        if self._cancelled or not self._seq_remaining:
            return
        page, idx = self._seq_remaining.pop(0)
        context = {
            "glossary": dict(self._seq_running_glossary),
            "recent_dialogues": self._seq_recent_dialogues,
            "keep_honorifics": self._seq_keep_honorifics,
        }
        task = _PageTranslateTask(
            idx, page.image_path, self._seq_source_lang, self._seq_target_lang, self.service, context=context
        )
        task.signals.started.connect(self._on_task_started)
        task.signals.finished.connect(self._on_sequential_task_finished)
        task.signals.error.connect(self._on_sequential_task_error)
        self._live[idx] = task
        self.pool.start(task)

    def _on_sequential_task_finished(self, page_index: int, page: Page) -> None:
        self._live.pop(page_index, None)
        self._running.discard(page_index)
        if self._cancelled:
            return
        self._done += 1
        # Cập nhật glossary + vài câu thoại cuối để làm ngữ cảnh cho trang TIẾP THEO
        for term, translation in (page.glossary_updates or {}).items():
            self._seq_running_glossary.setdefault(term, translation)
        if page.dialogues:
            self._seq_recent_dialogues = [
                {"original_text": d.original_text, "translated_text": d.translated_text}
                for d in page.dialogues[-4:]
            ]
        self.page_done.emit(page_index, page)
        self.progress.emit(self._done, self._total)
        self._advance_sequential_queue()

    def _on_sequential_task_error(self, page_index: int, message: str) -> None:
        self._live.pop(page_index, None)
        self._running.discard(page_index)
        if self._cancelled:
            return
        self._done += 1
        self.page_failed.emit(page_index, message)
        self.progress.emit(self._done, self._total)
        self._advance_sequential_queue()

    def _advance_sequential_queue(self) -> None:
        if self._cancelled:
            return
        if self._seq_remaining:
            self._start_next_sequential_task()
        elif self._done >= self._total:
            self.all_finished.emit()


# ---------------------------------------------------------------------------
# Dịch 1 vùng ảnh do người dùng tự khoanh (crop) trên Workspace.
# Đây là tác vụ đơn lẻ (không phải hàng loạt như trên) nên dùng 1 QRunnable
# nhẹ chạy trên QThreadPool toàn cục của Qt, không cần Concurrency Limit riêng.
# ---------------------------------------------------------------------------
class _RegionTranslateSignals(QObject):
    finished = Signal(object)  # Dialogue đã dịch xong
    error = Signal(str)


class _RegionTranslateTask(QRunnable):
    def __init__(
        self,
        image_path: str,
        crop_box: List[int],
        source_lang: str,
        target_lang: str,
        service: TranslationService,
    ):
        super().__init__()
        self.image_path = image_path
        self.crop_box = crop_box
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.service = service
        self.signals = _RegionTranslateSignals()

    @Slot()
    def run(self) -> None:
        try:
            dialogue = self.service.translate_region(
                self.image_path, self.crop_box, self.source_lang, self.target_lang
            )
            self.signals.finished.emit(dialogue)
        except TranslationServiceError as exc:
            log.error("Lỗi dịch vùng ảnh: %s", exc)
            self.signals.error.emit(str(exc))
        except Exception as exc:  # bảo vệ: không để 1 lỗi lạ làm crash worker pool
            log.exception("Lỗi không xác định khi dịch vùng ảnh")
            self.signals.error.emit(f"Lỗi không xác định: {exc}")


_async_pool_instance: Optional[QThreadPool] = None
_ACTIVE_ASYNC: set = set()  # task nền đang chạy: người gọi có thể không giữ lại, nên giữ ở đây tới khi xong


def _async_pool() -> QThreadPool:
    """Pool riêng cho tác vụ lẻ (dịch vùng / tra ngữ pháp) với stack lớn; pool toàn cục của Qt chỉ có stack mặc định."""
    global _async_pool_instance
    if _async_pool_instance is None:
        _async_pool_instance = QThreadPool()
        _async_pool_instance.setMaxThreadCount(2)
        _async_pool_instance.setStackSize(_WORKER_STACK_BYTES)
    return _async_pool_instance


def _start_async(task) -> None:
    _ACTIVE_ASYNC.add(task)
    task.signals.finished.connect(lambda *_a, t=task: _ACTIVE_ASYNC.discard(t))
    task.signals.error.connect(lambda *_a, t=task: _ACTIVE_ASYNC.discard(t))
    _async_pool().start(task)


def translate_region_async(
    service: TranslationService,
    image_path: str,
    crop_box: List[int],
    source_lang: str,
    target_lang: str,
    on_finished,
    on_error,
) -> _RegionTranslateTask:
    """
    Chạy translate_region() trên 1 luồng nền, không chặn UI.
    on_finished(dialogue) và on_error(message) sẽ được gọi lại trên main thread
    (Qt tự động dùng Queued Connection vì tín hiệu phát ra từ luồng khác).
    """
    task = _RegionTranslateTask(image_path, crop_box, source_lang, target_lang, service)
    task.signals.finished.connect(on_finished)
    task.signals.error.connect(on_error)
    _start_async(task)
    return task


# ---------------------------------------------------------------------------
# Phân tích từ vựng + ngữ pháp 1 câu ("Tra từ điển") - chạy nền, không chặn UI.
# ---------------------------------------------------------------------------
class _GrammarAnalysisSignals(QObject):
    finished = Signal(dict)  # {"words": [...], "grammar_notes": "..."}
    error = Signal(str)


class _GrammarAnalysisTask(QRunnable):
    def __init__(self, api_key: str, model_name: str, sentence: str, source_lang: str, target_lang: str):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.sentence = sentence
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.signals = _GrammarAnalysisSignals()

    @Slot()
    def run(self) -> None:
        from app.services.grammar_analysis import analyze_sentence

        try:
            result = analyze_sentence(
                self.api_key, self.model_name, self.sentence, self.source_lang, self.target_lang
            )
            self.signals.finished.emit(result)
        except TranslationServiceError as exc:
            log.error("Lỗi phân tích ngữ pháp: %s", exc)
            self.signals.error.emit(str(exc))
        except Exception as exc:  # bảo vệ: không để 1 lỗi lạ làm crash worker pool
            log.exception("Lỗi không xác định khi phân tích ngữ pháp")
            self.signals.error.emit(f"Lỗi không xác định: {exc}")


def analyze_sentence_async(
    api_key: str,
    model_name: str,
    sentence: str,
    source_lang: str,
    target_lang: str,
    on_finished,
    on_error,
) -> _GrammarAnalysisTask:
    """Chạy analyze_sentence() trên luồng nền, không chặn UI."""
    task = _GrammarAnalysisTask(api_key, model_name, sentence, source_lang, target_lang)
    task.signals.finished.connect(on_finished)
    task.signals.error.connect(on_error)
    _start_async(task)
    return task


# ---------------------------------------------------------------------------
# Chatbot hỏi đáp ngôn ngữ (đi kèm phân tích ngữ pháp) - chạy nền, không chặn UI.
# ---------------------------------------------------------------------------
class _ChatSignals(QObject):
    finished = Signal(str)  # câu trả lời
    error = Signal(str)


class _ChatTask(QRunnable):
    def __init__(self, api_key: str, model_name: str, system_instruction: str, messages: list, use_search: bool = False,
                 temperature=None, images=()):
        super().__init__()
        self.use_search = use_search
        self.temperature = temperature
        self.images = list(images)       # [(nhãn, đường dẫn ảnh trang)]: đọc + thu nhỏ trên luồng nền, không làm đơ giao diện
        self.api_key = api_key
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.messages = list(messages)   # bản sao: luồng chính có thể sửa lịch sử trong lúc chờ
        self.signals = _ChatSignals()

    @Slot()
    def run(self) -> None:
        from app.services.chat_service import chat_reply

        try:
            self.signals.finished.emit(chat_reply(self.api_key, self.model_name, self.system_instruction, self.messages, self.use_search, self.temperature, self.images))
        except TranslationServiceError as exc:
            log.error("Lỗi chatbot: %s", exc)
            self.signals.error.emit(str(exc))
        except Exception as exc:  # bảo vệ: không để 1 lỗi lạ làm crash worker pool
            log.exception("Lỗi không xác định khi chat")
            self.signals.error.emit(f"Lỗi không xác định: {exc}")


def chat_reply_async(api_key: str, model_name: str, system_instruction: str, messages: list, on_finished, on_error,
                     use_search: bool = False, temperature=None, images=()) -> _ChatTask:
    """Chạy chat_reply() trên luồng nền. Người gọi phải giữ tham chiếu task cho tới khi xong."""
    task = _ChatTask(api_key, model_name, system_instruction, messages, use_search, temperature, images)
    task.signals.finished.connect(on_finished)
    task.signals.error.connect(on_error)
    _start_async(task)
    return task
