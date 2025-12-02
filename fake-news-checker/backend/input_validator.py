import logging
import re
import unicodedata
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InputValidator:
    """
    TRUNG TÂM XỬ LÝ INPUT (STANDARD VERSION)

    Chức năng:
    1. Validate: Kiểm tra tính hợp lệ bằng quy tắc cứng (Rules).
    2. Normalize: Chuẩn hóa văn bản.
    * Đã loại bỏ AI Check để tăng tốc độ.
    """

    def __init__(self):
        # Cấu hình ngưỡng (Thresholds)
        self.MIN_LENGTH_CHARS = 10
        self.MIN_WORDS = 3
        self.MAX_LENGTH_CHARS = 10000

        # Regex patterns
        self.gibberish_patterns = [
            r"^[a-z]{20,}$",  # Chuỗi ký tự liền không có space
            r"^[^aeiouAEIOUàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]+$",  # Không nguyên âm
        ]
        self.vietnamese_chars = (
            "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
        )

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """
        Chuẩn hóa văn bản: Chuyển về chữ thường, bỏ dấu thừa, xóa ký tự đặc biệt.
        """
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Chuyển về chữ thường
        text = text.lower()

        # 3. Giữ lại chữ cái tiếng Việt, số và các dấu câu cơ bản
        text = re.sub(
            r"[^\w\s.,!?;:\-\(\)áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]",
            " ",
            text,
        )

        # 4. Xóa khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ==================== PHẦN 2: VALIDATION (LOGIC ONLY) ====================

    def validate(
        self, content: str, input_type: str = "text"
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Kiểm tra input có hợp lệ không.
        Returns: (is_valid, error_message, suggestions)
        """
        # Bước 0: Sơ chế nhẹ để check độ dài thật
        content = content.strip()

        # 1. Kiểm tra độ dài & rỗng
        if not content:
            return False, "Nội dung không được để trống", None

        if len(content) < self.MIN_LENGTH_CHARS:
            return (
                False,
                f"Nội dung quá ngắn (tối thiểu {self.MIN_LENGTH_CHARS} ký tự)",
                {
                    "suggestion": "Hãy nhập một câu hoàn chỉnh mô tả tin tức cần kiểm tra."
                },
            )

        if len(content) > self.MAX_LENGTH_CHARS:
            return (
                False,
                "Nội dung quá dài (nghi vấn spam)",
                {"suggestion": "Vui lòng chỉ nhập phần nội dung chính cần kiểm tra."},
            )

        # 2. Kiểm tra theo loại input
        if input_type == "url":
            return self._validate_url(content)
        else:
            # Logic riêng cho text
            if len(content.split()) < self.MIN_WORDS:
                return (
                    False,
                    "Nội dung quá ngắn (ít hơn 3 từ)",
                    {"suggestion": "Vui lòng nhập đầy đủ câu."},
                )

            # 3. Kiểm tra Gibberish (Ký tự vô nghĩa)
            if self._is_gibberish(content):
                return (
                    False,
                    "Nội dung có vẻ vô nghĩa hoặc chứa ký tự lạ",
                    {"suggestion": "Vui lòng nhập tiếng Việt có dấu."},
                )

            # 4. Kiểm tra Tiếng Việt
            if not self._has_vietnamese_content(content):
                return (
                    False,
                    "Hệ thống hiện tại ưu tiên hỗ trợ Tiếng Việt",
                    {"suggestion": "Vui lòng nhập tin tức bằng tiếng Việt."},
                )

        return True, None, None

    def _validate_url(self, url: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, url):
            return (
                False,
                "URL không đúng định dạng",
                {"suggestion": "URL phải bắt đầu bằng http:// hoặc https://"},
            )

        suspicious_domains = [".tk", ".ga", ".ml", ".cf", "bit.ly"]
        if any(d in url.lower() for d in suspicious_domains):
            return False, "URL thuộc tên miền đáng ngờ", None
        return True, None, None

    def _is_gibberish(self, text: str) -> bool:
        """Kiểm tra chuỗi vô nghĩa"""
        for pattern in self.gibberish_patterns:
            if re.search(pattern, text):
                return True
        # Tỷ lệ ký tự hợp lệ thấp -> rác
        clean_len = len(
            re.findall(
                r"[a-zA-Z0-9\s" + self.vietnamese_chars + "]", text, re.IGNORECASE
            )
        )
        if len(text) > 0 and (clean_len / len(text)) < 0.7:
            return True
        return False

    def _has_vietnamese_content(self, text: str) -> bool:
        """Kiểm tra xem có ít nhất 2 ký tự tiếng Việt không"""
        viet_char_count = sum(
            1 for char in text.lower() if char in self.vietnamese_chars
        )
        return viet_char_count >= 2
