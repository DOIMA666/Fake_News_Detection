import re
import unicodedata
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(
        r"[^\w\s.,!?;:\-\(\)áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text
