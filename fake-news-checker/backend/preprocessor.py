import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from keybert import KeyBERT
    from transformers import AutoModel, AutoTokenizer
    from underthesea import ner, sent_tokenize

    PHOBERT_AVAILABLE = True
except ImportError:
    PHOBERT_AVAILABLE = False
    logging.warning(
        "PhoBERT/KeyBERT/Underthesea dependencies not available. "
        "Using basic preprocessing."
    )

from crawler import Crawler
from text_utils import normalize_text

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, use_phobert: bool = True):
        self.use_phobert = use_phobert and PHOBERT_AVAILABLE
        self.stopwords = self._load_stopwords()
        self.crawler = Crawler()

        if self.use_phobert:
            self._init_phobert()

        logger.info(f"Preprocessor initialized (PhoBERT: {self.use_phobert})")

    def _init_phobert(self):
        try:
            logger.info("Loading PhoBERT models...")
            model_name = "vinai/phobert-base"

            self.phobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.phobert_model = AutoModel.from_pretrained(model_name)
            self.phobert_model.eval()

            # KeyBERT sẽ sử dụng mô hình PhoBERT làm nền
            self.kw_model = KeyBERT(model=self.phobert_model)

            logger.info(f"PhoBERT loaded successfully! (Model: {model_name})")

        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}", exc_info=True)
            self.use_phobert = False

    def _load_stopwords(self) -> set:
        return set(
            [
                "và",
                "hoặc",
                "của",
                "có",
                "được",
                "đã",
                "đang",
                "sẽ",
                "này",
                "đó",
                "kia",
                "các",
                "những",
                "cho",
                "từ",
                "với",
                "trong",
                "ngoài",
                "trên",
                "dưới",
                "là",
                "thì",
                "mà",
                "một",
                "hai",
                "ba",
                "bốn",
                "năm",
                "sáu",
                "bảy",
                "tám",
                "cũng",
                "để",
                "vào",
                "ra",
                "đến",
                "bị",
                "bởi",
                "còn",
                "khi",
                "lại",
                "sau",
                "trước",
                "nếu",
                "không",
                "chỉ",
                "như",
                "theo",
                "đều",
                "rất",
                "hay",
                "về",
                "tại",
                "do",
                "đây",
                "đấy",
                "ấy",
                "nào",
                "gì",
                "ai",
                "đâu",
                "bao",
                "lúc",
                "nơi",
                "người",
                "việc",
                "chúng",
                "nhiều",
            ]
        )

    def simple_tokenize(self, text: str) -> List[str]:
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def extract_title_from_text(self, text: str) -> str:
        if self.use_phobert:
            try:
                sentences = sent_tokenize(text)
                if not sentences:
                    return text.split(".")[0].strip()
                scores = []
                for i, sent in enumerate(sentences[:5]):
                    score = (5 - i) * 2
                    word_count = len(sent.split())
                    if 5 <= word_count <= 20:
                        score += 3
                    elif 3 <= word_count <= 25:
                        score += 1
                    entities = ner(sent)
                    score += len(entities)
                    scores.append((sent, score))
                best_sentence = max(scores, key=lambda x: x[1])[0]
                return best_sentence.strip()
            except Exception as e:
                logger.warning(f"PhoBERT title extraction failed: {e}")
        return text.split(".")[0].strip()

    def extract_keywords_phobert(self, text: str, top_n: int = 15) -> List[str]:
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words=list(self.stopwords),
                top_n=top_n,
                use_mmr=True,  # Sử dụng MMR để đa dạng hóa kết quả
                diversity=0.5,
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.error(f"KeyBERT extraction failed: {e}")
            return []

    def extract_keywords_basic(self, text: str, top_n: int = 15) -> List[str]:
        normalized = normalize_text(text)
        tokens = self.simple_tokenize(normalized)
        filtered_tokens = []
        for token in tokens:
            if token in ".,!?;:()-":
                continue
            if token.lower() in self.stopwords:
                continue
            if len(token) < 3 or token.isdigit():
                continue
            if not any(c.isalnum() for c in token):
                continue
            filtered_tokens.append(token.lower())
        word_freq = Counter(filtered_tokens)
        keywords = [word for word, freq in word_freq.most_common(top_n)]
        return keywords

    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        if self.use_phobert:
            phobert_kws = self.extract_keywords_phobert(text, top_n)
            if phobert_kws:
                logger.info(
                    f"Extracted {len(phobert_kws)} keywords via PhoBERT/KeyBERT"
                )
                return phobert_kws
        logger.info("Using basic keyword extraction (fallback)")
        return self.extract_keywords_basic(text, top_n)

    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        if not self.use_phobert:
            return []
        try:
            entities = ner(text)
            return [(e[0], e[3]) for e in entities if e[3] != "O"]
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            return []

    def extract_numbers_from_text(self, text: str) -> List[str]:
        patterns = [
            r"\d+[\.,]\d+[\.,]\d+",
            r"\d+\s*(?:triệu|tỷ|nghìn|ngàn|tỉ)",
            r"\d+\s*%",
            r"\d{4,}",
            r"\d+[\.,]\d+",
        ]
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        unique_numbers = []
        seen = set()
        for num in numbers:
            normalized = num.lower().replace(" ", "")
            if normalized not in seen:
                seen.add(normalized)
                unique_numbers.append(num)
        return unique_numbers[:3]

    def process_input(
        self, input_data: str, input_type: str = "text"
    ) -> Optional[Dict[str, Any]]:
        if input_type == "url":
            logger.info(f"Processing URL: {input_data}")
            return self._process_url(input_data)

        # Xử lý input_type == 'text'
        logger.info(f"Processing text: {len(input_data)} characters")
        normalized = normalize_text(input_data)

        # Chỉ chạy trích xuất từ khóa (KeyBERT).
        keywords = self.extract_keywords(normalized, top_n=15)
        logger.info(f"Keywords: {keywords[:10]}")

        entities = []
        numbers = []

        return {
            "original_input": input_data,
            "input_type": "text",
            "title": "",
            "content": normalized,
            "full_text": normalized,
            "keywords": keywords,
            "entities": entities,
            "numbers": numbers,
            "domain": None,
        }

    def _process_url(self, url: str) -> Optional[Dict[str, Any]]:
        extracted = self.crawler.extract_from_url(url)
        if not extracted:
            logger.error(f"Failed to extract content from URL: {url}")
            return None

        # Tối ưu: Chỉ trích xuất từ khóa từ Tiêu đề + Mô tả + 500 ký tự đầu
        text_for_keywords = (
            f"{extracted['title']} "
            f"{extracted['description']} "
            f"{extracted['content'][:500]}"
        )
        # full_text vẫn chứa toàn bộ nội dung để so sánh ở Bước 5
        full_text = (
            f"{extracted['title']} {extracted['description']} {extracted['content']}"
        )
        logger.info(f"Extracted {len(full_text)} characters from URL")

        # Tối ưu logic: Chỉ chạy trích xuất từ khóa (KeyBERT).
        keywords = self.extract_keywords(text_for_keywords, top_n=15)
        logger.info(f"Keywords: {keywords[:10]}")

        entities = []
        numbers = []

        return {
            "original_input": url,
            "input_type": "url",
            "title": extracted["title"],
            "content": extracted["content"],
            "full_text": full_text,
            "keywords": keywords,
            "entities": entities,
            "numbers": numbers,
            "domain": extracted["domain"],
        }


if __name__ == "__main__":
    pass
