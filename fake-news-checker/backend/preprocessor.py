import logging
import os
import re
import json
from typing import Any, Dict, List, Optional


# Import Groq Client
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from crawler import Crawler
from input_validator import InputValidator

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Sử dụng Groq Cloud (Llama 3.3 70B) để xử lý văn bản:
    1. Sinh truy vấn tìm kiếm (Search Queries).
    2. Phân loại chủ đề (Topic Classification).
    """

    CATEGORIES = {
        "politics": "Chính trị",
        "crime": "Pháp luật - Tội phạm",
        "health": "Sức khỏe - Y tế",
        "entertainment": "Giải trí - Showbiz",
        "sports": "Thể thao",
        "economy": "Kinh tế - Tài chính",
        "technology": "Công nghệ",
        "education": "Giáo dục",
        "society": "Xã hội",
        "international": "Quốc tế",
        "other": "Khác",
    }

    def __init__(self):
        self.crawler = Crawler()

        # --- LOGIC CẮT CHUỖI KEY GROQ ---
        raw_key = os.getenv("GROQ_API_KEY", "")
        # Chỉ lấy key đầu tiên nếu có danh sách phân cách bởi dấu phẩy
        self.groq_api_key = raw_key.split(",")[0].strip() if raw_key else None

        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_client = None

        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info(
                    f"✅ Preprocessor: Connected to Groq Cloud ({self.groq_model})"
                )
            except Exception as e:
                logger.error(f"❌ Groq Init Failed: {e}")
        else:
            logger.warning("⚠️ No AI configured. Using basic fallback.")

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Trích xuất JSON từ phản hồi của LLM (xử lý cả trường hợp Markdown)"""
        try:
            # Tìm đoạn nằm giữa { và }
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        return None

    def analyze_input(self, text: str) -> Dict[str, Any]:
        """
        Dùng LLM để phân tích input -> Trả về JSON {category, queries}
        """
        # Giá trị mặc định nếu lỗi
        default_result = {
            "category": "other",
            "category_label": self.CATEGORIES["other"],
            "queries": [InputValidator.normalize_text(text)[:200]],
        }

        if not self.groq_client:
            return default_result

        prompt = f"""
Bạn là một Chuyên gia Kiểm chứng Thông tin (Fact-Checking Expert) và Phân tích Nội dung.
Nhiệm vụ của bạn là xử lý văn bản đầu vào để tạo ra các truy vấn tìm kiếm xác thực và phân loại chủ đề chính xác.

INPUT VĂN BẢN: "{text[:1000]}"

NHIỆM VỤ 1: TẠO 3 TRUY VẤN TÌM KIẾM (SEARCH QUERIES)
Mục tiêu: Từ thông tin người dùng nhập, hãy tạo ra 3 truy vấn tìm kiếm (Search Query) tối ưu nhất để kiểm chứng sự thật trên Google.
Kỹ thuật:
1. Phân tích ý định người dùng: Họ muốn kiểm tra tin đồn, sự kiện, hay câu nói?
2. Tối ưu từ khóa: Loại bỏ từ thừa (là, của, những...), giữ lại thực thể quan trọng (Tên người, địa danh, sự kiện).
3. Nếu input quá dài: Tóm tắt thành luận điểm cốt lõi (Main Claim).
4. Nếu input quá ngắn/mơ hồ: Tìm kiếm các từ khóa ngữ cảnh

NHIỆM VỤ 2: PHÂN LOẠI CHỦ ĐỀ (CATEGORY CLASSIFICATION)
Phân loại văn bản vào DUY NHẤT MỘT trong các danh mục sau (dựa trên nội dung chủ đạo):
- politics: (Chính trị, bầu cử, chính sách, quan chức nhà nước)
- crime: (Pháp luật, tội phạm, bắt bớ, tòa án, lừa đảo)
- health: (Y tế, dịch bệnh, thuốc, bác sĩ, sức khỏe)
- entertainment: (Giải trí, người nổi tiếng, showbiz, phim, nhạc)
- sports: (Thể thao, bóng đá, giải đấu, vận động viên)
- economy: (Kinh tế, tài chính, chứng khoán, giá vàng/đô la, doanh nghiệp)
- technology: (Công nghệ, AI, phần mềm, thiết bị số, mạng xã hội)
- education: (Giáo dục, trường học, tuyển sinh, thi cử)
- society: (Đời sống xã hội, giao thông, môi trường, thời tiết)
- international: (Thời sự quốc tế, quan hệ ngoại giao, xung đột thế giới)
- other: (Các chủ đề không thuộc danh sách trên)

YÊU CẦU ĐẦU RA (QUAN TRỌNG):
- Chỉ trả về định dạng JSON hợp lệ.
- Không sử dụng Markdown (```json).
- Không thêm bất kỳ lời dẫn hay giải thích nào.

MẪU OUTPUT:
{{
  "category": "health",
  "queries": [
    "thực hư thông tin uống nước chanh chữa ung thư",
    "bác sĩ đính chính tin đồn nước chanh nóng giết tế bào ung thư",
    "nghiên cứu khoa học về tác dụng nước chanh với ung thư"
  ]
}}
"""
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON generator."},
                    {"role": "user", "content": prompt},
                ],
                model=self.groq_model,
                temperature=0.1,
                response_format={"type": "json_object"},  # Bắt buộc JSON
                max_tokens=200,
            )

            response_text = completion.choices[0].message.content.strip()
            data = self._extract_json(response_text)

            if data:
                cat_code = data.get("category", "other")
                if cat_code not in self.CATEGORIES:
                    cat_code = "other"

                return {
                    "category": cat_code,
                    "category_label": self.CATEGORIES.get(cat_code, "Khác"),
                    "queries": data.get("queries", [])[:3],
                }

        except Exception as e:
            logger.error(f"LLM Analysis Error: {e}")

        return default_result

    def process_input(
        self, input_data: str, input_type: str = "text"
    ) -> Optional[Dict[str, Any]]:
        """
        Xử lý đầu vào: Crawl (nếu URL) -> Analyze (LLM)
        """
        content = input_data
        title = ""
        domain = None

        if input_type == "url":
            logger.info(f"Processing URL: {input_data}")
            extracted = self.crawler.extract_from_url(input_data)
            if extracted:
                content = f"{extracted['title']}. {extracted['description']}."
                title = extracted["title"]
                domain = extracted["domain"]
            else:
                return None

        # Cắt ngắn nếu quá dài để tiết kiệm token
        if len(content) > 3000:
            content = content[:3000]

        clean_content = (
            InputValidator.normalize_text(content) if input_type == "text" else content
        )

        # Gọi AI phân tích
        logger.info("Analyzing input via Groq...")
        analysis = self.analyze_input(content)

        logger.info(
            f"Category: {analysis['category_label']} | Queries: {analysis['queries']}"
        )

        return {
            "original_input": input_data,
            "input_type": input_type,
            "title": title,
            "content": content,
            "full_text": content,
            "summary_text": content,
            "keywords": analysis["queries"],  # ✅ Lấy từ dict
            "category": analysis["category"],  # ✅ Lấy từ dict
            "category_label": analysis["category_label"],
            "domain": domain,
        }
