import logging
import requests
import json
import re
import hashlib
import os
import time
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class Stance(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    DISCUSS = "discuss"
    UNRELATED = "unrelated"


# --- QUẢN LÝ KEY GROQ ---
class GroqKeyManager:
    def __init__(self, api_keys_str: Optional[str]):
        self.keys = []
        if api_keys_str:
            raw_keys = api_keys_str.replace("\n", ",").split(",")
            self.keys = [k.strip() for k in raw_keys if k.strip().startswith("gsk_")]

        self.current_index = 0

    def get_current_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def rotate_key(self):
        """Chuyển sang key tiếp theo"""
        if not self.keys:
            return
        prev_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.info(f"Rotated Key: #{prev_index + 1} -> #{self.current_index + 1}")

    def get_key_count(self):
        return len(self.keys)


class StanceAnalyzer:
    """
    Phân tích stance sử dụng Groq Cloud.
    Fix: Xử lý lỗi 429 thông minh hơn với nhiều Key.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,  
        groq_model: str = "llama-3.3-70b-versatile",
        **kwargs,
    ):
        if not groq_api_key:
            groq_api_key = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY")

        self.key_manager = GroqKeyManager(groq_api_key)
        self.groq_model = groq_model
        self._stance_cache = {}

        if not self.key_manager.keys:
            logger.error(
                "Critical Error: No valid GROQ_API_KEY found starting with 'gsk_'"
            )
            raise ValueError("Missing GROQ_API_KEY.")

        logger.info(
            f"StanceAnalyzer ready: {groq_model} | {self.key_manager.get_key_count()} Keys Loaded"
        )

    def analyze_stance_batch(
        self, claim: str, articles: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:

        if not articles:
            return []

        BATCH_SIZE = 15
        results = []

        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i : i + BATCH_SIZE]
            logger.info(f"Groq batch {i//BATCH_SIZE + 1}: {len(batch)} articles")

            try:
                batch_output = self._call_llm_batch_json(claim, batch, start_index=i)
                results.extend(batch_output)
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                for idx, art in enumerate(batch):
                    results.append(
                        {
                            "index": i + idx,
                            "stance": Stance.UNRELATED,
                            "confidence": 0.0,
                            "reasoning": f"API Error: {str(e)[:50]}",
                        }
                    )

        final_output = []
        for idx, art in enumerate(articles):
            item = next((x for x in results if x.get("index") == idx), None)
            if not item:
                final_output.append(
                    {
                        "article": art,
                        "stance": Stance.UNRELATED,
                        "confidence": 0.0,
                        "reasoning": "Missing API Output",
                    }
                )
            else:
                final_output.append(
                    {
                        "article": art,
                        "stance": item["stance"],
                        "confidence": item["confidence"],
                        "reasoning": item.get("reasoning", "LLM Analysis"),
                    }
                )

        return final_output

    def _call_llm_batch_json(self, claim: str, articles: List[Dict], start_index: int):
        def cache_key(art):
            base = f"{claim}|{art.get('title','')[:80]}"
            return hashlib.md5(base.encode()).hexdigest()

        parsed_cache = {}
        need_inference = []

        for idx, art in enumerate(articles):
            ck = cache_key(art)
            if ck in self._stance_cache:
                cached = self._stance_cache[ck].copy()
                cached["index"] = start_index + idx
                parsed_cache[start_index + idx] = cached
            else:
                need_inference.append((idx, art))

        if not need_inference:
            return list(parsed_cache.values())

        articles_text_block = ""
        for local_idx, art in need_inference:
            real_idx = start_index + local_idx
            content = (art.get("content") or art.get("snippet") or "")[:400].replace(
                "\n", " "
            )
            articles_text_block += f"--- ARTICLE ID {real_idx} ---\nTITLE: {art.get('title','')}\nCONTENT: {content}\n\n"

        prompt = f"""
Bạn là hệ thống FACT-CHECKING chuyên nghiệp.
Nhiệm vụ: Xác định quan hệ giữa CLAIM và từng ARTICLE.

REFUTE (BÁC BỎ) - Chỉ chọn khi có BẰT KỲ DẤU HIỆU NÀO sau:
✓ "bác bỏ", "phủ nhận", "không đúng", "sai sự thật"
✓ "tin giả", "tin đồn", "thất thiệt", "bịa đặt"  
✓ "chưa có bằng chứng", "không xác nhận được"
✓ "cơ quan chức năng đã làm rõ"
✓ Các từ ngữ phủ định + sự kiện claim

SUPPORT (XÁC NHẬN) - Chỉ khi:
✓ Có bằng chứng cụ thể, số liệu, tên người
✓ "đã xác nhận", "chính thức", "thông báo"

DISCUSS - Khi:
✓ Chỉ đề cập mà không kết luận
✓ "đang điều tra", "chưa rõ"

UNRELATED - Khi:
✓ Không nhắc đến claim

CLAIM: "{claim[:500]}"

DANH SÁCH BÀI BÁO:
{articles_text_block}

YÊU CẦU:
Trả về JSON ARRAY duy nhất:
[ {{ "id": <id>, "stance": "support/refute/discuss/unrelated", "confidence": 0.0-1.0 }} ]
KHÔNG giải thích thêm.
"""

        raw = self._call_groq_api_robust(prompt)

        try:
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw)
            if isinstance(data, dict):
                data = [data]

            results = []
            for item in data:
                sid = item.get("id")
                stance_str = str(item.get("stance", "unrelated")).lower().strip()
                confidence = float(item.get("confidence", 0.5))

                if "support" in stance_str:
                    st = Stance.SUPPORT
                elif "refute" in stance_str:
                    st = Stance.REFUTE
                elif "discuss" in stance_str:
                    st = Stance.DISCUSS
                else:
                    st = Stance.UNRELATED

                res = {
                    "index": sid,
                    "stance": st,
                    "confidence": confidence,
                    "reasoning": "Groq Analysis",
                }

                for l_idx, art in need_inference:
                    if start_index + l_idx == sid:
                        self._stance_cache[cache_key(art)] = res
                        break
                results.append(res)

            return results + list(parsed_cache.values())

        except Exception as e:
            logger.error(f"JSON Parse Error: {e}\nRAW: {raw[:200]}")
            raise

    def _call_groq_api_robust(self, prompt: str) -> str:
        """
        Hàm gọi API với cơ chế chờ đợi thông minh (Smart Backoff)
        """
        url = "https://api.groq.com/openai/v1/chat/completions"

        system_msg = "You are a strict JSON response bot. Output ONLY valid JSON array."

        max_retries = 10  

        consecutive_429 = 0

        for attempt in range(max_retries):
            current_key = self.key_manager.get_current_key()

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 2048,
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=45)

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]

                elif response.status_code == 429:
                    consecutive_429 += 1
                    # === LOGIC XỬ LÝ 429 ===
                    self.key_manager.rotate_key()
                    num_keys = self.key_manager.get_key_count()

                    # Nếu có > 1 key, xoay vòng và thử lại ngay
                    if num_keys > 1:
                        # Nếu bị 429 liên tiếp nhiều lần (tức là nhiều key đều bị), tăng thời gian chờ
                        wait_time = 5 * consecutive_429
                        logger.warning(
                            f"Key #{self.key_manager.current_index} Rate Limit. Rotated. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        # Nếu chỉ có 1 key, bắt buộc chờ lâu
                        wait_time = 20 + (attempt * 10)
                        logger.warning(
                            f"Single Key Rate Limit. Sleeping {wait_time}s..."
                        )
                        time.sleep(wait_time)

                    continue  

                else:
                    consecutive_429 = 0  # Reset đếm lỗi 429 nếu gặp lỗi khác
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    response.raise_for_status()

            except Exception as e:
                # Reset đếm lỗi 429 nếu gặp lỗi mạng
                consecutive_429 = 0
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(3)  # Lỗi mạng thì chờ 3s

        raise Exception("Groq API failed after multiple retries.")

    # --- Display Utils ---
    def get_stance_label_vi(self, stance: Any) -> str:
        val = str(stance.value if hasattr(stance, "value") else stance).lower()
        return {
            "support": "Xác nhận (Đúng)",
            "refute": "Bác bỏ (Sai)",
            "discuss": "Trung lập / Bàn luận",
            "unrelated": "Không liên quan",
        }.get(val, "Lỗi")

    def get_stance_emoji(self, stance: Any) -> str:
        val = str(stance.value if hasattr(stance, "value") else stance).lower()
        return {
            "support": "✅",
            "refute": "❌",
            "discuss": "💬",
            "unrelated": "➖",
        }.get(val, "❓")
