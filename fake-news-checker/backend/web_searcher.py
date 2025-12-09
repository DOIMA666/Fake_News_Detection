import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SmartCache:
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, tuple] = {}
        self.ttl = timedelta(hours=ttl_hours)

    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        key = self._get_key(query)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"Cache HIT: {query[:50]}")
                return data
            else:
                del self.cache[key]
        return None

    def set(self, query: str, data: List[Dict[str, Any]]):
        key = self._get_key(query)
        self.cache[key] = (data, datetime.now())


class EnhancedWebSearcher:

    def __init__(
        self,
        google_api_keys: List[str] = None,  
        google_cse_id: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self.api_keys = google_api_keys if google_api_keys else []
        self.google_cse_id = google_cse_id
        self.current_key_index = 0  

        self.cache = SmartCache(ttl_hours=24) if cache_enabled else None

        if self.api_keys and self.google_cse_id:
            logger.info(f"✓ Google API configured with {len(self.api_keys)} keys")
        else:
            logger.error("✗ Google API NOT configured")

    def _get_current_key(self) -> Optional[str]:
        """Lấy key hiện tại đang active"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _rotate_key(self):
        """Chuyển sang key tiếp theo trong danh sách"""
        if not self.api_keys:
            return
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.warning(
            f"🔄 Rotating Google Key: #{old_index} -> #{self.current_key_index}"
        )

    def build_smart_queries(self, queries_from_ai: List[str]) -> List[str]:
        """
        Nhận danh sách truy vấn từ AI Preprocessor.
        """
        if not queries_from_ai:
            return []

        unique_queries = list(dict.fromkeys(queries_from_ai))
        return unique_queries[:3] 

    def search_google_custom_api(
        self, query: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hàm gọi API với cơ chế Retry & Rotation
        """
        if not self.api_keys or not self.google_cse_id:
            return []

        url = "https://www.googleapis.com/customsearch/v1"

        max_retries = len(self.api_keys)

        for attempt in range(max_retries):
            current_key = self._get_current_key()

            params = {
                "key": current_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": min(10, num_results),
                "lr": "lang_vi",
                "gl": "vn",
            }

            try:
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    results = []
                    if "items" in data:
                        for item in data["items"]:
                            link = item.get("link", "")
                            domain = urlparse(link).netloc.replace("www.", "")

                            snippet = item.get("snippet", "")
                            meta_desc = ""

                            if "pagemap" in item and "metatags" in item["pagemap"]:
                                tags = item["pagemap"]["metatags"][0]
                                meta_desc = (
                                    tags.get("og:description")
                                    or tags.get("twitter:description")
                                    or tags.get("description")
                                    or ""
                                )

                            final_content = snippet
                            if meta_desc:
                                if len(snippet) < 50 and len(meta_desc) > 50:
                                    final_content = meta_desc
                                elif len(meta_desc) > len(snippet) + 20:
                                    final_content = meta_desc

                            final_content = final_content.replace("\n", " ").strip()

                            results.append(
                                {
                                    "url": link,
                                    "title": item.get("title", ""),
                                    "snippet": final_content,
                                    "source": "Google Search",
                                    "domain": domain,
                                }
                            )
                    return results

                # Nếu gặp lỗi Quota (429) hoặc Quyền (403 - do hết quota billing)
                elif response.status_code in [403, 429]:
                    logger.warning(
                        f"⚠️ Key #{self.current_key_index} exhausted/error ({response.status_code})..."
                    )
                    self._rotate_key()  # Đổi key
                    continue  # Thử lại vòng lặp với key mới

                else:
                    # Các lỗi khác (400 Bad Request, 500 Server Error) thì không đổi key
                    logger.error(
                        f"Google API Error {response.status_code}: {response.text}"
                    )
                    break

            except Exception as e:
                logger.error(f"Connection error: {e}")
                break

        logger.error(f"❌ All keys failed for query: {query}")
        return []

    def search_for_fact_check(
        self, processed_data: Dict, num_results: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Main search method - SỬ DỤNG PARALLEL PROCESSING
        """
        if not self.api_keys or not self.google_cse_id:
            logger.error("Search aborted - Google API not configured")
            return []

        # Lấy queries từ AI Preprocessor
        ai_queries = processed_data.get("keywords", [])
        original_url = processed_data.get("original_input", "")
        is_url_input = original_url.startswith("http")

        queries = self.build_smart_queries(ai_queries)

        if not queries:
            # Fallback nếu AI lỗi: dùng text gốc
            queries = [processed_data.get("content", "")[:100]]

        logger.info("\n" + "=" * 70)
        logger.info(f"PARALLEL SEARCHING ({len(queries)} queries)")
        logger.info("=" * 70)

        all_results = []

        # --- BẮT ĐẦU XỬ LÝ SONG SONG ---
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for query in queries:
                if self.cache:
                    cached = self.cache.get(query)
                    if cached:
                        logger.info(f"✓ Using cached results for: {query}")
                        all_results.extend(cached)
                        continue

                logger.info(f"  → Submitting query: {query}")
                future = executor.submit(self.search_google_custom_api, query, 10)
                futures[future] = query

            for future in as_completed(futures):
                query = futures[future]
                try:
                    data = future.result()
                    if data:
                        logger.info(f"    ✓ Query returned {len(data)} results")
                        if self.cache:
                            self.cache.set(query, data)
                        all_results.extend(data)
                    else:
                        logger.warning(f"    ⚠ Query returned no results")
                except Exception as exc:
                    logger.error(f"    ✗ Query generated exception: {exc}")
        # --- KẾT THÚC XỬ LÝ SONG SONG ---

        unique_results = {}
        for result in all_results:
            url = result["url"]
            if is_url_input and url == original_url:
                continue
            if url not in unique_results:
                unique_results[url] = result

        final_results = list(unique_results.values())[:num_results]
        logger.info(f"✓ Final: {len(final_results)} unique results")

        return final_results


WebSearcher = EnhancedWebSearcher
