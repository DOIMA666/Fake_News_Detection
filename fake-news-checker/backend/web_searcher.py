import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse  # <-- Đã xóa 'quote_plus' (F401)

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
                # Cache đã hết hạn, xóa nó đi
                del self.cache[key]
        return None

    def set(self, query: str, data: List[Dict[str, Any]]):
        key = self._get_key(query)
        self.cache[key] = (data, datetime.now())


class WebSearcher:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        cache_enabled: bool = True,
    ):

        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

        if self.google_api_key and self.google_cse_id:
            logger.info(
                f"Google API configured: Key={self.google_api_key[:10]}..., "
                f"CSE={self.google_cse_id}"
            )
        else:
            logger.error("=" * 70)
            logger.error("Google API NOT configured. Search will NOT work.")
            logger.error(
                "Please add GOOGLE_API_KEY and GOOGLE_CSE_ID to your .env file."
            )
            logger.error("=" * 70)

        self.trusted_sources = {
            "vnexpress.net": {},
            "tuoitre.vn": {},
            "thanhnien.vn": {},
            "dantri.com.vn": {},
            "vietnamnet.vn": {},
        }

        self.cache = SmartCache(ttl_hours=24) if cache_enabled else None

    def build_smart_queries(self, keywords: List[str]) -> List[str]:

        if not keywords:
            return []
        queries = []

        top_5_query = " ".join(keywords[:5])
        if len(top_5_query) > 10:
            queries.append(top_5_query)

        top_3_query = " ".join(keywords[:3])
        if len(top_3_query) > 10 and top_3_query not in queries:
            queries.append(top_3_query)

        top_7_query = " ".join(keywords[:7])
        if len(top_7_query) > 10 and top_7_query not in queries:
            queries.append(top_7_query)

        if not queries and keywords:
            fallback_query = " ".join(keywords)
            if len(fallback_query) > 5:
                queries.append(fallback_query)

        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:3]  # Giới hạn 3 truy vấn

    def search_google_custom_api(
        self, query: str, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google API credentials not configured")
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            site_filter = " OR ".join(
                [f"site:{d}" for d in list(self.trusted_sources.keys())]
            )
            full_query = f"{query} ({site_filter})"

            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": full_query,
                "num": min(10, num_results),
                "lr": "lang_vi",
                "gl": "vn",
                "dateRestrict": "m6",
            }

            logger.info(f"Google Custom Search API: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = []

            if "items" in data:
                for item in data["items"]:
                    link = item.get("link", "")
                    if not any(
                        domain in link for domain in self.trusted_sources.keys()
                    ):
                        continue
                    results.append(
                        {
                            "url": link,
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "Google Search",
                            "domain": urlparse(link).netloc,
                        }
                    )
            logger.info(f"  Google API: Found {len(results)} articles")
            return results
        except Exception as e:
            logger.error(f"  Google API error: {str(e)[:100]}")
            return []

    def search_for_fact_check(
        self, processed_data: Dict, num_results: int = 10
    ) -> List[Dict[str, Any]]:

        if not self.google_api_key or not self.google_cse_id:
            logger.error("Search aborted. Google API is not configured.")
            return []

        keywords = processed_data.get("keywords", [])
        if not keywords:
            logger.error("No keywords provided by preprocessor")
            return []

        queries = self.build_smart_queries(keywords)
        if not queries:
            logger.error("No valid queries generated from keywords")
            return []

        logger.info("=" * 70)
        logger.info("SEARCH STARTED (Google API Only)")
        logger.info("=" * 70)
        logger.info(f"Using keywords: {keywords[:10]}")
        logger.info(f"Generated {len(queries)} query variants:")
        for i, q in enumerate(queries, 1):
            logger.info(f"  {i}. {q}")
        logger.info("=" * 70)

        all_results = []

        for query_idx, query in enumerate(queries, 1):
            logger.info(f"\nQuery {query_idx}/{len(queries)}: '{query}'")

            query_results = []
            if self.cache:
                cached = self.cache.get(query)
                if cached:
                    all_results.extend(cached)
                    continue

            logger.info("  [Strategy 1] Google Custom Search API...")
            api_results = self.search_google_custom_api(query, num_results)
            query_results.extend(api_results)

            if self.cache and query_results:
                self.cache.set(query, query_results)

            all_results.extend(query_results)
            time.sleep(0.5)

        # De-duplicate (Loại bỏ trùng lặp)
        unique_results = {}
        for result in all_results:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

        final_results = list(unique_results.values())[:num_results]

        logger.info("\n" + "=" * 70)
        logger.info("SEARCH COMPLETED")
        logger.info(
            f"Total found: {len(all_results)} → Unique: "
            f"{len(unique_results)} → Final: {len(final_results)}"
        )
        logger.info("=" * 70)
        for i, result in enumerate(final_results, 1):
            logger.info(f"{i}. [{result['domain']}] {result['title'][:80]}")
            logger.info(f"   {result['url']}")
        logger.info("=" * 70)

        return final_results


if __name__ == "__main__":
    pass
