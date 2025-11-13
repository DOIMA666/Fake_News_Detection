import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from config import Config
except ImportError:

    class Config:
        GOOGLE_API_KEY = None
        GOOGLE_CSE_ID = None
        NEWS_API_KEY = None
        ENABLE_CACHE = True
        DEFAULT_NUM_RESULTS = 5


try:
    from preprocessor import TextPreprocessor
    from similarity_checker import SimilarityChecker
    from web_searcher import WebSearcher
except ImportError as e:
    logging.error(f"Lỗi import quan trọng: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        news_api_key: Optional[str] = None,
    ):

        logger.info("=" * 70)
        logger.info(" Initializing Fact Checker...")
        logger.info("=" * 70)
        api_key = google_api_key or Config.GOOGLE_API_KEY
        cse_id = google_cse_id or Config.GOOGLE_CSE_ID

        self.preprocessor = TextPreprocessor()
        logger.info("Preprocessor (and Crawler) initialized")
        self.searcher = WebSearcher(
            google_api_key=api_key,
            google_cse_id=cse_id,
            cache_enabled=Config.ENABLE_CACHE,
        )
        logger.info(" Web Searcher initialized")
        self.similarity_checker = SimilarityChecker()
        logger.info("Similarity Checker initialized")
        logger.info("=" * 70)
        logger.info(" Fact Checker ready!")
        logger.info("=" * 70 + "\n")

    def check_fact(
        self,
        user_input: str,
        input_type: str = "text",
        num_sources: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ """
        if num_sources is None:
            num_sources = Config.DEFAULT_NUM_RESULTS
        results = {
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "original_input": user_input,
            "status": "processing",
        }
        try:
            # --- BƯỚC 1: TIỀN XỬ LÝ ---
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1: PREPROCESSING")
            logger.info("=" * 70)
            processed = self.preprocessor.process_input(user_input, input_type)
            if not processed:
                results["status"] = "error"
                results["error"] = "Không thể xử lý input"
                logger.error("Failed to preprocess input")
                return results

            if not processed["keywords"] and len(processed["full_text"]) < 15:
                logger.warning(
                    f"Input '{user_input}' quá ngắn hoặc không có ngữ nghĩa."
                )
                results["status"] = "input_too_short"
                results["message"] = (
                    "Nội dung quá ngắn hoặc không đủ ngữ nghĩa để phân tích. "
                    "Vui lòng cung cấp thêm chi tiết."
                )
                return results

            results["processed_data"] = {
                "title": processed["title"],
                "keywords": processed["keywords"],
                "domain": processed["domain"],
            }
            logger.info(
                f" Extracted {len(processed['keywords'])} keywords: {processed['keywords'][:10]}"
            )

            # --- BƯỚC 2: TÌM KIẾM ---
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: SEARCHING FOR REFERENCE ARTICLES")
            logger.info("=" * 70)
            reference_articles = self.searcher.search_for_fact_check(
                processed, num_sources
            )
            if not reference_articles:
                results["status"] = "no_references"
                results["message"] = "Không tìm thấy bài báo tham khảo từ nguồn uy tín"
                logger.warning("No reference articles found")
                return results
            logger.info(f"Found {len(reference_articles)} reference articles")

            logger.info(
                f"Found {len(reference_articles)} articles to crawl (NO filtering)"
            )

            # --- BƯỚC 4: THU THẬP NỘI DUNG (Song song) ---
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: CRAWLING REFERENCE ARTICLES (PARALLEL)")
            logger.info("=" * 70)
            reference_contents = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_article = {
                    executor.submit(
                        self.preprocessor._process_url, article["url"]
                    ): article
                    for article in reference_articles
                }
                for future in as_completed(future_to_article):
                    article = future_to_article[future]
                    try:
                        content = future.result()
                        if content and content["content"]:
                            reference_contents.append(
                                {
                                    "url": article["url"],
                                    "title": content["title"] or article["title"],
                                    "content": content["content"],
                                    "domain": content["domain"],
                                    "snippet": article.get("snippet", ""),
                                    "source": article.get("source", ""),
                                }
                            )
                            logger.info(
                                f"Success ({article['domain']}): {len(content['content'])} chars"
                            )
                        else:
                            logger.warning(f"Failed to crawl: {article['url']}")
                    except Exception as e:
                        logger.error(f"Error crawling {article['url']}: {e}")

            if not reference_contents:
                results["status"] = "crawl_failed"
                results["message"] = "Không thể crawl nội dung từ các bài báo tham khảo"
                logger.error("All crawl attempts failed")
                return results
            logger.info(
                f"Successfully crawled {len(reference_contents)}/{len(reference_articles)} articles"
            )

            # --- BƯỚC 5: TÍNH TOÁN TƯƠNG ĐỒNG (Batch) ---
            logger.info("\n" + "=" * 70)
            logger.info("STEP 5: CALCULATING SIMILARITY (BATCHED)")
            logger.info("=" * 70)

            reference_texts = [ref["content"] for ref in reference_contents]
            text_to_compare = processed["full_text"]
            logger.info("Using FULL TEXT for comparison (Content-vs-Content)")

            logger.info(
                f"Running batch similarity for {len(reference_texts)} articles..."
            )
            batch_results = self.similarity_checker.calculate_similarity_batch(
                text_to_compare, reference_texts
            )
            similarity_results = []
            for batch_item in batch_results:
                original_index = batch_item["index"]
                ref_metadata = reference_contents[original_index]
                overall_sim = batch_item["similarity"]
                similarity_results.append(
                    {
                        "url": ref_metadata["url"],
                        "title": ref_metadata["title"],
                        "domain": ref_metadata["domain"],
                        "source": ref_metadata.get("source", ""),
                        "overall_similarity": overall_sim,
                        "detailed_similarity": None,
                    }
                )
                logger.info(f"{ref_metadata['domain']}: {overall_sim:.2%}")

            # --- BƯỚC 6: ĐƯA RA KẾT LUẬN (Kiểm tra Phủ định & Lấy Max) ---
            logger.info("\n" + "=" * 70)
            logger.info("STEP 6: GENERATING VERDICT (REFUTATION CHECK & MAX)")
            logger.info("=" * 70)

            top_results = similarity_results[: min(3, len(similarity_results))]

            refutation_keywords = [
                "bác bỏ",
                "phủ nhận",
                "đính chính",
                "tin đồn",
                "tin giả",
                "sự thật",
                "thực hư",
                "giả mạo",
                "vu khống",
            ]
            final_scores = []

            if not top_results:
                highest_similarity = 0
                top_scores = []
            else:
                for r in top_results:
                    title_lower = r["title"].lower()
                    score = r["overall_similarity"]
                    if (
                        any(keyword in title_lower for keyword in refutation_keywords)
                        and score > 0.6
                    ):
                        logger.warning(f"Refutation detected: {r['title']}")
                        final_scores.append(1.0 - score)  # Lật ngược điểm
                    else:
                        final_scores.append(score)

                top_scores = sorted(final_scores, reverse=True)
                highest_similarity = top_scores[0] if top_scores else 0

            logger.info(f"Adjusted Top scores: {[f'{s:.2%}' for s in top_scores]}")
            logger.info(
                f"Calculated Adjusted Highest Similarity: {highest_similarity:.2%}"
            )

            verdict = self.similarity_checker.generate_verdict(highest_similarity)

            results["status"] = "success"
            results["verdict"] = verdict
            results["highest_similarity"] = highest_similarity
            results["similarity_details"] = similarity_results
            results["top_references"] = [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "domain": r["domain"],
                    "source": r.get("source", ""),
                    "similarity": r["overall_similarity"],
                }
                for r in top_results
            ]

            logger.info("\n" + "=" * 70)
            logger.info("FINAL VERDICT")
            logger.info("=" * 70)
            logger.info(f"Label: {verdict['label']}")
            logger.info(f"Verdict: {verdict['verdict']}")
            logger.info(f"Highest Similarity: {highest_similarity:.2%}")
            logger.info(f"Confidence: {verdict['confidence']:.2%}")
            logger.info(f"Color: {verdict['color']}")
            logger.info("=" * 70 + "\n")

            return results

        except Exception as e:
            logger.error(f"Lỗi trong quá trình fact checking: {str(e)}", exc_info=True)
            results["status"] = "error"
            results["error"] = str(e)
            return results

    def format_result_for_frontend(self, results: Dict[str, Any]) -> Dict[str, Any]:

        if results["status"] == "input_too_short":
            return {
                "success": False,
                "message": results.get("message", "Nội dung không hợp lệ"),
            }

        if results["status"] != "success":
            return {
                "success": False,
                "message": results.get(
                    "message", results.get("error", "Unknown error")
                ),
            }

        verdict = results.get("verdict", {})
        return {
            "success": True,
            "verdict": {
                "label": verdict.get("label"),
                "code": verdict.get("verdict"),
                "explanation": verdict.get("explanation"),
                "color": verdict.get("color"),
                "similarity_percentage": round(
                    results.get("highest_similarity", 0) * 100, 2
                ),
                "confidence_percentage": round(verdict.get("confidence", 0) * 100, 2),
            },
            "references": [
                {
                    "title": ref["title"],
                    "url": ref["url"],
                    "domain": ref["domain"],
                    "source": ref.get("source", ""),
                    "similarity_percentage": round(ref["similarity"] * 100, 2),
                }
                for ref in results.get("top_references", [])
            ],
            "keywords": results.get("processed_data", {}).get("keywords", []),
            "timestamp": results.get("timestamp"),
        }


if __name__ == "__main__":
    pass
