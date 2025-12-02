import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional, List

# Import Config
try:
    from config import Config
except ImportError:
    # Fallback config để test độc lập
    class Config:
        GOOGLE_API_KEY = None
        GOOGLE_API_KEYS_LIST = []
        GOOGLE_CSE_ID = None
        ENABLE_CACHE = True
        USE_GROQ = True
        GROQ_API_KEY = None
        GROQ_MODEL = "llama-3.3-70b-versatile"


# Import Modules
try:
    from preprocessor import TextPreprocessor
    from similarity_checker import SimilarityChecker
    from web_searcher import WebSearcher
    from stance_analyzer import StanceAnalyzer, Stance
    from credibility_scorer import CredibilityScorer
except ImportError as e:
    logging.error(f"Lỗi import quan trọng: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class EnhancedFactChecker:
    """
    ✅ FACT CHECKER v3.0 (OPTIMIZED)
    - Core: Weighted Consensus Algorithm
    - Thresholds: Data-Driven from Training
    - AI: Groq Llama 3.3 70B
    """

    # Định nghĩa giá trị cho từng loại quan điểm
    STANCE_VALUES = {
        "SUPPORT": 1.0,
        "REFUTE": -1.0,
        "DISCUSS": 0.0,
        "UNRELATED": 0.0,
    }

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        use_snippet_mode: bool = True,
    ):
        logger.info("=" * 70)
        logger.info(" Initializing Enhanced Fact Checker v3.0 (Production Ready)")
        logger.info("=" * 70)

        # Lấy danh sách key từ Config
        # Nếu người dùng truyền vào init thì bọc nó thành list, nếu không lấy từ Config
        api_keys = [google_api_key] if google_api_key else Config.GOOGLE_API_KEYS_LIST
        cse_id = google_cse_id or Config.GOOGLE_CSE_ID

        # 1. Preprocessor
        self.preprocessor = TextPreprocessor()

        # 2. Searcher (Truyền List Key)
        self.searcher = WebSearcher(
            google_api_keys=api_keys,
            google_cse_id=cse_id,
            cache_enabled=Config.ENABLE_CACHE,
        )

        # 3. Similarity
        self.similarity_checker = SimilarityChecker()

        # 4. Stance Analyzer (Chỉ dùng Groq, không fallback)
        self.stance_analyzer = StanceAnalyzer(
            groq_api_key=Config.GROQ_API_KEY, groq_model=Config.GROQ_MODEL
        )

        # 5. Credibility
        self.credibility_scorer = CredibilityScorer()

        self.use_snippet_mode = use_snippet_mode
        logger.info("✓ All components initialized successfully.\n")

    def check_fact(
        self,
        user_input: str,
        input_type: str = "text",
        num_sources: Optional[int] = None,
    ):
        if num_sources is None:
            num_sources = 15

        results = {
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "original_input": user_input,
            "status": "processing",
        }

        try:
            # ============= BƯỚC 1: TIỀN XỬ LÝ =============
            logger.info("STEP 1: PREPROCESSING")
            processed = self.preprocessor.process_input(user_input, input_type)

            if not processed or not processed["keywords"]:
                return {"status": "error", "message": "Input too short or invalid"}

            # Lấy text để so sánh
            if input_type == "url":
                claim_text = processed.get("summary_text", processed["full_text"][:800])
            else:
                claim_text = processed["full_text"][:800]

            results["processed_data"] = {
                "keywords": processed["keywords"],
                "domain": processed["domain"],
                "category": processed.get("category", "other"),
                "category_label": processed.get("category_label", "Khác")
            }

            # ============= BƯỚC 2: TÌM KIẾM =============
            logger.info("STEP 2: SEARCHING")
            reference_articles = self.searcher.search_for_fact_check(
                processed, num_sources
            )

            if not reference_articles:
                return {"status": "no_references", "message": "No articles found"}

            # ============= BƯỚC 3: CHUẨN BỊ CONTENT =============
            # Chỉ lấy snippet > 30 ký tự để tránh rác
            reference_contents = []
            for article in reference_articles:
                snippet = article.get("snippet", "")
                if snippet and len(snippet) > 30:
                    reference_contents.append(
                        {
                            "url": article["url"],
                            "title": article["title"],
                            "content": snippet,
                            "domain": article["domain"],
                        }
                    )

            if not reference_contents:
                return {"status": "error", "message": "No valid content to analyze"}

            # ============= BƯỚC 4: STANCE ANALYSIS (AI) =============
            logger.info("STEP 4: STANCE ANALYSIS")
            stance_results = self.stance_analyzer.analyze_stance_batch(
                claim=claim_text, articles=reference_contents
            )

            # ============= BƯỚC 5: SIMILARITY CHECK =============
            logger.info("STEP 5: SIMILARITY CHECK")
            sim_results = self.similarity_checker.calculate_similarity_batch(
                claim_text[:300], [ref["content"] for ref in reference_contents]
            )

            # ============= BƯỚC 6: SCORING (OPTIMIZED) =============
            logger.info("STEP 6: SCORING WITH OPTIMIZED FORMULA")

            evidence_scores = []

            for i, ref in enumerate(reference_contents):
                # Match results
                stance_res = next(
                    (s for s in stance_results if s["article"]["url"] == ref["url"]),
                    None,
                )
                sim_res = next((s for s in sim_results if s["index"] == i), None)

                if not stance_res or not sim_res:
                    continue

                # --- 1. Lấy các chỉ số cơ bản ---
                # Stance
                stance_label = stance_res["stance"].value.upper()
                stance_val = self.STANCE_VALUES.get(stance_label, 0.0)

                # Credibility (Normalized 0-1)
                cred_info = self.credibility_scorer.get_domain_score(ref["url"])
                norm_cred = cred_info["score"] / 10.0

                # Similarity
                raw_sim = sim_res["similarity"]

                # --- 2. Bộ lọc thông minh (Smart Filtering) ---
                # Loại bỏ bài Unrelated hoặc Sim quá thấp
                if stance_label == "UNRELATED":
                    continue
                if raw_sim < 0.25:
                    continue

                # --- 3. Similarity Boost ---
                # Nếu AI đã chắc chắn (Support/Refute), nâng điểm Sim lên để không bị kéo tụt điểm
                effective_sim = raw_sim
                # if stance_label in ["SUPPORT", "REFUTE"]:
                #     effective_sim = max(raw_sim, 0.85)

                # --- 4. Tính Trọng số (Weight Calculation) ---
                # Dùng hàm mũ để ưu tiên nguồn uy tín: Weight = Credibility ^ 2
                weight = norm_cred
                # weight = pow(norm_cred, 2)

                # --- 5. Refutation Boost (Cú đấm thép) ---
                # Nếu nguồn Tier 1 bác bỏ -> Nhân 4 lần sức mạnh
                # if stance_label == "REFUTE":
                #    if cred_info["tier"] == 1:
                #        weight *= 4.0
                #    elif cred_info["tier"] == 2:
                #        weight *= 2.5
                #    elif cred_info["tier"] <= 3:
                #        weight *= 1.5

                # --- 6. Tính điểm thành phần (Evidence Score) ---
                # Score = Stance * Credibility * Sim
                # Lưu ý: Discuss (Stance=0) sẽ có Score=0, nhưng vẫn có Weight > 0
                evidence_score = stance_val * norm_cred * effective_sim

                evidence_scores.append(
                    {
                        "url": ref["url"],
                        "title": ref["title"],
                        "domain": ref["domain"],
                        "stance": stance_res["stance"],
                        "stance_label": stance_res["stance"],
                        "stance_conf": stance_res["confidence"],
                        "similarity": raw_sim,
                        "credibility": cred_info,
                        "evidence_score": evidence_score,
                        "weight": weight,
                    }
                )

                logger.info(
                    f" > {ref['domain'][:15]}: Stance={stance_label}, Score={evidence_score:.2f}, Weight={weight:.2f}"
                )

            if not evidence_scores:
                return {
                    "status": "no_valid_evidence",
                    "message": "Filtered all evidence",
                }

            # ============= BƯỚC 7: TỔNG HỢP (AGGREGATION) =============
            logger.info("STEP 7: FINAL VERDICT")

            numerator = sum(e["evidence_score"] * e["weight"] for e in evidence_scores)
            denominator = sum(e["weight"] for e in evidence_scores)

            # Weighted Average
            final_score = numerator / denominator if denominator > 0 else 0.0

            # Mapping sang kết luận
            verdict_result = self._map_score_to_verdict(
                final_score, len(evidence_scores)
            )

            # Sắp xếp evidence theo trọng số để hiển thị đẹp
            evidence_scores.sort(key=lambda x: x["weight"], reverse=True)

            # Đóng gói kết quả
            results["status"] = "success"
            results["verdict"] = verdict_result
            results["final_score"] = final_score
            results["voting_summary"] = self._create_voting_summary(
                evidence_scores, final_score
            )
            results["all_references"] = evidence_scores[:15]

            return results

        except Exception as e:
            logger.error(f"Fact check failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _map_score_to_verdict(self, score: float, num_evidence: int) -> Dict[str, Any]:
        """
        ✅ 3-LEVEL THRESHOLD (DYNAMIC FROM CONFIG)
        Đồng bộ hoàn toàn với Config.VERDICT_THRESHOLDS
        """

        # 1. Lấy ngưỡng từ Config (hoặc dùng mặc định nếu thiếu)
        t_upper = getattr(Config, "T_UPPER", 0.05)
        t_lower = getattr(Config, "T_LOWER", -0.05)

        # 2. Lấy cấu hình hiển thị từ Config
        thresholds = getattr(Config, "VERDICT_THRESHOLDS", {})

        # Helper để lấy config an toàn
        def get_cfg(key, default_label, default_color):
            item = thresholds.get(key, {})
            return item.get("label", default_label), item.get("color", default_color)

        # === LOGIC PHÂN LOẠI ===

        # TRƯỜNG HỢP 1: LIKELY_TRUE (ĐÚNG)
        if score >= t_upper:
            label, color = get_cfg(
                "LIKELY_TRUE", "Thông tin có khả năng đúng", "#22c55e"
            )
            # Confidence: 0.7 -> 0.99 tùy độ lớn của điểm
            confidence = min(0.99, 0.70 + abs(score) * 0.25)
            explanation = (
                f"Mức độ tin cậy cao, thông tin được xác thực bởi nhiều nguồn uy tín."
            )
            code = "LIKELY_TRUE"

        # TRƯỜNG HỢP 2: LIKELY_FALSE (SAI)
        elif score <= t_lower:
            label, color = get_cfg(
                "LIKELY_FALSE", "Thông tin có khả năng sai", "#ef4444"
            )
            # Confidence: 0.7 -> 0.99 tùy độ lớn của điểm
            confidence = min(0.99, 0.70 + abs(score) * 0.25)
            explanation = f"Thông tin có dấu hiệu sai lệch hoặc đã bị bác bỏ bởi các nguồn uy tín."
            code = "LIKELY_FALSE"

        # TRƯỜNG HỢP 3: UNCERTAIN (KHÔNG CHẮC)
        else:
            label, color = get_cfg("UNCERTAIN", "Không chắc chắn", "#fbbf24")
            confidence = 0.50  # Luôn giữ mức trung bình cho trường hợp này
            explanation = f"Hiện chưa đủ bằng chứng rõ ràng hoặc các nguồn tin đang có ý kiến trái chiều."
            code = "UNCERTAIN"

        return {
            "code": code,
            "label": label,
            "color": color,
            "confidence": confidence,
            "explanation": explanation,
        }

    def _create_voting_summary(self, evidences, final_score):
        """Tạo bảng tổng kết để hiển thị UI"""
        stats = {"support": 0, "refute": 0, "discuss": 0}
        scores = {"support": 0.0, "refute": 0.0, "discuss": 0.0}

        for ev in evidences:
            lbl = ev["stance_label"].lower()
            if lbl in stats:
                stats[lbl] += 1
                scores[lbl] += ev[
                    "weight"
                ]  # Cộng trọng số để hiển thị mức độ ảnh hưởng

        return {
            "total_score": round(final_score * 10, 2),  # Scale 10 cho đẹp
            "support_count": stats["support"],
            "refute_count": stats["refute"],
            "discuss_count": stats["discuss"],
            "support_score": round(scores["support"], 1),
            "refute_score": round(scores["refute"], 1),
            "discuss_score": round(scores["discuss"], 1),
        }

    def format_result_for_frontend(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Chuẩn hóa format trả về API"""
        if results.get("status") != "success":
            return {"success": False, "message": results.get("message", "Error")}

        formatted_refs = []
        for r in results.get("all_references", []):
            formatted_refs.append(
                {
                    "title": r["title"],
                    "url": r["url"],
                    "domain": r["domain"],
                    "stance": {
                        "code": r["stance"].value,
                        "label": self.stance_analyzer.get_stance_label_vi(r["stance"]),
                        "emoji": self.stance_analyzer.get_stance_emoji(r["stance"]),
                        "confidence": round(r["stance_conf"] * 100, 1),
                    },
                    "similarity_percentage": round(r["similarity"] * 100, 1),
                    "credibility": {
                        "score": r["credibility"]["score"],
                        "tier": r["credibility"]["tier"],
                        "label": self.credibility_scorer.get_tier_label(
                            r["credibility"]["tier"]
                        ),
                        "color": self.credibility_scorer.get_tier_color(
                            r["credibility"]["tier"]
                        ),
                    },
                    "weighted_score": round(r["evidence_score"] * 10, 2),
                }
            )

        return {
            "success": True,
            "verdict": {
                "code": results["verdict"]["code"],
                "label": results["verdict"]["label"],
                "explanation": results["verdict"]["explanation"],
                "color": results["verdict"]["color"],
                "confidence_percentage": round(
                    results["verdict"]["confidence"] * 100, 1
                ),
            },
            "voting_summary": results["voting_summary"],
            "references": formatted_refs,
            "keywords": results.get("processed_data", {}).get("keywords", []),
            "timestamp": results.get("timestamp"),
            "processed_data": results.get("processed_data", {})
        }


# Backward compatibility
FactChecker = EnhancedFactChecker
