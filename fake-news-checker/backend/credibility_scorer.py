import logging
from typing import Dict, Optional, List, Set
from urllib.parse import urlparse
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CredibilityScorer:

    def __init__(self, enable_time_decay: bool = False):
        """Khởi tạo với database mở rộng"""
        self.enable_time_decay = enable_time_decay
        self.reputation_cache = {}  # Cache điểm tin cậy

        # === TIER 1: Báo chí chính thống (9-10 điểm) ===
        self.tier1_sources = {
            # Báo Đảng, Nhà nước
            "nhandan.vn": {"score": 10.0, "name": "Nhân Dân", "tier": 1},
            "baochinhphu.vn": {"score": 10.0, "name": "Báo Chính phủ", "tier": 1},
            "qdnd.vn": {"score": 9.8, "name": "Quân Đội Nhân Dân", "tier": 1},
            "cand.com.vn": {"score": 9.8, "name": "Công An Nhân Dân", "tier": 1},
            "xaydungchinhsach.chinhphu.vn": {
                "score": 10.0,
                "name": "Xây dựng Chính sách",
                "tier": 1,
            },
            # TTXVN và các cơ quan thông tấn
            "vietnamplus.vn": {"score": 9.8, "name": "VietnamPlus (TTXVN)", "tier": 1},
            "baotintuc.vn": {"score": 9.5, "name": "Báo Tin Tức (TTXVN)", "tier": 1},
            # Báo trung ương lớn
            "vnexpress.net": {"score": 10.0, "name": "VnExpress", "tier": 1},
            "tuoitre.vn": {"score": 10.0, "name": "Tuổi Trẻ", "tier": 1},
            "thanhnien.vn": {"score": 10.0, "name": "Thanh Niên", "tier": 1},
            "dantri.com.vn": {"score": 9.5, "name": "Dân Trí", "tier": 1},
            # Đài truyền hình/phát thanh
            "vtv.vn": {"score": 10.0, "name": "VTV", "tier": 1},
            "vov.vn": {"score": 8.5, "name": "VOV", "tier": 1},
        }

        # === TIER 2: Báo điện tử uy tín (7-8.5 điểm) ===
        self.tier2_sources = {
            # Báo trung ương khác
            "vietnamnet.vn": {"score": 8.5, "name": "VietnamNet", "tier": 2},
            "laodong.vn": {"score": 8.5, "name": "Lao Động", "tier": 2},
            "tienphong.vn": {"score": 8.5, "name": "Tiền Phong", "tier": 2},
            "sggp.org.vn": {"score": 8.0, "name": "Sài Gòn Giải Phóng", "tier": 2},
            "nld.com.vn": {"score": 8.0, "name": "Người Lao Động", "tier": 2},
            # Báo chuyên ngành
            "baodautu.vn": {"score": 8.3, "name": "Báo Đầu Tư", "tier": 2},
            "baogiaothong.vn": {"score": 8.3, "name": "Báo Giao Thông", "tier": 2},
            "baophapluat.vn": {"score": 8.3, "name": "Báo Pháp Luật", "tier": 2},
            "vietstock.vn": {"score": 8.0, "name": "VietStock", "tier": 2},
            "vneconomy.vn": {"score": 8.0, "name": "VnEconomy", "tier": 2},
            # Báo địa phương lớn
            "baomoi.com": {"score": 7.5, "name": "Báo Mới", "tier": 2},
            "zingnews.vn": {"score": 7.5, "name": "Zing News", "tier": 2},
            "kenh14.vn": {"score": 7.0, "name": "Kenh14", "tier": 2},
        }

        # === TIER 3: Trang tin đăng ký (5-6.5 điểm) ===
        self.tier3_sources = {
            # Trang tin tổng hợp
            "24h.com.vn": {"score": 6.5, "name": "24h", "tier": 3},
            "soha.vn": {"score": 6.0, "name": "Soha", "tier": 3},
            "tinnhanhchungkhoan.vn": {"score": 6.0, "name": "Tin Nhanh CK", "tier": 3},
            # Trang tin kinh tế
            "cafef.vn": {"score": 6.5, "name": "CafeF", "tier": 3},
            "cafebiz.vn": {"score": 6.0, "name": "CafeBiz", "tier": 3},
            "thoibaotaichinhvietnam.vn": {
                "score": 6.5,
                "name": "Thời báo Tài chính",
                "tier": 3,
            },
            # Trang tin công nghệ/giải trí
            "genk.vn": {"score": 5.5, "name": "Genk", "tier": 3},
            "afamily.vn": {"score": 5.5, "name": "Afamily", "tier": 3},
            "nguoiduatin.vn": {"score": 5.5, "name": "Người Đưa Tin", "tier": 3},
            "eva.vn": {"score": 5.5, "name": "Eva.vn", "tier": 3},
            "vnreview.vn": {"score": 6.0, "name": "VNReview", "tier": 3},
        }

        # === BLACKLIST: Nguồn tin giả nổi tiếng (0-2 điểm) ===
        self.blacklist_domains = {
            "baotintuc.org.vn": {"score": 1.0, "reason": "Giả mạo TTXVN"},
            "tinmoi.com": {"score": 2.0, "reason": "Aggregator không xác thực"},
            "tinnongngay.com": {"score": 1.5, "reason": "Clickbait, tin đồn"},
            # Thêm các domain tin giả khác nếu biết
        }

        # === SOCIAL MEDIA & BLOG PATTERNS ===
        self.social_patterns = [
            "facebook.com",
            "fb.com",
            "m.facebook.com",
            "youtube.com",
            "youtu.be",
            "tiktok.com",
            "twitter.com",
            "x.com",
            "instagram.com",
            "zalo.me",
            "blogspot.com",
            "wordpress.com",
            "medium.com",
            "reddit.com",
            "quora.com",
        ]

        self.all_sources = {
            **self.tier1_sources,
            **self.tier2_sources,
            **self.tier3_sources,
        }

        logger.info(f"CredibilityScorer initialized:")
        logger.info(f"   - {len(self.tier1_sources)} Tier 1 sources")
        logger.info(f"   - {len(self.tier2_sources)} Tier 2 sources")
        logger.info(f"   - {len(self.tier3_sources)} Tier 3 sources")
        logger.info(f"   - {len(self.blacklist_domains)} blacklisted domains")

    def get_domain_score(
        self, url: str, article_date: Optional[datetime] = None
    ) -> Dict[str, any]:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")

            if domain in self.blacklist_domains:
                info = self.blacklist_domains[domain]
                logger.warning(f"⚠️ BLACKLISTED domain: {domain} - {info['reason']}")
                return {
                    "domain": domain,
                    "score": info["score"],
                    "tier": 4,
                    "name": domain,
                    "is_trusted": False,
                    "is_blacklisted": True,
                    "reason": info["reason"],
                }

            if domain in self.all_sources:
                info = self.all_sources[domain]
                base_score = info["score"]

                if self.enable_time_decay and article_date:
                    base_score = self._apply_time_decay(base_score, article_date)

                return {
                    "domain": domain,
                    "score": base_score,
                    "tier": info["tier"],
                    "name": info["name"],
                    "is_trusted": True,
                    "is_blacklisted": False,
                }
            if domain.endswith(".gov.vn"):
                return {
                    "domain": domain,
                    "score": 9.0,
                    "tier": 1,
                    "name": domain,
                    "is_trusted": True,
                    "is_blacklisted": False,
                    "note": "Government domain",
                }
            elif domain.endswith((".edu.vn", ".ac.vn")):
                return {
                    "domain": domain,
                    "score": 8.0,
                    "tier": 2,
                    "name": domain,
                    "is_trusted": True,
                    "is_blacklisted": False,
                    "note": "Educational domain",
                }
            elif domain.endswith(".org.vn"):
                return {
                    "domain": domain,
                    "score": 7.0,
                    "tier": 2,
                    "name": domain,
                    "is_trusted": True,
                    "is_blacklisted": False,
                    "note": "Organization domain",
                }

            if any(pattern in domain for pattern in self.social_patterns):
                return {
                    "domain": domain,
                    "score": 2.0,
                    "tier": 4,
                    "name": domain,
                    "is_trusted": False,
                    "is_blacklisted": False,
                    "note": "Social media/Blog",
                }

            return {
                "domain": domain,
                "score": 4.0,
                "tier": 3,
                "name": domain,
                "is_trusted": False,
                "is_blacklisted": False,
                "note": "Unknown source",
            }

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return {
                "domain": "unknown",
                "score": 3.0,
                "tier": 4,
                "name": "Unknown",
                "is_trusted": False,
                "is_blacklisted": False,
            }

    def _apply_time_decay(self, base_score: float, article_date: datetime) -> float:
        """
        Giảm điểm cho bài báo cũ (time decay)
        - Bài > 1 năm: -5%
        - Bài > 2 năm: -10%
        - Bài > 3 năm: -15%
        """
        now = datetime.now()
        age_days = (now - article_date).days

        if age_days > 1095:  # > 3 years
            return base_score * 0.85
        elif age_days > 730:  # > 2 years
            return base_score * 0.90
        elif age_days > 365:  # > 1 year
            return base_score * 0.95
        return base_score

    def is_blacklisted(self, url: str) -> bool:
        """Kiểm tra nhanh xem domain có bị blacklist không"""
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")
            return domain in self.blacklist_domains
        except:
            return False

    def get_tier_color(self, tier: int) -> str:
        """Trả về màu cho mỗi tier"""
        colors = {
            1: "#22c55e",  
            2: "#3b82f6",  
            3: "#f59e0b",  
            4: "#6b7280",  
        }
        return colors.get(tier, "#9ca3af")

    def get_tier_label(self, tier: int) -> str:
        """Trả về nhãn cho mỗi tier"""
        labels = {
            1: "Nguồn uy tín cao",
            2: "Nguồn đáng tin cậy",
            3: "Nguồn trung bình",
            4: "Nguồn độc lập/Chưa xác thực",
        }
        return labels.get(tier, "Nguồn không xác định")

    def get_trusted_domains_list(self) -> List[str]:
        """Trả về list domains đáng tin cậy"""
        return list(self.all_sources.keys())

    def get_blacklist(self) -> Dict[str, Dict]:
        """Trả về blacklist"""
        return self.blacklist_domains

    def add_to_blacklist(self, domain: str, reason: str, score: float = 1.0):
        """Thêm domain vào blacklist động"""
        domain = domain.lower().replace("www.", "")
        self.blacklist_domains[domain] = {
            "score": score,
            "reason": reason,
            "added_at": datetime.now().isoformat(),
        }
        logger.info(f" Added to blacklist: {domain} - {reason}")

    def get_statistics(self) -> Dict[str, int]:
        """Thống kê số lượng nguồn"""
        return {
            "tier1": len(self.tier1_sources),
            "tier2": len(self.tier2_sources),
            "tier3": len(self.tier3_sources),
            "blacklisted": len(self.blacklist_domains),
            "total_trusted": len(self.all_sources),
        }

    def export_database(self) -> Dict[str, any]:
        """Xuất toàn bộ database"""
        return {
            "tier1": self.tier1_sources,
            "tier2": self.tier2_sources,
            "tier3": self.tier3_sources,
            "blacklist": self.blacklist_domains,
            "statistics": self.get_statistics(),
        }


if __name__ == "__main__":
    pass
