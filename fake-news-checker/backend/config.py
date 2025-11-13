import os
from typing import Dict, Any, List

from dotenv import load_dotenv

load_dotenv()


class Config:

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

    # --- Cấu hình Cache ---
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

    # --- Cấu hình logic tìm kiếm ---
    DEFAULT_NUM_RESULTS = int(os.getenv("DEFAULT_NUM_RESULTS", "5"))
    MAX_NUM_RESULTS = int(os.getenv("MAX_NUM_RESULTS", "10"))

    # --- Cấu hình mô hình AI ---
    SIMILARITY_MODEL = os.getenv(
        "SIMILARITY_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
    )

    VERDICT_THRESHOLDS: Dict[str, float] = {
        "HIGHLY_LIKELY_TRUE": 0.85,
        "LIKELY_TRUE": 0.70,
        "UNCERTAIN": 0.50,
        "LIKELY_FALSE": 0.30,
        "HIGHLY_LIKELY_FALSE": 0.0,
    }

    # --- Cấu hình Server (Uvicorn) ---
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Cấu hình mức độ log
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        print("\n" + "=" * 70)
        print(" CONFIGURATION STATUS")
        print("=" * 70)

        # [Logic đã cập nhật] Kiểm tra xem Google API (chiến lược duy nhất)
        if cls.GOOGLE_API_KEY and cls.GOOGLE_CSE_ID:
            print(" Google Custom Search API: CONFIGURED")
            print("   → Will use Google API for searching (RECOMMENDED)")
        else:
            # Nếu không có API, web_searcher.py sẽ không hoạt động
            print(" Google Custom Search API: NOT CONFIGURED")
            print("   → ERROR: Search functionality will NOT work.")
            print("   → Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env file.")

        if cls.NEWS_API_KEY:
            print(" NewsAPI: CONFIGURED")
        else:
            print("   NewsAPI: NOT CONFIGURED (optional)")

        print(f"\n Cache: {'ENABLED' if cls.ENABLE_CACHE else 'DISABLED'}")
        print(f" Cache TTL: {cls.CACHE_TTL_HOURS} hours")
        print(f" Default results: {cls.DEFAULT_NUM_RESULTS}")
        print(f" Similarity model: {cls.SIMILARITY_MODEL}")
        print(f" API Server: {cls.API_HOST}:{cls.API_PORT}")
        print("=" * 70 + "\n")

        return True


Config.validate()
