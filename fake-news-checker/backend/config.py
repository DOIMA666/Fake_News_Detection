import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Enhanced Configuration hỗ trợ Groq Cloud & Local Ollama"""

    # === API Keys ===
    # 1. Lấy chuỗi thô từ env (Biến GOOGLE_API_KEY chứa danh sách phân cách bởi dấu phẩy)
    _RAW_GOOGLE_KEYS = os.getenv("GOOGLE_API_KEY", "")

    # 2. Tách chuỗi thành List các key (Xử lý dấu phẩy và khoảng trắng)
    GOOGLE_API_KEYS_LIST = [k.strip() for k in _RAW_GOOGLE_KEYS.split(",") if k.strip()]

    # 3. Active Key mặc định (cho các module cũ hoặc logic chỉ cần 1 key)
    ACTIVE_GOOGLE_KEY = GOOGLE_API_KEYS_LIST[0] if GOOGLE_API_KEYS_LIST else None

    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

    # === GROQ CLOUD (Primary AI) ===
    USE_GROQ = True
    # Groq Key cũng có thể là list, giữ nguyên string để StanceAnalyzer tự xử lý
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # === Cache Configuration ===
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

    # === Search Configuration ===
    DEFAULT_NUM_RESULTS = int(os.getenv("DEFAULT_NUM_RESULTS", "15"))
    MAX_NUM_RESULTS = int(os.getenv("MAX_NUM_RESULTS", "20"))

    # === Model Configuration ===
    SIMILARITY_MODEL = os.getenv(
        "SIMILARITY_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
    )

    # === Verdict Thresholds ===
    # Định nghĩa các hằng số ngưỡng tại đây để dễ quản lý
    T_UPPER = 0.05
    T_LOWER = -0.05

    VERDICT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
        "LIKELY_TRUE": {"label": "Thông tin có khả năng đúng", "color": "#22c55e"},
        "LIKELY_FALSE": {"label": "Thông tin có khả năng sai", "color": "#ef4444"},
        "UNCERTAIN": {"label": "Không chắc chắn", "color": "#fbbf24"},
    }

    # === Feature Flags ===
    ENABLE_STANCE_DETECTION = True
    ENABLE_CREDIBILITY_SCORING = True

    # === Server Configuration ===
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate configuration and show status"""
        print("\n" + "=" * 70)
        print(" FACT CHECKER SYSTEM STATUS (GROQ ONLY)")
        print("=" * 70)

        # Google Search
        if cls.GOOGLE_API_KEYS_LIST and cls.GOOGLE_CSE_ID:
            print(" ✓ Google Search API:  CONNECTED")
            print(f"   → Loaded {len(cls.GOOGLE_API_KEYS_LIST)} API Keys for rotation")
        else:
            print(" ✗ Google Search API:  MISSING (Search will fail)")

        print("-" * 70)

        # AI Status
        if cls.GROQ_API_KEY:
            print(f" ✓ AI Engine (Groq):   CONNECTED ({cls.GROQ_MODEL})")
        else:
            print(" ✗ AI Engine (Groq):   MISSING API KEY")

        print("-" * 70)
        print(f" Server: {cls.API_HOST}:{cls.API_PORT}")
        print("=" * 70 + "\n")

        return bool(cls.GOOGLE_API_KEYS_LIST and cls.GOOGLE_CSE_ID and cls.GROQ_API_KEY)


# Validate on import
Config.validate()
