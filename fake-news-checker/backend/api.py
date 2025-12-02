import os
import traceback
from contextlib import asynccontextmanager
from typing import Literal, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from fact_checker import FactChecker
from database import DatabaseManager
from input_validator import InputValidator

# === GLOBAL INSTANCES ===
fact_checker_instance: Optional[FactChecker] = None
db: Optional[DatabaseManager] = None

input_validator = InputValidator()

# === LIFECYCLE ===


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fact_checker_instance, db
    print("Starting up Fact Checker API...")

    # Initialize Database
    try:
        db = DatabaseManager("fact_checker.db")
        stats = db.get_statistics()
        print(f"✅ Database initialized: {stats['totalChecks']} records")
    except Exception as e:
        print(f"❌ Error initializing Database: {e}")
        traceback.print_exc()

    # Initialize FactChecker
    try:
        fact_checker_instance = FactChecker()
        print("✅ Fact Checker initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing FactChecker: {e}")
        traceback.print_exc()

    yield
    print("Shutting down API.")


# === FASTAPI APP ===

app = FastAPI(
    title="Fake News Detection API with Dashboard",
    description="API để phát hiện tin giả với Dashboard Analytics",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === MODELS ===


class FactCheckRequest(BaseModel):
    content: str
    input_type: Literal["text", "url"] = "text"
    num_sources: Optional[int] = Field(
        default=15, ge=1, le=20, description="Số lượng nguồn tham khảo (1-20)"
    )

class FeedbackRequest(BaseModel):
    check_id: int
    is_correct: bool
# === ENDPOINTS ===


@app.get("/", tags=["Health"])
async def root():
    """Homepage"""
    return {
        "status": "online",
        "message": "Fake News Detection API with Dashboard (SQLite Backend)",
        "version": "3.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        stats = db.get_statistics() if db else {"totalChecks": 0}
        return {
            "status": "healthy",
            "fact_checker_initialized": fact_checker_instance is not None,
            "database_initialized": db is not None,
            "database_records": stats.get("totalChecks", 0),
            "endpoints": {
                "check": "/api/check",
                "stats": "/api/stats",
                "recent": "/api/recent",
                "trending": "/api/trending",
                "history": "/api/history",
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/api/check", tags=["Core"])
async def check_fact(request: FactCheckRequest):
    """Main fact checking endpoint"""
    try:
        # 1. Kiểm tra cơ bản
        if not request.content:
            raise HTTPException(status_code=400, detail="Content is required")

        print(f"\n{'='*60}")
        print("[API] New request:")
        print(f"  Type: {request.input_type}")
        print(f"  Content: {request.content[:100]}...")
        print(f"  Num sources: {request.num_sources}")
        print(f"{'='*60}\n")

        # 2. VALIDATE INPUT (Sử dụng InputValidator)
        # Bước này cực quan trọng để chặn spam/rác trước khi gọi AI tốn tiền
        is_valid, error_msg, suggestion = input_validator.validate(
            request.content, request.input_type
        )

        if not is_valid:
            print(f"[API] ❌ Validation failed: {error_msg}")
            # Trả về kết quả lỗi ngay lập tức mà không cần chạy pipeline
            return JSONResponse(
                content={
                    "success": False,
                    "message": error_msg,
                    "suggestion": suggestion,
                    "verdict": None,  # Không có verdict vì chưa check
                }
            )

        # 3. Chạy pipeline (chỉ chạy khi input hợp lệ)
        result = fact_checker_instance.check_fact(
            user_input=request.content,
            input_type=request.input_type,
            num_sources=request.num_sources,
        )

        print(f"\n[API] Result status: {result['status']}")

        # 4. Format kết quả
        formatted_result = fact_checker_instance.format_result_for_frontend(result)

        # 5. LƯU VÀO DATABASE (Chỉ lưu khi có kết quả xử lý thành công từ pipeline)
        if formatted_result.get("success") and db:
            processed_data = formatted_result.get("processed_data", {})

            history_item = {
                "content_preview": request.content[:500],
                "input_type": request.input_type,
                "verdict": formatted_result.get("verdict", {}),
                "voting_summary": formatted_result.get("voting_summary", {}),
                "keywords": formatted_result.get("keywords", []),
                "timestamp": datetime.now().isoformat(),
                "num_references": len(formatted_result.get("references", [])),
                "processed_data": processed_data,
            }

            try:
                check_id = db.add_check(history_item)
                cat = processed_data.get("category", "other")
                formatted_result["db_id"] = check_id
                print(f"[API] ✅ Saved to database (ID: {check_id}, Category: {cat})")
            except Exception as e:
                print(f"[API] ⚠️ Failed to save to database: {e}")
                traceback.print_exc()

        return JSONResponse(content=formatted_result)

    except ValueError as e:
        print(f"[API] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API] Exception: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {str(e)}")


@app.post("/api/feedback", tags=["Core"])
async def submit_feedback(feedback: FeedbackRequest):
    """Nhận feedback từ người dùng"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")
            
        success = db.submit_feedback(feedback.check_id, feedback.is_correct)
        
        if success:
            return {"success": True, "message": "Đã ghi nhận phản hồi"}
        else:
            # Có thể ID không tồn tại hoặc lỗi DB
            return JSONResponse(
                status_code=404, 
                content={"success": False, "message": "Không tìm thấy bài kiểm tra này"}
            )
            
    except Exception as e:
        print(f"Error feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", tags=["Dashboard"])
async def get_stats():
    """
    Lấy thống kê tổng quan
    """
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")

        stats = db.get_statistics()
        return JSONResponse(content={"success": True, "data": stats})
    except Exception as e:
        print(f"Error in get_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recent", tags=["Dashboard"])
async def get_recent(limit: int = 10):
    """Lấy các lần kiểm tra gần nhất"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")

        if limit > 50:
            limit = 50

        recent = db.get_recent_checks(limit)
        return JSONResponse(
            content={"success": True, "data": recent, "count": len(recent)}
        )
    except Exception as e:
        print(f"Error in get_recent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trending", tags=["Dashboard"])
async def get_trending(limit: int = 5):
    """Lấy các chủ đề nổi bật"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")

        if limit > 10:
            limit = 10

        trending = db.get_trending_topics(limit)
        return JSONResponse(
            content={"success": True, "data": trending, "count": len(trending)}
        )
    except Exception as e:
        print(f"Error in get_trending: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", tags=["Dashboard"])
async def get_history_endpoint(skip: int = 0, limit: int = 20):
    """
    Lấy toàn bộ lịch sử với pagination (dùng cho tab Community)
    """
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")

        if limit > 100:
            limit = 100

        # Sử dụng get_recent_checks với offset
        all_recent = db.get_recent_checks(skip + limit)
        paginated = all_recent[skip : skip + limit]

        # Format cho frontend
        formatted_data = []
        for item in paginated:
            formatted_data.append(
                {
                    "id": item.get("id"),
                    "content_preview": item.get("title", ""),
                    "verdict_code": item.get("verdict"),
                    "verdict_label": _get_verdict_label(item.get("verdict")),
                    "confidence_percentage": item.get("confidence", 0),
                    "timestamp": item.get("timestamp", datetime.now().isoformat()),
                    "input_type": "text",
                    "num_references": 0,
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "data": formatted_data,
                "total": len(all_recent),
                "skip": skip,
                "limit": limit,
                "has_more": (skip + limit) < len(all_recent),
            }
        )
    except Exception as e:
        print(f"Error in get_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/clear", tags=["Dashboard"])
async def clear_history_endpoint():
    """Xóa toàn bộ lịch sử"""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")

        db.clear_all()
        return JSONResponse(
            content={"success": True, "message": "Đã xóa toàn bộ lịch sử"}
        )
    except Exception as e:
        print(f"Error in clear_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === HELPER FUNCTIONS ===


# === HELPER FUNCTIONS ===


def _get_verdict_label(verdict_code: str) -> str:
    """Map verdict code sang label tiếng Việt (3 Levels)"""
    labels = {
        "LIKELY_TRUE": "Thông tin có khả năng đúng",
        "LIKELY_FALSE": "Thông tin có khả năng sai",
        "UNCERTAIN": "Không chắc chắn / Cần kiểm chứng thêm",
    }
    return labels.get(verdict_code, "Không rõ")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False, log_level="info")
