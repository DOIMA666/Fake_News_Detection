import os
import traceback
from contextlib import asynccontextmanager
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from fact_checker import FactChecker


fact_checker_instance: Optional[FactChecker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global fact_checker_instance
    print("Starting up Fact Checker API...")
    fact_checker_instance = FactChecker()
    print("Fact Checker initialized successfully!")
    yield
    print("Shutting down API.")


# Khởi tạo FastAPI app với lifespan
app = FastAPI(
    title="Fake News Detection API",
    description="API để phát hiện tin giả trên mạng xã hội",
    version="1.0.1",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Các Lớp Model (Pydantic) ---


class FactCheckRequest(BaseModel):

    content: str
    input_type: Literal["text", "url"] = "text"
    num_sources: Optional[int] = Field(
        default=5, ge=1, le=10, description="Số lượng nguồn tham khảo (1-10)"
    )

    @model_validator(mode="after")
    def validate_content_based_on_type(self):
        if not self.content or not self.content.strip():
            raise ValueError("Content không được để trống")

        if self.input_type == "text":
            word_count = len(self.content.split())
            if word_count < 3:
                raise ValueError("Content quá ngắn (tối thiểu 3 từ)")

        self.content = self.content.strip()
        return self


class FactCheckResponse(BaseModel):

    success: bool
    message: Optional[str] = None
    verdict: Optional[dict] = None
    references: Optional[list] = None
    keywords: Optional[list] = None
    timestamp: Optional[str] = None


# --- API Endpoints ---


@app.get("/", tags=["Health"])
async def root():
    """Endpoint cơ bản để kiểm tra server có đang chạy hay không."""
    return {
        "status": "online",
        "message": "Fake News Detection API is running",
        "version": "1.0.1",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "fact_checker_initialized": fact_checker_instance is not None,
        "endpoints": {"check": "/api/check", "health": "/health"},
    }


@app.post("/api/check", response_model=FactCheckResponse, tags=["Core"])
async def check_fact(request: FactCheckRequest):
    try:
        if fact_checker_instance is None:
            raise HTTPException(
                status_code=503, detail="Fact checker chưa được khởi tạo"
            )

        print(f"\n{'='*60}")
        print("[API] New request:")
        print(f"  Type: {request.input_type}")
        print(f"  Content: {request.content[:100]}...")
        print(f"{'='*60}\n")

        # Chạy pipeline kiểm tra
        result = fact_checker_instance.check_fact(
            user_input=request.content,
            input_type=request.input_type,
            num_sources=request.num_sources,
        )

        print(f"\n[API] Result status: {result['status']}")

        # Format kết quả cho frontend
        formatted_result = fact_checker_instance.format_result_for_frontend(result)

        return JSONResponse(content=formatted_result)

    except ValueError as e:
        # Lỗi 400 cho các vấn đề validation (ví dụ: text quá ngắn)
        print(f"[API] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Lỗi 500 cho các lỗi server nội bộ
        print(f"[API] Exception: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {str(e)}")


@app.get("/api/trusted-sources", tags=["Utility"])
async def get_trusted_sources():
    if fact_checker_instance is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Trả về danh sách tên (keys) của các nguồn
        sources_list = list(fact_checker_instance.searcher.trusted_sources.keys())
        return {"sources": sources_list, "count": len(sources_list)}
    except Exception:
        raise HTTPException(status_code=500, detail="Source list not available")


# --- Xử lý Lỗi (Error Handlers) ---


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint không tồn tại",
            "path": str(request.url),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Lỗi server nội bộ",
            "detail": str(exc),
        },
    )


# --- Điểm vào (Entry Point) ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False, log_level="info")
