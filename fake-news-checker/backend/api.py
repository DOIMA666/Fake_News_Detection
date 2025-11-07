from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, Literal
from fact_checker import FactChecker
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API để phát hiện tin giả trên mạng xã hội",
    version="1.0.0"
)

# CORS middleware - cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fact checker (singleton pattern)
fact_checker = None

@app.on_event("startup")
async def startup_event():
    """Initialize fact checker khi server khởi động"""
    global fact_checker
    print("Starting up Fact Checker API...")
    fact_checker = FactChecker()
    print("Fact Checker initialized successfully!")

# Request models
class FactCheckRequest(BaseModel):
    """Request model cho fact checking"""
    content: str
    input_type: Literal['text', 'url'] = 'text'
    num_sources: Optional[int] = 5
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content không được để trống')
        if len(v) < 20:
            raise ValueError('Content quá ngắn (tối thiểu 20 ký tự)')
        return v.strip()
    
    @validator('num_sources')
    def validate_num_sources(cls, v):
        if v is not None and (v < 1 or v > 10):
            raise ValueError('num_sources phải từ 1 đến 10')
        return v

# Response models
class FactCheckResponse(BaseModel):
    """Response model cho fact checking"""
    success: bool
    message: Optional[str] = None
    verdict: Optional[dict] = None
    references: Optional[list] = None
    keywords: Optional[list] = None
    timestamp: Optional[str] = None

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Fake News Detection API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "fact_checker_initialized": fact_checker is not None,
        "endpoints": {
            "check": "/api/check",
            "health": "/health"
        }
    }

@app.post("/api/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest):
    """
    Main endpoint để kiểm tra tin tức
    """
    try:
        if fact_checker is None:
            raise HTTPException(
                status_code=503,
                detail="Fact checker chưa được khởi tạo"
            )
        
        print(f"\n{'='*60}")
        print(f"[API] New request:")
        print(f"  Type: {request.input_type}")
        print(f"  Content: {request.content[:100]}...")
        print(f"{'='*60}\n")
        
        # Chạy fact checking
        result = fact_checker.check_fact(
            user_input=request.content,
            input_type=request.input_type,
            num_sources=request.num_sources
        )
        
        print(f"\n[API] Result status: {result['status']}")
        
        # Format kết quả cho frontend
        formatted_result = fact_checker.format_result_for_frontend(result)
        
        return formatted_result
        
    except ValueError as e:
        print(f"[API] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API] Exception: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý: {str(e)}"
        )

@app.post("/api/check/text")
async def check_text(request: FactCheckRequest):
    """Endpoint riêng cho việc check text"""
    request.input_type = 'text'
    return await check_fact(request)

@app.post("/api/check/url")
async def check_url(request: FactCheckRequest):
    """Endpoint riêng cho việc check URL"""
    request.input_type = 'url'
    return await check_fact(request)

@app.get("/api/trusted-sources")
async def get_trusted_sources():
    """Lấy danh sách các nguồn tin uy tín"""
    if fact_checker is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "sources": fact_checker.searcher.trusted_sources,
        "count": len(fact_checker.searcher.trusted_sources)
    }

# Error handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "success": False,
        "message": "Endpoint không tồn tại",
        "path": request.url.path
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "success": False,
        "message": "Lỗi server nội bộ",
        "detail": str(exc)
    }

# Main entry point
if __name__ == "__main__":
    # Get port from environment variable (for deployment) or use 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Chạy server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )