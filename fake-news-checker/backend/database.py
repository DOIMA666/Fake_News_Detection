import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager

class DatabaseManager:
    """
    Quản lý database SQLite cho hệ thống fact-checking.
    Phiên bản Final: Đã sửa lỗi sqlite3.Row không dùng được .get()
    """
    
    def __init__(self, db_path: str = "fact_checker.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager để quản lý kết nối database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Trả về object Row để truy cập bằng tên cột
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Khởi tạo schema database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Bảng chính: lưu lịch sử kiểm tra
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS check_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_preview TEXT NOT NULL,
                    input_type TEXT NOT NULL CHECK(input_type IN ('text', 'url')),
                    verdict_code TEXT NOT NULL,
                    verdict_label TEXT,
                    confidence_percentage REAL,
                    category TEXT DEFAULT 'other',
                    
                    -- Điểm số chi tiết
                    total_score REAL DEFAULT 0,
                    support_count INTEGER DEFAULT 0,
                    refute_count INTEGER DEFAULT 0,
                    discuss_count INTEGER DEFAULT 0,
                    
                    num_references INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Feedback
                    user_feedback TEXT CHECK(user_feedback IN ('CORRECT', 'INCORRECT'))
                )
            """)

            # Tạo indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict_code ON check_history(verdict_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON check_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_type ON check_history(input_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON check_history(category)")
            
            # 2. Bảng phụ: từ khóa
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    position INTEGER,
                    FOREIGN KEY (check_id) REFERENCES check_history(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword ON keywords(keyword)")
            
            # 3. Bảng thống kê
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_checks INTEGER DEFAULT 0,
                    true_news INTEGER DEFAULT 0,
                    false_news INTEGER DEFAULT 0,
                    uncertain INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("INSERT OR IGNORE INTO statistics (id) VALUES (1)")
            print(f"✅ Database initialized: {self.db_path}")
    
    def add_check(self, check_data: Dict) -> int:
        """Thêm 1 lần kiểm tra vào database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Parse dữ liệu
            verdict = check_data.get('verdict', {})
            voting = check_data.get('voting_summary', {}) 
            keywords = check_data.get('keywords', [])
            
            category = 'other'
            if 'processed_data' in check_data:
                category = check_data['processed_data'].get('category', 'other')
            elif 'category' in check_data:
                category = check_data['category']
            
            # Insert đầy đủ cột
            cursor.execute("""
                INSERT INTO check_history (
                    content_preview, input_type, verdict_code, verdict_label,
                    confidence_percentage, category, num_references, timestamp,
                    total_score, support_count, refute_count, discuss_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                check_data.get('content_preview', '')[:500],
                check_data.get('input_type', 'text'),
                verdict.get('code', 'UNCERTAIN'),
                verdict.get('label', ''),
                verdict.get('confidence_percentage', 0),
                category,
                check_data.get('num_references', 0),
                check_data.get('timestamp', datetime.now().isoformat()),
                
                voting.get('total_score', 0),
                voting.get('support_count', 0),
                voting.get('refute_count', 0),
                voting.get('discuss_count', 0)
            ))
            
            check_id = cursor.lastrowid
            
            # Insert keywords
            for idx, kw in enumerate(keywords[:15]):
                cursor.execute("""
                    INSERT INTO keywords (check_id, keyword, position)
                    VALUES (?, ?, ?)
                """, (check_id, kw[:100], idx))
            
            # Cập nhật statistics
            self._update_statistics(cursor, verdict.get('code', 'UNCERTAIN'))
            
            return check_id
    
    def submit_feedback(self, check_id: int, is_correct: bool) -> bool:
        """Lưu phản hồi người dùng"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                feedback_val = 'CORRECT' if is_correct else 'INCORRECT'
                
                cursor.execute("""
                    UPDATE check_history 
                    SET user_feedback = ? 
                    WHERE id = ?
                """, (feedback_val, check_id))
                
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return False

    def _update_statistics(self, cursor, verdict_code: str):
        """Cập nhật thống kê"""
        cursor.execute("UPDATE statistics SET total_checks = total_checks + 1")
        
        if verdict_code == 'LIKELY_TRUE':
            cursor.execute("UPDATE statistics SET true_news = true_news + 1")
        elif verdict_code == 'LIKELY_FALSE':
            cursor.execute("UPDATE statistics SET false_news = false_news + 1")
        else:
            cursor.execute("UPDATE statistics SET uncertain = uncertain + 1")
        
        cursor.execute("UPDATE statistics SET last_updated = CURRENT_TIMESTAMP")
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê tổng quan"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM statistics WHERE id = 1")
            row = cursor.fetchone()
            
            if not row:
                return {
                    "totalChecks": 0, "trueNews": 0, "falseNews": 0, 
                    "uncertain": 0, "todayChecks": 0, "accuracy": 0.0
                }
            
            # Tính Accuracy từ Feedback
            cursor.execute("SELECT COUNT(*) FROM check_history WHERE user_feedback IS NOT NULL")
            total_feedback = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM check_history WHERE user_feedback = 'CORRECT'")
            positive_feedback = cursor.fetchone()[0]
            
            real_accuracy = 0.0
            if total_feedback > 0:
                real_accuracy = (positive_feedback / total_feedback) * 100
            
            # Đếm checks hôm nay
            cursor.execute("""
                SELECT COUNT(*) as today_count
                FROM check_history
                WHERE DATE(timestamp) = DATE('now')
            """)
            today_row = cursor.fetchone()
            today = today_row['today_count'] if today_row else 0
            
            return {
                "totalChecks": row['total_checks'],
                "trueNews": row['true_news'],
                "falseNews": row['false_news'],
                "uncertain": row['uncertain'],
                "todayChecks": today,
                "accuracy": round(real_accuracy, 1),
                "total_feedback": total_feedback
            }
    
    def get_recent_checks(self, limit: int = 10) -> List[Dict]:
        """
        Lấy các lần kiểm tra gần nhất.
        ✅ ĐÃ SỬA LỖI: Dùng cú pháp row['key'] thay vì row.get()
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Lấy tất cả các cột cần thiết
            cursor.execute("""
                SELECT id, content_preview, verdict_code, verdict_label,
                       confidence_percentage, timestamp, input_type, num_references
                FROM check_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                # Tính time ago an toàn
                try:
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    delta = datetime.now() - timestamp
                    
                    if delta.days > 0:
                        time_ago = f"{delta.days} ngày trước"
                    elif delta.seconds >= 3600:
                        time_ago = f"{delta.seconds // 3600} giờ trước"
                    elif delta.seconds >= 60:
                        time_ago = f"{delta.seconds // 60} phút trước"
                    else:
                        time_ago = "Vừa xong"
                except:
                    time_ago = ""

                # ✅ FIX Ở ĐÂY: Truy cập trực tiếp index, dùng 'or' để xử lý NULL
                results.append({
                    "id": row['id'],
                    "title": (row['content_preview'] or "")[:60] + "...",
                    "verdict": row['verdict_code'],
                    "verdict_label": row['verdict_label'] or '',
                    "confidence": round(row['confidence_percentage'] or 0, 1),
                    "input_type": row['input_type'] or 'text',
                    "num_references": row['num_references'] or 0,
                    "time": time_ago,
                    "timestamp": row['timestamp']
                })
            
            return results
    
    def get_trending_topics(self, limit: int = 5, days: int = 7) -> List[Dict]:
        """Lấy các chủ đề nổi bật"""
        CATEGORY_MAP = {
            "politics": "Chính trị", "crime": "Pháp luật", "health": "Y tế",
            "entertainment": "Giải trí", "sports": "Thể thao", "economy": "Kinh tế",
            "technology": "Công nghệ", "education": "Giáo dục", "society": "Xã hội",
            "international": "Quốc tế", "other": "Khác"
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM check_history
                WHERE category IS NOT NULL 
                  AND category != ''
                  AND DATE(timestamp) >= DATE('now', '-' || ? || ' days')
                GROUP BY category
                ORDER BY count DESC
                LIMIT ?
            """, (days, limit))
            
            results = []
            for row in cursor.fetchall():
                cat_code = row['category']
                label = CATEGORY_MAP.get(cat_code, "Chủ đề khác")
                
                results.append({
                    "topic": label,
                    "count": row['count'],
                    "trend": 'up' 
                })
            
            return results
    
    def clear_all(self):
        """Xóa toàn bộ dữ liệu"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM keywords")
            cursor.execute("DELETE FROM check_history")
            cursor.execute("""
                UPDATE statistics SET 
                total_checks = 0, true_news = 0, false_news = 0, uncertain = 0
            """)
            print("🗑️ Database cleared")
            
    def export_to_json(self, output_path: str = "backup.json"):
        """Xuất database ra file JSON"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM check_history ORDER BY timestamp DESC
            """)
            
            # Convert row objects to dicts
            data = [dict(row) for row in cursor.fetchall()]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Exported {len(data)} records to {output_path}")


if __name__ == "__main__":
    pass