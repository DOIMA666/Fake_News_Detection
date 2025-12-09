"""
PERFORMANCE TESTING SUITE v2.1 - RATE LIMIT HANDLER
====================================================
Limits: 30 RPM | 1K RPD | 12K TPM | 100K TPD
"""

import json
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from collections import defaultdict
import statistics

# ============================
# 1. RATE LIMIT DETECTOR
# ============================

class RateLimitDetector:
    """Phát hiện và xử lý rate limit errors"""
    RATE_LIMIT_INDICATORS = [
        "rate limit",
        "quota exceeded",
        "too many requests",
        "429",
        "limit reached",
        "requests per minute",
        "tokens per minute",
        "rpm",
        "tpm"
    ]
    
    @classmethod
    def is_rate_limited(cls, error_message: str, status_code: int = None) -> bool:
        """Kiểm tra xem có phải lỗi rate limit không"""
        if status_code == 429:
            return True
        
        error_lower = str(error_message).lower()
        return any(indicator in error_lower for indicator in cls.RATE_LIMIT_INDICATORS)
    
    @classmethod
    def extract_retry_after(cls, response: requests.Response) -> Optional[int]:
        """Lấy thời gian retry từ response header"""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                pass
        return None


# ============================
# 2. CSV DATA LOADER
# ============================

class CSVDataLoader:
    """Đọc Ground Truth từ file CSV"""
    
    def __init__(self, csv_path: str = "training_results.csv"):
        self.csv_path = Path(csv_path)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Đọc CSV và convert sang format test dataset"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        dataset = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, start=1):
                true_label = float(row['true_label'])
                
                if true_label == 1:
                    expected_verdict = "LIKELY_TRUE"
                elif true_label == -1:
                    expected_verdict = "LIKELY_FALSE"
                else:
                    expected_verdict = "UNCERTAIN"
                
                dataset.append({
                    "id": idx,
                    "content": row['claim'].strip(),
                    "type": "text",
                    "expected_verdict": expected_verdict,
                    "category": self._infer_category(row['claim']),
                    "note": f"From training CSV (original verdict: {row.get('verdict_code', 'N/A')})",
                    "csv_row": row
                })
        
        print(f"✓ Loaded {len(dataset)} test cases from {self.csv_path}")
        return dataset
    
    def _infer_category(self, claim: str) -> str:
        """Tự động phân loại category dựa trên keyword"""
        claim_lower = claim.lower()
        
        if any(word in claim_lower for word in ['ngân hàng', 'tài khoản', 'tiền', 'lừa đảo', 'chuyển khoản']):
            return 'crime'
        elif any(word in claim_lower for word in ['y tế', 'bệnh', 'thuốc', 'vaccine', 'sức khỏe']):
            return 'health'
        elif any(word in claim_lower for word in ['chính phủ', 'thủ tướng', 'chính sách', 'quốc hội']):
            return 'politics'
        elif any(word in claim_lower for word in ['kinh tế', 'gdp', 'tăng trưởng', 'xuất khẩu']):
            return 'economy'
        elif any(word in claim_lower for word in ['5g', 'công nghệ', 'ai', 'internet']):
            return 'technology'
        else:
            return 'other'


# ============================
# 3. ENHANCED TEST CONFIG
# ============================

class TestConfig:
    API_URL = "http://localhost:8000"
    NUM_SOURCES = 15
    TIMEOUT = 120  
    OUTPUT_DIR = Path("test_results")
    
    # CSV Settings
    USE_CSV = True
    CSV_PATH = r"C:\Users\ACER\Downloads\training_results.csv"
    
    # Test Settings
    MAX_TESTS = None
    SKIP_FIRST_N = 0
    DELAY_BETWEEN_TESTS = 2  
    
    # Rate Limit Settings
    MAX_RETRIES = 2  
    RETRY_DELAY = 60  
    STOP_ON_RATE_LIMIT = True  
    
    # API Quota Limits 
    RPM_LIMIT = 30  # Requests per minute
    RPD_LIMIT = 1000  # Requests per day
    TPM_LIMIT = 12000  # Tokens per minute
    TPD_LIMIT = 100000  # Tokens per day
    
    VERDICT_MAPPING = {
        "LIKELY_TRUE": "ĐÚNG",
        "LIKELY_FALSE": "SAI", 
        "UNCERTAIN": "KHÔNG CHẮC"
    }


# ============================
# 4. FALLBACK DATASET
# ============================

FALLBACK_DATASET = [
    {
        "id": 1,
        "content": "Việt Nam đã gia nhập ASEAN vào năm 1995",
        "type": "text",
        "expected_verdict": "LIKELY_TRUE",
        "category": "politics",
        "note": "Sự kiện lịch sử có thể xác minh"
    },
    {
        "id": 2,
        "content": "Uống nước chanh nóng có thể chữa khỏi ung thư",
        "type": "text",
        "expected_verdict": "LIKELY_FALSE",
        "category": "health",
        "note": "Tin đồn y tế phổ biến"
    },
]


# ============================
# 5. PERFORMANCE TESTER WITH RATE LIMIT HANDLING
# ============================

class PerformanceTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.results = []
        self.metrics = {
            "total_tests": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "errors": 0,
            "rate_limit_errors": 0,
            "total_time": 0,
            "avg_time_per_test": 0,
            "confusion_matrix": defaultdict(lambda: defaultdict(int)),
            "confidence_by_verdict": defaultdict(list),
            "accuracy_by_category": defaultdict(lambda: {"correct": 0, "total": 0}),
            "stopped_early": False,
            "stop_reason": None
        }
        self.consecutive_rate_limits = 0
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Chạy 1 test case với retry logic"""
        print(f"\n{'='*70}")
        print(f"Test #{test_case['id']}: {test_case['content'][:60]}...")
        print(f"Expected: {test_case['expected_verdict']} | Category: {test_case['category']}")
        
        for attempt in range(self.config.MAX_RETRIES + 1):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.config.API_URL}/api/check",
                    json={
                        "content": test_case["content"],
                        "input_type": test_case["type"],
                        "num_sources": self.config.NUM_SOURCES
                    },
                    timeout=self.config.TIMEOUT
                )
                
                elapsed_time = time.time() - start_time
                
                #  CHECK RATE LIMIT
                if response.status_code == 429 or RateLimitDetector.is_rate_limited("", response.status_code):
                    self.consecutive_rate_limits += 1
                    retry_after = RateLimitDetector.extract_retry_after(response) or self.config.RETRY_DELAY
                    
                    print(f"RATE LIMIT HIT (Attempt {attempt + 1}/{self.config.MAX_RETRIES + 1})")
                    
                    if attempt < self.config.MAX_RETRIES:
                        print(f" Retrying after {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    else:
                        return {
                            "test_id": test_case["id"],
                            "status": "rate_limited",
                            "error": "API rate limit exceeded",
                            "time": elapsed_time,
                            "content": test_case["content"][:100],
                            "category": test_case["category"]
                        }
                
                self.consecutive_rate_limits = 0
                
                if response.status_code != 200:
                    error_text = response.text
                    if RateLimitDetector.is_rate_limited(error_text):
                        self.consecutive_rate_limits += 1
                        return {
                            "test_id": test_case["id"],
                            "status": "rate_limited",
                            "error": f"Rate limit in response: {error_text[:100]}",
                            "time": elapsed_time,
                            "content": test_case["content"][:100],
                            "category": test_case["category"]
                        }
                    
                    return {
                        "test_id": test_case["id"],
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {error_text[:100]}",
                        "time": elapsed_time
                    }
                
                data = response.json()
                
                if not data.get("success"):
                    error_msg = data.get("message", "Unknown error")
                    if RateLimitDetector.is_rate_limited(error_msg):
                        self.consecutive_rate_limits += 1
                        return {
                            "test_id": test_case["id"],
                            "status": "rate_limited",
                            "error": error_msg,
                            "time": elapsed_time,
                            "content": test_case["content"][:100],
                            "category": test_case["category"]
                        }
                    
                    return {
                        "test_id": test_case["id"],
                        "status": "error", 
                        "error": error_msg,
                        "time": elapsed_time
                    }
                
                # SUCCESS
                actual_verdict = data["verdict"]["code"]
                expected_verdict = test_case["expected_verdict"]
                confidence = data["verdict"]["confidence_percentage"]
                is_correct = (actual_verdict == expected_verdict)
                
                print(f"✓ Actual: {actual_verdict} ({confidence}%)")
                print(f"{'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"⏱ Time: {elapsed_time:.2f}s")
                
                return {
                    "test_id": test_case["id"],
                    "content": test_case["content"][:100],
                    "expected": expected_verdict,
                    "actual": actual_verdict,
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "time": elapsed_time,
                    "status": "success",
                    "category": test_case["category"],
                    "num_references": len(data.get("references", [])),
                    "voting_summary": data.get("voting_summary"),
                    "full_response": data
                }
                
            except requests.exceptions.Timeout:
                if attempt < self.config.MAX_RETRIES:
                    print(f"Timeout, retrying... ({attempt + 1}/{self.config.MAX_RETRIES + 1})")
                    continue
                return {
                    "test_id": test_case["id"],
                    "status": "timeout",
                    "error": "Request timeout",
                    "time": self.config.TIMEOUT
                }
            except Exception as e:
                error_msg = str(e)
                if RateLimitDetector.is_rate_limited(error_msg):
                    self.consecutive_rate_limits += 1
                    return {
                        "test_id": test_case["id"],
                        "status": "rate_limited",
                        "error": error_msg,
                        "time": time.time() - start_time,
                        "content": test_case["content"][:100],
                        "category": test_case["category"]
                    }
                
                return {
                    "test_id": test_case["id"],
                    "status": "error",
                    "error": error_msg,
                    "time": time.time() - start_time
                }
        
        return {
            "test_id": test_case["id"],
            "status": "error",
            "error": "Max retries exceeded",
            "time": 0
        }
    
    def should_stop_testing(self) -> tuple[bool, str]:
        """Kiểm tra xem có nên dừng testing không"""
        if self.consecutive_rate_limits >= 3:
            return True, "3 consecutive rate limit errors detected"
        
        recent_results = self.results[-10:] if len(self.results) >= 10 else self.results
        if recent_results:
            rate_limited_count = sum(1 for r in recent_results if r.get("status") == "rate_limited")
            if rate_limited_count / len(recent_results) > 0.5:
                return True, f"High rate limit rate: {rate_limited_count}/{len(recent_results)} recent tests"
        
        return False, ""
    
    def run_all_tests(self, dataset: List[Dict] = None):
        """Chạy toàn bộ test suite với graceful shutdown"""
        if dataset is None:
            dataset = FALLBACK_DATASET
        
        if self.config.SKIP_FIRST_N > 0:
            dataset = dataset[self.config.SKIP_FIRST_N:]
            print(f"Skipped first {self.config.SKIP_FIRST_N} tests")
        
        if self.config.MAX_TESTS:
            dataset = dataset[:self.config.MAX_TESTS]
            print(f"Limited to {self.config.MAX_TESTS} tests")
        
        print(f"\n{'='*70}")
        print(f" STARTING PERFORMANCE TEST SUITE v2.1")
        print(f" Total Test Cases: {len(dataset)}")
        print(f" API Endpoint: {self.config.API_URL}")
        print(f" Rate Limit Protection: ENABLED")
        print(f" Limits: {self.config.RPM_LIMIT} RPM | {self.config.RPD_LIMIT} RPD")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        for idx, test_case in enumerate(dataset, start=1):
            if self.config.STOP_ON_RATE_LIMIT:
                should_stop, reason = self.should_stop_testing()
                if should_stop:
                    print(f"\n{'🛑'*35}")
                    print(f" STOPPING TEST SUITE EARLY")
                    print(f" Reason: {reason}")
                    print(f" Completed: {idx-1}/{len(dataset)} tests")
                    print(f"{'🛑'*35}\n")
                    
                    self.metrics["stopped_early"] = True
                    self.metrics["stop_reason"] = reason
                    break
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # Update metrics
            self.metrics["total_tests"] += 1
            
            if result["status"] == "rate_limited":
                self.metrics["rate_limit_errors"] += 1
                self.metrics["errors"] += 1
            elif result["status"] == "success":
                category = result["category"]
                
                if result["is_correct"]:
                    self.metrics["correct_predictions"] += 1
                    self.metrics["accuracy_by_category"][category]["correct"] += 1
                else:
                    self.metrics["incorrect_predictions"] += 1
                
                self.metrics["accuracy_by_category"][category]["total"] += 1
                self.metrics["confusion_matrix"][result["expected"]][result["actual"]] += 1
                self.metrics["confidence_by_verdict"][result["actual"]].append(result["confidence"])
            else:
                self.metrics["errors"] += 1
            
            self.metrics["total_time"] += result["time"]
            
            # Delay between tests
            if idx < len(dataset) and self.config.DELAY_BETWEEN_TESTS > 0:
                print(f"Waiting {self.config.DELAY_BETWEEN_TESTS}s before next test...")
                time.sleep(self.config.DELAY_BETWEEN_TESTS)
        
        # Calculate final metrics
        if self.metrics["total_tests"] > 0:
            self.metrics["avg_time_per_test"] = self.metrics["total_time"] / self.metrics["total_tests"]
            
            successful_tests = self.metrics["correct_predictions"] + self.metrics["incorrect_predictions"]
            if successful_tests > 0:
                self.metrics["accuracy"] = (self.metrics["correct_predictions"] / successful_tests) * 100
            else:
                self.metrics["accuracy"] = 0
        
        print(f"\n{'='*70}")
        print(f" TEST SUITE COMPLETED")
        print(f" Total Time: {time.time() - start_time:.2f}s")
        print(f"{'='*70}\n")
        
        return self.results, self.metrics
    
    def print_summary(self):
        """In tóm tắt kết quả với rate limit info"""
        print("\n" + "="*70)
        print(" PERFORMANCE TEST SUMMARY v2.1")
        print("="*70)
        
        if self.metrics.get("stopped_early"):
            print(f"\nTEST STOPPED EARLY")
            print(f"   Reason: {self.metrics['stop_reason']}")
        
        print(f"\nOVERALL METRICS:")
        print(f"  • Total Tests: {self.metrics['total_tests']}")
        print(f"  • Correct: {self.metrics['correct_predictions']} ✅")
        print(f"  • Incorrect: {self.metrics['incorrect_predictions']} ❌")
        print(f"  • Errors: {self.metrics['errors']} ⚠️")
        print(f"  • Rate Limit Errors: {self.metrics['rate_limit_errors']} 🚫")
        
        successful_tests = self.metrics['correct_predictions'] + self.metrics['incorrect_predictions']
        if successful_tests > 0:
            print(f"  • Accuracy: {self.metrics.get('accuracy', 0):.2f}% (on {successful_tests} successful tests)")
        else:
            print(f"  • Accuracy: N/A (no successful tests)")
        
        print(f"  • Avg Time/Test: {self.metrics['avg_time_per_test']:.2f}s")
        
        # Accuracy by Category
        if self.metrics['accuracy_by_category']:
            print(f"\n📂 ACCURACY BY CATEGORY:")
            for category, stats in sorted(self.metrics['accuracy_by_category'].items()):
                if stats['total'] > 0:
                    cat_accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"  • {category.upper()}: {cat_accuracy:.1f}% ({stats['correct']}/{stats['total']})")
        
        # Confusion Matrix
        if any(self.metrics['confusion_matrix'].values()):
            print(f"\n🎯 CONFUSION MATRIX:")
            verdicts = ["LIKELY_TRUE", "LIKELY_FALSE", "UNCERTAIN"]
            
            print(f"\n{'Actual →':<15}", end="")
            for v in verdicts:
                print(f"{v:<15}", end="")
            print(f"\n{'Expected ↓':<15}", end="")
            print("-" * 60)
            
            for expected in verdicts:
                print(f"{expected:<15}", end="")
                for actual in verdicts:
                    count = self.metrics['confusion_matrix'][expected][actual]
                    print(f"{count:<15}", end="")
                print()
        
        # Confidence Analysis
        if self.metrics['confidence_by_verdict']:
            print(f"\n💪 CONFIDENCE ANALYSIS:")
            for verdict, confidences in self.metrics['confidence_by_verdict'].items():
                if confidences:
                    avg_conf = statistics.mean(confidences)
                    min_conf = min(confidences)
                    max_conf = max(confidences)
                    std_dev = statistics.stdev(confidences) if len(confidences) > 1 else 0
                    print(f"  • {verdict}:")
                    print(f"    - Average: {avg_conf:.1f}%")
                    print(f"    - Range: {min_conf:.1f}% - {max_conf:.1f}%")
                    print(f"    - Std Dev: {std_dev:.1f}%")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self):
        """Lưu kết quả với rate limit info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        confusion_matrix_dict = {
            k: dict(v) for k, v in self.metrics["confusion_matrix"].items()
        }
        
        accuracy_by_category_dict = {
            k: dict(v) for k, v in self.metrics["accuracy_by_category"].items()
        }
        
        # Detailed results JSON
        results_file = self.config.OUTPUT_DIR / f"test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "stopped_early": self.metrics.get("stopped_early", False),
                "stop_reason": self.metrics.get("stop_reason"),
                "results": self.results,
                "metrics": {
                    "total_tests": self.metrics["total_tests"],
                    "correct_predictions": self.metrics["correct_predictions"],
                    "incorrect_predictions": self.metrics["incorrect_predictions"],
                    "errors": self.metrics["errors"],
                    "rate_limit_errors": self.metrics["rate_limit_errors"],
                    "accuracy": self.metrics.get("accuracy", 0),
                    "avg_time_per_test": self.metrics["avg_time_per_test"],
                    "confusion_matrix": confusion_matrix_dict,
                    "confidence_by_verdict": dict(self.metrics["confidence_by_verdict"]),
                    "accuracy_by_category": accuracy_by_category_dict
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
        
        # CSV Summary
        csv_file = self.config.OUTPUT_DIR / f"test_summary_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["TestID", "Content", "Category", "Expected", "Actual", "Confidence", "Correct", "Time", "Status"])
            
            for r in self.results:
                if r["status"] == "success":
                    writer.writerow([
                        r['test_id'],
                        r['content'][:100],
                        r['category'],
                        r['expected'],
                        r['actual'],
                        r['confidence'],
                        r['is_correct'],
                        f"{r['time']:.2f}",
                        r['status']
                    ])
                else:
                    writer.writerow([
                        r['test_id'],
                        r.get('content', 'N/A')[:100],
                        r.get('category', 'N/A'),
                        'N/A',
                        'N/A',
                        'N/A',
                        False,
                        f"{r['time']:.2f}",
                        r['status']
                    ])
        
        print(f"✓ CSV saved to: {csv_file}")
        
        return results_file


# ============================
# 6. MAIN EXECUTION
# ============================

def main():
    """Main function với rate limit handling"""
    
    config = TestConfig()
    tester = PerformanceTester(config)
    
    print("\n" + "🚀" * 35)
    print(" FACT-CHECKING SYSTEM PERFORMANCE TEST v2.1")
    print(" RATE LIMIT PROTECTION ENABLED")
    print("🚀" * 35 + "\n")
    
    # API Health Check
    try:
        response = requests.get(f"{config.API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding. Please start the server first.")
            return
        print("✓ API Health Check: OK\n")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"Please make sure the server is running at {config.API_URL}\n")
        return
    
    # Load dataset
    dataset = None
    
    if config.USE_CSV:
        try:
            loader = CSVDataLoader(config.CSV_PATH)
            dataset = loader.load_dataset()
            print(f"✓ Using CSV dataset: {config.CSV_PATH}\n")
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print(f"⚠️  Falling back to hardcoded dataset\n")
            dataset = FALLBACK_DATASET
    else:
        dataset = FALLBACK_DATASET
        print(f"✓ Using hardcoded dataset\n")
    
    if not dataset:
        print("❌ No dataset available!")
        return
    
    # Run test suite
    results, metrics = tester.run_all_tests(dataset)
    
    # Print summary
    tester.print_summary()
    
    # Save results
    results_file = tester.save_results()
    
    # Final evaluation
    print("\n📝 EVALUATION:")
    
    if metrics.get("stopped_early"):
        print(f"⚠️  Testing was stopped early due to rate limits")
        print(f"   Completed: {metrics['total_tests']} tests")
        print(f"   Rate limit errors: {metrics['rate_limit_errors']}")
    
    successful_tests = metrics['correct_predictions'] + metrics['incorrect_predictions']
    
    if successful_tests > 0:
        accuracy = metrics.get('accuracy', 0)
        
        if accuracy >= 90:
            print(f"🌟 EXCELLENT: System accuracy is {accuracy:.1f}% ({successful_tests} successful tests)")
        elif accuracy >= 80:
            print(f"✓ VERY GOOD: System accuracy is {accuracy:.1f}% ({successful_tests} successful tests)")
        elif accuracy >= 70:
            print(f"✓ GOOD: System accuracy is {accuracy:.1f}% ({successful_tests} successful tests)")
        elif accuracy >= 60:
            print(f"⚠️  ACCEPTABLE: System accuracy is {accuracy:.1f}% ({successful_tests} successful tests)")
        else:
            print(f"❌ NEEDS IMPROVEMENT: System accuracy is {accuracy:.1f}% ({successful_tests} successful tests)")
    else:
        print(f"❌ No successful tests completed (all failed or rate limited)")
    
    print("\n" + "="*70)
    print(f" TEST COMPLETED")
    print(f" Results saved to: {results_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()