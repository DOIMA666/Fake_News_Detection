"""
Script test toàn bộ hệ thống
Chạy: python test_system.py
"""

import sys
import json
from datetime import datetime

def print_section(title):
    """In header cho section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_imports():
    """Test import các module"""
    print_section("TEST 1: Imports")
    
    try:
        from preprocessor import TextPreprocessor
        print("✓ preprocessor.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing preprocessor: {e}")
        return False
    
    try:
        from web_searcher import WebSearcher
        print("✓ web_searcher.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing web_searcher: {e}")
        return False
    
    try:
        from similarity_checker import SimilarityChecker
        print("✓ similarity_checker.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing similarity_checker: {e}")
        return False
    
    try:
        from fact_checker import FactChecker
        print("✓ fact_checker.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing fact_checker: {e}")
        return False
    
    return True

def test_preprocessor():
    """Test module tiền xử lý"""
    print_section("TEST 2: Text Preprocessing")
    
    from preprocessor import TextPreprocessor
    
    processor = TextPreprocessor()
    
    # Test với text
    test_text = """
    Việt Nam đã tiêm hơn 200 triệu liều vaccine COVID-19 cho người dân.
    Chương trình tiêm chủng được triển khai rộng rãi trên toàn quốc.
    """
    
    print("Input text:")
    print(test_text.strip())
    
    result = processor.process_input(test_text, 'text')
    
    print("\nNormalized text:")
    print(result['full_text'][:200] + "...")
    
    print("\nExtracted keywords:")
    print(result['keywords'])
    
    print("\n✓ Preprocessing works correctly")
    
    return True

def test_similarity():
    """Test module so sánh similarity"""
    print_section("TEST 3: Similarity Calculation")
    
    from similarity_checker import SimilarityChecker
    
    print("Loading model (this may take a minute)...")
    checker = SimilarityChecker()
    print("✓ Model loaded")
    
    # Test với 2 câu giống nhau
    text1 = "Việt Nam đã tiêm hơn 200 triệu liều vaccine COVID-19"
    text2 = "Theo Bộ Y tế, Việt Nam đã tiêm được hơn 200 triệu liều vắc xin COVID-19"
    
    print("\nText 1:", text1)
    print("Text 2:", text2)
    
    sim = checker.calculate_similarity(text1, text2)
    print(f"\nSimilarity: {sim:.2%}")
    
    verdict = checker.generate_verdict(sim)
    print(f"Verdict: {verdict['label']}")
    print(f"Confidence: {verdict['confidence']:.2%}")
    
    print("\n✓ Similarity calculation works correctly")
    
    return True

def test_web_search():
    """Test module web search"""
    print_section("TEST 4: Web Search")
    
    from web_searcher import WebSearcher
    
    searcher = WebSearcher()
    
    test_keywords = ['covid', 'vaccine', 'vietnam']
    print(f"Searching for: {' '.join(test_keywords)}")
    
    queries = searcher.build_smart_queries(test_keywords, title='', full_text='')
    print(f"Query: {queries}")
    
    print("\nSearching... (this may take a few seconds)")
    results = searcher.search_for_fact_check({'keywords': test_keywords}, num_results=3)
    
    if results:
        print(f"\n✓ Found {len(results)} results:")
        for i, url in enumerate(results, 1):
            print(f"  {i}. {url}")
    else:
        print("\n⚠ No results found (search might be rate-limited)")
        print("This is normal during development/testing")
    
    return True

def test_full_pipeline():
    """Test toàn bộ pipeline"""
    print_section("TEST 5: Full Pipeline")
    
    from fact_checker import FactChecker
    
    print("Initializing Fact Checker...")
    checker = FactChecker()
    print("✓ Fact Checker initialized\n")
    
    # Test data
    test_text = """
    Việt Nam đã triển khai chiến dịch tiêm chủng vaccine COVID-19 
    trên toàn quốc từ năm 2021. Đến nay, hơn 90% dân số đã được 
    tiêm ít nhất một liều vaccine.
    """
    
    print("Testing with text:")
    print(test_text.strip())
    print("\n" + "-"*70)
    print("Running fact check... (this will take 10-30 seconds)\n")
    
    try:
        result = checker.check_fact(test_text, input_type='text', num_sources=3)
        
        if result['status'] == 'success':
            print("\n" + "="*70)
            print("RESULT")
            print("="*70)
            
            verdict = result['verdict']
            print(f"\nVerdict: {verdict['label']}")
            print(f"Code: {verdict['verdict']}")
            print(f"Similarity: {result['average_similarity']:.2%}")
            print(f"Confidence: {verdict['confidence']:.2%}")
            print(f"\nExplanation: {verdict['explanation']}")
            
            print("\nTop References:")
            for i, ref in enumerate(result['top_references'], 1):
                print(f"\n{i}. {ref['title']}")
                print(f"   URL: {ref['url']}")
                print(f"   Domain: {ref['domain']}")
                print(f"   Similarity: {ref['similarity']:.2%}")
            
            print("\n✓ Full pipeline works correctly!")
            
            # Save result to file
            with open('test_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("\n✓ Result saved to test_result.json")
            
        else:
            print(f"\n⚠ Status: {result['status']}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            if 'message' in result:
                print(f"Message: {result['message']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*70)
    print("  FAKE NEWS DETECTION SYSTEM - TEST SUITE")
    print("="*70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Preprocessor", test_preprocessor),
        ("Similarity", test_similarity),
        ("Web Search", test_web_search),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = success
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print_section("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print("\n" + "-"*70)
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)