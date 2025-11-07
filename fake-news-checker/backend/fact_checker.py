from preprocessor import TextPreprocessor
from similarity_checker import SimilarityChecker
import json
from datetime import datetime
import logging

try:
    from web_searcher import WebSearcher
except ImportError:
    from web_searcher import WebSearcher

try:
    from config import Config
except ImportError:
    class Config:
        GOOGLE_API_KEY = None
        GOOGLE_CSE_ID = None
        NEWS_API_KEY = None
        ENABLE_CACHE = True
        DEFAULT_NUM_RESULTS = 5

logger = logging.getLogger(__name__)


class FactChecker:
    
    def __init__(self, 
                 google_api_key=None, 
                 google_cse_id=None,
                 news_api_key=None):
        logger.info("="*70)
        logger.info(" Initializing Fact Checker...")
        logger.info("="*70)
        
        # Use provided keys or fallback to config
        api_key = google_api_key or Config.GOOGLE_API_KEY
        cse_id = google_cse_id or Config.GOOGLE_CSE_ID
        news_key = news_api_key or Config.NEWS_API_KEY
        
        self.preprocessor = TextPreprocessor()
        logger.info("Preprocessor initialized")
        
        self.searcher = WebSearcher(
            google_api_key=api_key,
            google_cse_id=cse_id,
            cache_enabled=Config.ENABLE_CACHE
        )
        logger.info(" Web Searcher initialized")
        
        self.similarity_checker = SimilarityChecker()
        logger.info("Similarity Checker initialized")
        
        logger.info("="*70)
        logger.info(" Fact Checker ready!")
        logger.info("="*70 + "\n")
    
    def check_fact(self, user_input, input_type='text', num_sources=None):
        if num_sources is None:
            num_sources = Config.DEFAULT_NUM_RESULTS
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_type': input_type,
            'original_input': user_input,
            'status': 'processing'
        }
        
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 1: PREPROCESSING")
            logger.info("="*70)
            
            processed = self.preprocessor.process_input(user_input, input_type)
            
            if not processed:
                results['status'] = 'error'
                results['error'] = 'Khong the xu ly input'
                logger.error("Failed to preprocess input")
                return results
            
            results['processed_data'] = {
                'title': processed['title'],
                'keywords': processed['keywords'],
                'domain': processed['domain']
            }
            
            logger.info(f" Extracted {len(processed['keywords'])} keywords")
            logger.info(f"   Keywords: {processed['keywords'][:10]}")
            
            logger.info("\n" + "="*70)
            logger.info("STEP 2: SEARCHING FOR REFERENCE ARTICLES")
            logger.info("="*70)
            
            reference_articles = self.searcher.search_for_fact_check(processed, num_sources)
            
            if not reference_articles:
                results['status'] = 'no_references'
                results['message'] = 'Khong tim thay bai bao tham khao tu nguon uy tin'
                logger.warning("No reference articles found")
                return results
            
            results['reference_articles'] = reference_articles
            logger.info(f"Found {len(reference_articles)} reference articles")
            
            logger.info("\n" + "="*70)
            logger.info("STEP 3: CRAWLING REFERENCE ARTICLES")
            logger.info("="*70)
            
            reference_contents = []
            
            for i, article in enumerate(reference_articles, 1):
                url = article['url']
                logger.info(f"Crawling {i}/{len(reference_articles)}: {article['domain']}")
                
                content = self.preprocessor.extract_from_url(url)
                
                if content and content['content']:
                    reference_contents.append({
                        'url': url,
                        'title': content['title'] or article['title'],
                        'content': content['content'],
                        'domain': content['domain'],
                        'snippet': article.get('snippet', ''),
                        'source': article.get('source', '')
                    })
                    logger.info(f"Success ({len(content['content'])} chars)")
                else:
                    logger.warning(f"Failed to crawl")
            
            if not reference_contents:
                results['status'] = 'crawl_failed'
                results['message'] = 'Khong the crawl noi dung tu cac bai bao tham khao'
                logger.error("All crawl attempts failed")
                return results
            
            logger.info(f"Successfully crawled {len(reference_contents)}/{len(reference_articles)} articles")
            
            logger.info("\n" + "="*70)
            logger.info("STEP 4: CALCULATING SIMILARITY")
            logger.info("="*70)
            
            similarity_results = []
            
            for ref in reference_contents:
                # TÃ­nh similarity tá»•ng thá»ƒ
                overall_sim = self.similarity_checker.calculate_similarity(
                    processed['full_text'],
                    ref['content']
                )
                
                # TÃ­nh detailed similarity
                detailed_sim = self.similarity_checker.calculate_detailed_similarity(
                    processed['full_text'],
                    ref['content']
                )
                
                similarity_results.append({
                    'url': ref['url'],
                    'title': ref['title'],
                    'domain': ref['domain'],
                    'source': ref.get('source', ''),
                    'overall_similarity': overall_sim,
                    'detailed_similarity': detailed_sim
                })
                
                logger.info(f"{ref['domain']}: {overall_sim:.2%}")
            
            
            similarity_results.sort(key=lambda x: x['overall_similarity'], reverse=True)
            
            logger.info("\n" + "="*70)
            logger.info("STEP 5: GENERATING VERDICT")
            logger.info("="*70)
            
            top_results = similarity_results[:min(3, len(similarity_results))]

            highest_similarity = top_results[0]['overall_similarity'] if top_results else 0

            verdict = self.similarity_checker.generate_verdict(highest_similarity)

            avg_similarity = highest_similarity
            
            
            results['status'] = 'success'
            results['verdict'] = verdict
            results['average_similarity'] = avg_similarity
            results['similarity_details'] = similarity_results
            results['top_references'] = [
                {
                    'url': r['url'],
                    'title': r['title'],
                    'domain': r['domain'],
                    'source': r.get('source', ''),
                    'similarity': r['overall_similarity']
                }
                for r in top_results
            ]
            
            logger.info("\n" + "="*70)
            logger.info("FINAL VERDICT")
            logger.info("="*70)
            logger.info(f"Label: {verdict['label']}")
            logger.info(f"Verdict: {verdict['verdict']}")
            logger.info(f"Average Similarity: {avg_similarity:.2%}")
            logger.info(f"Confidence: {verdict['confidence']:.2%}")
            logger.info(f"Color: {verdict['color']}")
            logger.info("="*70 + "\n")
            
            return results
            
        except Exception as e:
            logger.error(f"âŒ Error in fact checking: {str(e)}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def format_result_for_frontend(self, results):
        if results['status'] != 'success':
            return {
                'success': False,
                'message': results.get('message', results.get('error', 'Unknown error'))
            }
        
        return {
            'success': True,
            'verdict': {
                'label': results['verdict']['label'],
                'code': results['verdict']['verdict'],
                'explanation': results['verdict']['explanation'],
                'color': results['verdict']['color'],
                'similarity_percentage': round(results['average_similarity'] * 100, 2),
                'confidence_percentage': round(results['verdict']['confidence'] * 100, 2)
            },
            'references': [
                {
                    'title': ref['title'],
                    'url': ref['url'],
                    'domain': ref['domain'],
                    'source': ref.get('source', ''),
                    'similarity_percentage': round(ref['similarity'] * 100, 2)
                }
                for ref in results['top_references']
            ],
            'keywords': results['processed_data']['keywords'],
            'timestamp': results['timestamp']
        }

if __name__ == "__main__":
    pass