import requests
from urllib.parse import quote_plus, urljoin, urlparse
import time
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import hashlib
import random
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_in_seconds=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while x < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries - 1:
                        logger.error(f"❌ Failed after {retries} retries: {e}")
                        raise
                    wait = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                    logger.warning(f"⚠️ Retry {x+1}/{retries} after {wait:.1f}s: {str(e)[:100]}")
                    time.sleep(wait)
                    x += 1
        return wrapper
    return decorator


class UserAgentRotator:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def get_random(self):
        return random.choice(self.user_agents)


class SmartCache:
    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List]:
        key = self._get_key(query)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"💾 Cache HIT: {query[:50]}")
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, data: List):
        key = self._get_key(query)
        self.cache[key] = (data, datetime.now())


class WebSearcher:
    
    def __init__(self, 
                 google_api_key: str = None,
                 google_cse_id: str = None,
                 cache_enabled: bool = True):
        
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        if self.google_api_key and self.google_cse_id:
            logger.info(f"✅ Google API configured: Key={self.google_api_key[:10]}..., CSE={self.google_cse_id}")
        else:
            logger.warning("⚠️ Google API NOT configured - will use fallback methods")
        
        self.trusted_sources = {
            'vnexpress.net': {
                'name': 'VnExpress',
                'search_url': 'https://timkiem.vnexpress.net/?q={query}',
                'weight': 1.0,
                'selectors': {
                    'container': 'article.item-news',
                    'link': 'h3.title-news a, h2.title-news a',
                    'title': 'h3.title-news, h2.title-news',
                    'snippet': 'p.description'
                }
            },
            'tuoitre.vn': {
                'name': 'Tuổi Trẻ',
                'search_url': 'https://tuoitre.vn/tim-kiem.htm?keywords={query}',
                'weight': 1.0,
                'selectors': {
                    'container': 'li.news-item',
                    'link': 'h3 a, h2 a',
                    'title': 'h3, h2',
                    'snippet': 'p.sapo, div.sapo'
                }
            },
            'thanhnien.vn': {
                'name': 'Thanh Niên',
                'search_url': 'https://thanhnien.vn/tim-kiem/?keywords={query}',
                'weight': 0.95,
                'selectors': {
                    'container': 'article.story',
                    'link': 'h2.story__heading a, h3.story__heading a',
                    'title': 'h2.story__heading, h3.story__heading',
                    'snippet': 'p.story__summary'
                }
            },
            'dantri.com.vn': {
                'name': 'Dân Trí',
                'search_url': 'https://dantri.com.vn/tim-kiem.htm?q={query}',
                'weight': 0.95,
                'selectors': {
                    'container': 'article',
                    'link': 'h3 a, h2 a',
                    'title': 'h3, h2',
                    'snippet': 'div.article-excerpt, p'
                }
            },
            'vietnamnet.vn': {
                'name': 'VietnamNet',
                'search_url': 'https://vietnamnet.vn/tim-kiem?q={query}',
                'weight': 0.95,
                'selectors': {
                    'container': 'article, div.item-news',
                    'link': 'h3 a, h2 a',
                    'title': 'h3, h2',
                    'snippet': 'p.description'
                }
            }
        }
        
        self.ua_rotator = UserAgentRotator()
        self.cache = SmartCache() if cache_enabled else None
        
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def get_headers(self):
        return {
            'User-Agent': self.ua_rotator.get_random(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        }
    
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def fetch_url(self, url: str, timeout: int = 15) -> requests.Response:
        response = self.session.get(
            url, 
            headers=self.get_headers(), 
            timeout=timeout,
            allow_redirects=True,
            verify=True
        )
        response.raise_for_status()
        return response
    
    def extract_numbers_from_text(self, text: str) -> List[str]:
        patterns = [
            r'\d+[\.,]\d+[\.,]\d+',  
            r'\d+\s*(?:triệu|tỷ|nghìn|ngàn)',  
            r'\d+\s*%',               
            r'\d{4,}',                
            r'\d+[\.,]\d+',           
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        unique_numbers = []
        seen = set()
        for num in numbers:
            normalized = num.lower().replace(' ', '')
            if normalized not in seen:
                seen.add(normalized)
                unique_numbers.append(num)
        
        return unique_numbers[:3]  
    
    def build_smart_queries(self, keywords: List[str], title: str = "", full_text: str = "") -> List[str]:
        
        queries = []
        
        stopwords = {
            # Từ gốc
            'của', 'và', 'có', 'được', 'là', 'cho', 'với', 'tại', 'từ', 
            'này', 'đó', 'đã', 'sẽ', 'trong', 'một', 'các', 'những', 'chỉ', 'khắp', 'khỏi',
            
            # BỔ SUNG - Từ phổ biến bị thiếu
            'không', 'người', 'dân', 'phải', 'bị', 'theo', 'về', 'hay', 
            'khi', 'nếu', 'như', 'đến', 'đang', 'còn', 'đều', 'cả',
            'sau', 'trước', 'giữa', 'ngoài', 'trên', 'dưới', 'ngày',
            
            # Từ nối & giới từ
            'hoặc', 'nhưng', 'mà', 'nên', 'thì', 'rằng', 'vì', 'do',
            
            # Đại từ & số lượng
            'nào', 'gì', 'ai', 'đâu', 'bao', 'lúc', 'nơi', 'việc', 'chúng',
            'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
            
            # Từ thường gặp trong tin tức
            'cho biết', 'theo đó', 'tại đây', 'như vậy', 'bởi vì',
            'đồng thời', 'ngoài ra', 'tuy nhiên', 'do đó', 'vì vậy',
            
            # Động từ phổ biến (ít ý nghĩa)
            'nói', 'biết', 'thấy', 'thể', 'cần', 'làm', 'đi'
        }
        
        numbers = self.extract_numbers_from_text(title + " " + full_text)
        
        if title and numbers:
            title_words = title.split()
            important_words = [
                w for w in title_words 
                if w.lower() not in stopwords and len(w) > 2 and not w.isdigit()
            ][:4]
            
            query_parts = important_words + numbers[:2]
            if len(query_parts) >= 3:
                queries.append(' '.join(query_parts))
        
        if title:
            title_words = title.split()
            important_words = [
                w for w in title_words 
                if w.lower() not in stopwords and len(w) > 3 and not w.isdigit()
            ][:5]
            if len(important_words) >= 3:
                queries.append(' '.join(important_words))
        
        if keywords and numbers:
            filtered_keywords = [
                k for k in keywords 
                if len(k) > 2 and k.lower() not in stopwords and not k.isdigit()
            ][:4]
            if filtered_keywords:
                query_parts = filtered_keywords + numbers[:1]
                queries.append(' '.join(query_parts))
        
        if not queries and keywords:
            filtered_keywords = [
                k for k in keywords 
                if len(k) > 2 and k.lower() not in stopwords and not k.isdigit()
            ][:5]
            if filtered_keywords:
                queries.append(' '.join(filtered_keywords))
        
        unique_queries = []
        seen = set()
        for q in queries:
            q = q.strip()
            has_number = any(char.isdigit() for char in q)
            min_length = 10 if has_number else 15
            
            if len(q) > min_length and q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        if not unique_queries and (title or keywords):
            emergency_words = []
            if title:
                emergency_words = [w for w in title.split() if len(w) > 3][:3]
            if not emergency_words and keywords:
                emergency_words = [k for k in keywords if len(k) > 3][:3]
            
            if emergency_words:
                unique_queries.append(' '.join(emergency_words))
        
        return unique_queries[:3]  
    
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def search_on_source_advanced(self, source_domain: str, query: str, 
                                   max_results: int = 5) -> List[Dict]:
        
        source_info = self.trusted_sources.get(source_domain)
        if not source_info:
            return []
        
        results = []
        
        try:
            search_url = source_info['search_url'].format(query=quote_plus(query))
            logger.info(f"🔍 Searching {source_domain}: {search_url}")
            
            response = self.fetch_url(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selectors = source_info.get('selectors', {})
            
            if selectors.get('container'):
                containers = soup.select(selectors['container'])
                logger.info(f"  Found {len(containers)} containers")
                
                for container in containers[:max_results * 2]:
                    link_elem = container.select_one(selectors.get('link', 'a'))
                    if not link_elem:
                        continue
                    
                    href = link_elem.get('href')
                    if not href:
                        continue
                    
                    if href.startswith('/'):
                        href = urljoin(f"https://{source_domain}", href)
                    
                    if source_domain not in href:
                        continue
                    
                    skip_patterns = [
                        '/tim-kiem', '/search', '/tag/', '/category/', 
                        '/chuyen-muc/', '/video/', '/photo/', '/folder/'
                    ]
                    if any(pattern in href for pattern in skip_patterns):
                        continue
                    
                    if any(r['url'] == href for r in results):
                        continue
                    
                    title = ""
                    if selectors.get('title'):
                        title_elem = container.select_one(selectors['title'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                    if not title:
                        title = link_elem.get_text(strip=True)
                    
                    snippet = ""
                    if selectors.get('snippet'):
                        snippet_elem = container.select_one(selectors['snippet'])
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)[:300]
                    
                    if not title or len(title) < 20:
                        continue
                    
                    results.append({
                        'url': href,
                        'title': title,
                        'snippet': snippet,
                        'source': source_info['name'],
                        'domain': source_domain
                    })
                    
                    if len(results) >= max_results:
                        break
            
            if len(results) < max_results:
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    href = link['href']
                    
                    if href.startswith('/'):
                        href = urljoin(f"https://{source_domain}", href)
                    
                    if source_domain not in href:
                        continue
                    
                    skip_patterns = [
                        '/tim-kiem', '/search', '/tag/', '/category/', 
                        '/chuyen-muc/', '/video/', '/photo/', '/folder/',
                        '.jpg', '.png', '.gif', '.pdf'
                    ]
                    if any(pattern in href.lower() for pattern in skip_patterns):
                        continue
                    
                    if any(r['url'] == href for r in results):
                        continue
                    
                    title = link.get_text(strip=True)
                    if not title or len(title) < 20:
                        title = link.get('title', '')
                    
                    if not title or len(title) < 20:
                        continue
                    
                    snippet = ""
                    parent = link.find_parent(['article', 'div', 'li'])
                    if parent:
                        # Tìm description/sapo trong parent
                        desc_elem = parent.find(['p', 'div'], class_=re.compile('desc|sapo|summary|excerpt', re.I))
                        if desc_elem:
                            snippet = desc_elem.get_text(strip=True)[:300]
                    
                    results.append({
                        'url': href,
                        'title': title,
                        'snippet': snippet,
                        'source': source_info['name'],
                        'domain': source_domain
                    })
                    
                    if len(results) >= max_results:
                        break
            
            logger.info(f"  ✅ {source_domain}: Found {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"  ❌ {source_domain}: {str(e)[:100]}")
            return []
    
    def search_google_custom_api(self, query: str, num_results: int = 10) -> List[Dict]:
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("⚠️ Google API credentials not configured")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            
            site_filter = ' OR '.join([f'site:{d}' for d in list(self.trusted_sources.keys())[:5]])
            full_query = f"{query} ({site_filter})"
            
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': full_query,
                'num': min(10, num_results),
                'lr': 'lang_vi',
                'gl': 'vn'
            }
            
            logger.info(f"🔍 Google Custom Search API: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    link = item.get('link', '')
                    
                    if not any(domain in link for domain in self.trusted_sources.keys()):
                        continue
                    
                    results.append({
                        'url': link,
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Google Search',
                        'domain': urlparse(link).netloc
                    })
            
            logger.info(f"  ✅ Google API: Found {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"  ❌ Google API error: {str(e)[:100]}")
            return []
    
    def search_google_scraping(self, query: str, num_results: int = 10) -> List[Dict]:
        
        try:
            # Thêm site filter
            site_filter = ' OR '.join([f'site:{d}' for d in list(self.trusted_sources.keys())[:5]])
            full_query = f"{query} ({site_filter})"
            
            search_url = f"https://www.google.com/search?q={quote_plus(full_query)}&num={num_results}&hl=vi&gl=vn"
            
            logger.info(f"🔍 Google Scraping: {query}")
            response = self.fetch_url(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            search_divs = soup.find_all('div', class_='g')
            
            for div in search_divs:

                link_elem = div.find('a', href=True)
                if not link_elem:
                    continue
                
                href = link_elem['href']
                
                if '/url?q=' in href:
                    url = href.split('/url?q=')[1].split('&')[0]
                else:
                    url = href
                
                if not any(domain in url for domain in self.trusted_sources.keys()):
                    continue
                
                title_elem = div.find('h3')
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                snippet = ''
                snippet_divs = div.find_all(['div', 'span'], class_=re.compile('VwiC3b|s3v9rd'))
                if snippet_divs:
                    snippet = snippet_divs[0].get_text(strip=True)[:300]
                
                if not title:
                    continue
                
                results.append({
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'source': 'Google Search',
                    'domain': urlparse(url).netloc
                })
            
            logger.info(f"  ✅ Google Scraping: Found {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"  ❌ Google Scraping error: {str(e)[:100]}")
            return []
    
    def search_parallel(self, query: str, sources: List[str], max_workers: int = 5) -> List[Dict]:
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.search_on_source_advanced, source, query, 5): source 
                for source in sources
            }
            
            for future in as_completed(futures, timeout=30):
                try:
                    results = future.result(timeout=10)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"  ❌ Parallel search error: {str(e)[:100]}")
        
        return all_results
    
    def search_for_fact_check(self, processed_data: Dict, num_results: int = 10) -> List[Dict]:
        keywords = processed_data.get('keywords', [])
        title = processed_data.get('title', '')
        full_text = processed_data.get('full_text', '')
        
        if not keywords and not title:
            logger.error("❌ No keywords or title provided")
            return []
        
        queries = self.build_smart_queries(keywords, title, full_text)
        
        if not queries:
            logger.error("❌ No valid queries generated")
            return []
        
        logger.info("="*70)
        logger.info("🚀 SEARCH STARTED")
        logger.info("="*70)
        logger.info(f"Keywords: {keywords[:10]}")
        logger.info(f"Generated {len(queries)} query variants:")
        for i, q in enumerate(queries, 1):
            logger.info(f"  {i}. {q}")
        logger.info("="*70)
        
        all_results = []
        
        for query_idx, query in enumerate(queries, 1):
            logger.info(f"\n🔍 Query {query_idx}/{len(queries)}: '{query}'")
            
            if self.cache:
                cached = self.cache.get(query)
                if cached:
                    all_results.extend(cached)
                    continue
            
            query_results = []
            
            if self.google_api_key and self.google_cse_id:
                logger.info("  [Strategy 1] Google Custom Search API...")
                api_results = self.search_google_custom_api(query, num_results)
                query_results.extend(api_results)
            
            if len(query_results) < num_results:
                logger.info("  [Strategy 2] Parallel search on news sites...")
                priority_sources = ['vnexpress.net', 'tuoitre.vn', 'thanhnien.vn', 
                                  'dantri.com.vn', 'vietnamnet.vn']
                parallel_results = self.search_parallel(query, priority_sources, max_workers=5)
                query_results.extend(parallel_results)
            
            if len(query_results) < num_results:
                logger.info("  [Strategy 3] Google scraping fallback...")
                google_results = self.search_google_scraping(query, num_results)
                query_results.extend(google_results)
            
            if self.cache and query_results:
                self.cache.set(query, query_results)
            
            all_results.extend(query_results)
            
            time.sleep(1)
            
            if len(all_results) >= num_results * 2:
                break
        
        unique_results = {}
        for result in all_results:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
        
        final_results = list(unique_results.values())[:num_results]
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ SEARCH COMPLETED")
        logger.info(f"Total found: {len(all_results)} → Unique: {len(unique_results)} → Final: {len(final_results)}")
        logger.info("="*70)
        
        for i, result in enumerate(final_results, 1):
            logger.info(f"{i}. [{result['domain']}] {result['title'][:80]}")
            logger.info(f"   {result['url']}")
        
        logger.info("="*70)
        
        return final_results



if __name__ == "__main__":
    pass