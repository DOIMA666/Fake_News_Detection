import re
import unicodedata
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from collections import Counter
import logging
from typing import List, Dict, Optional, Tuple

# PhoBERT & NLP imports
try:
    from transformers import AutoTokenizer, AutoModel
    from keybert import KeyBERT
    from underthesea import sent_tokenize, pos_tag, ner
    import torch
    PHOBERT_AVAILABLE = True
except ImportError:
    PHOBERT_AVAILABLE = False
    logging.warning("PhoBERT/KeyBERT/Underthesea dependencies not available. Using basic preprocessing.")

logger = logging.getLogger(__name__)


class TextPreprocessor: 
    def __init__(self, use_phobert: bool = True):
        self.use_phobert = use_phobert and PHOBERT_AVAILABLE
        
        self.stopwords = self._load_stopwords()
        
        self.trusted_domains = {
            'vnexpress.net', 'tuoitre.vn', 'thanhnien.vn',
            'dantri.com.vn', 'vietnamnet.vn'
        }
        
        if self.use_phobert:
            self._init_phobert()
        
        logger.info(f"✅ Preprocessor initialized (PhoBERT: {self.use_phobert})")
    
    def _init_phobert(self):
        try:
            logger.info("Loading PhoBERT models...")
            
            model_name = 'bkai-nits-hust/sbert-v-phobert-base'
            
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.phobert_model = AutoModel.from_pretrained(model_name)
            self.phobert_model.eval()  
            
            self.kw_model = KeyBERT(model=self.phobert_model)
            
            logger.info(f"✅ PhoBERT loaded successfully! (Model: {model_name})")
            
        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}")
            self.use_phobert = False
    
    def _load_stopwords(self):

        return set([
            'và', 'hoặc', 'của', 'có', 'được', 'đã', 'đang', 'sẽ',
            'này', 'đó', 'kia', 'các', 'những', 'cho', 'từ', 'với',
            'trong', 'ngoài', 'trên', 'dưới', 'là', 'thì', 'mà',
            'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám',
            'cũng', 'để', 'vào', 'ra', 'đến', 'bị', 'bởi', 'còn',
            'khi', 'lại', 'sau', 'trước', 'nếu', 'không', 'chỉ',
            'như', 'theo', 'đều', 'rất', 'hay', 'về', 'tại', 'do',
            'đây', 'đấy', 'ấy', 'nào', 'gì', 'ai', 'đâu',
            'bao', 'lúc', 'nơi', 'người', 'việc', 'chúng', 'nhiều'
        ])
    
    def normalize_text(self, text: str) -> str:

        if not text:
            return ""
        
        text = unicodedata.normalize('NFC', text)
        text = text.lower()
        
        text = re.sub(
            r'[^\w\s.,!?;:\-\(\)áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]',
            ' ',
            text
        )
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def simple_tokenize(self, text):
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def extract_title_from_text(self, text: str) -> str:
        
        if self.use_phobert:
            try:
                sentences = sent_tokenize(text)
                if not sentences:
                    return text.split('.')[0].strip()
                
                scores = []
                for i, sent in enumerate(sentences[:5]):  # Chỉ xét 5 câu đầu
                    score = 0
                    
                    score += (5 - i) * 2
                    
                    word_count = len(sent.split())
                    if 5 <= word_count <= 20:
                        score += 3
                    elif 3 <= word_count <= 25:
                        score += 1
                    
                    entities = ner(sent)
                    score += len(entities)
                    
                    scores.append((sent, score))
                
                best_sentence = max(scores, key=lambda x: x[1])[0]
                return best_sentence.strip()
                
            except Exception as e:
                logger.warning(f"PhoBERT title extraction failed: {e}")
        
        return text.split('.')[0].strip()
    
    def extract_keywords_phobert(self, text: str, top_n: int = 15) -> List[str]:
        try:

            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  
                stop_words=list(self.stopwords),
                top_n=top_n,
                use_mmr=True,  
                diversity=0.5
            )
            
            return [kw[0] for kw in keywords]
            
        except Exception as e:
            logger.error(f"KeyBERT extraction failed: {e}")
            return []
    
    def extract_keywords_basic(self, text: str, top_n: int = 15) -> List[str]:

        normalized = self.normalize_text(text)
        tokens = self.simple_tokenize(normalized)
        
        filtered_tokens = []
        for token in tokens:
            if token in '.,!?;:()-':
                continue
            if token.lower() in self.stopwords:
                continue
            if len(token) < 3 or token.isdigit():
                continue
            if not any(c.isalnum() for c in token):
                continue
            
            filtered_tokens.append(token.lower())
        
        word_freq = Counter(filtered_tokens)
        keywords = [word for word, freq in word_freq.most_common(top_n)]
        
        return keywords
    
    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        
        if self.use_phobert:
            phobert_kws = self.extract_keywords_phobert(text, top_n)
            if phobert_kws:
                logger.info(f"✅ Extracted {len(phobert_kws)} keywords via PhoBERT/KeyBERT")
                return phobert_kws
        
        logger.info("Using basic keyword extraction (fallback)")
        return self.extract_keywords_basic(text, top_n)
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        
        if not self.use_phobert:
            return []
        
        try:
            entities = ner(text)
            return [(e[0], e[3]) for e in entities if e[3] != 'O']
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            return []
    
    def extract_numbers_from_text(self, text: str) -> List[str]:
        patterns = [
            r'\d+[\.,]\d+[\.,]\d+',  
            r'\d+\s*(?:triệu|tỷ|nghìn|ngàn|tỉ)',
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


    def is_valid_article_url(self, url):
        
        invalid_patterns = [
            '/topic/', '/category/', '/tag/', '/search', '/tim-kiem',
            '/video/', '/podcast/', '/page/', '/chu-de/', '/folder/',
            '/gallery/', '/photo/'
        ]
        
        has_number = any(char.isdigit() for char in url)
        is_not_category = not any(pattern in url.lower() for pattern in invalid_patterns)
        
        url_path = url.split('/')[-1]
        has_sufficient_length = len(url_path) > 15
        
        return has_number and is_not_category and has_sufficient_length
    
    def extract_from_url(self, url):
        
        if not self.is_valid_article_url(url):
            logger.warning(f"URL may not be a valid article: {url}")
        
        try:
            logger.info(f"Extracting content from: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            logger.info(f"HTTP Status: {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(["script", "style", "iframe", "noscript", "nav", "footer", "header"]):
                tag.decompose()
            
            title = self._extract_title(soup)
            logger.info(f"Title extracted: {title[:100]}...")
            
            description = self._extract_description(soup)
            
            content = self._extract_content(soup, url)
            
            if not content or len(content) < 100:
                logger.error(f"Content too short: {len(content)} characters")
                return None
            
            logger.info(f"Successfully extracted {len(content)} characters")
            
            return {
                'title': self.normalize_text(title),
                'description': self.normalize_text(description),
                'content': self.normalize_text(content),
                'url': url,
                'domain': urlparse(url).netloc
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__} - {str(e)}")
            return None
    
    def _extract_title(self, soup):
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()
        
        if not title:
            meta_title = soup.find('meta', property='og:title')
            if meta_title and meta_title.has_attr('content'):
                title = meta_title['content'].strip()
        
        return title
    
    def _extract_description(self, soup):
        
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc:
            meta_desc = soup.find('meta', attrs={'property': 'og:description'})
        
        if meta_desc and meta_desc.has_attr('content'):
            description = meta_desc['content'].strip()
        
        return description
    
    def _extract_content(self, soup, url):
        
        content = ""
        domain = urlparse(url).netloc
        
        article = soup.find('article')
        if article:
            content = self._extract_paragraphs(article)
            if len(content) > 200:
                logger.info(f"Content found in <article> tag: {len(content)} chars")
                return content
        
        if not content or len(content) < 200:
            content_divs = soup.find_all(
                'div',
                class_=re.compile(r'(content|article|body|detail|story|entry|post)', re.I)
            )
            for div in content_divs:
                temp_content = self._extract_paragraphs(div)
                if len(temp_content) > len(content):
                    content = temp_content
                    if len(content) > 500:
                        logger.info(f"Content found in content div: {len(content)} chars")
                        break
        
        if not content or len(content) < 200:
            content = self._extract_domain_specific(soup, domain)
        
        if not content or len(content) < 200:
            paragraphs = soup.find_all('p')
            texts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
            content = ' '.join(texts[:50])
            logger.info(f"Fallback extraction: {len(content)} chars")
        
        return content
    
    def _extract_paragraphs(self, element):
        
        paragraphs = element.find_all('p')
        texts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
        return ' '.join(texts)
    
    def _extract_domain_specific(self, soup, domain):
        
        content = ""
        
        if 'vnexpress.net' in domain:
            vnexpress = soup.find(['article', 'div'], class_='fck_detail')
            if vnexpress:
                content = self._extract_paragraphs(vnexpress)
                logger.info(f"VnExpress content: {len(content)} chars")
        
        elif 'tuoitre.vn' in domain:
            tuoitre = soup.find('div', id='main-detail-content')
            if tuoitre:
                content = self._extract_paragraphs(tuoitre)
                logger.info(f"Tuổi Trẻ content: {len(content)} chars")
        
        elif 'thanhnien.vn' in domain:
            thanhnien = soup.find('div', class_=re.compile('content|detail|body', re.I))
            if thanhnien:
                content = self._extract_paragraphs(thanhnien)
                logger.info(f"Thanh Niên content: {len(content)} chars")
        
        elif 'dantri.com.vn' in domain:
            dantri = soup.find('div', class_=re.compile('detail|content', re.I))
            if dantri:
                content = self._extract_paragraphs(dantri)
                logger.info(f"Dân Trí content: {len(content)} chars")
        
        return content

    def process_input(self, input_data: str, input_type: str = 'text') -> Optional[Dict]:
        if input_type == 'url':

            logger.info(f"Processing URL: {input_data}")
            return self._process_url(input_data)
        
        else:  

            logger.info(f"Processing text: {len(input_data)} characters")
            
            normalized = self.normalize_text(input_data)
            
            extracted_title = ""  
            logger.info("📝 Input type is 'text', skipping title extraction.")

            keywords = self.extract_keywords(normalized, top_n=15)
            logger.info(f"🔑 Keywords: {keywords[:10]}")
            
            entities = self.extract_named_entities(input_data)
            if entities:
                logger.info(f"👤 Entities: {entities[:5]}")
            
            numbers = self.extract_numbers_from_text(input_data)
            if numbers:
                logger.info(f"🔢 Numbers: {numbers}")
            
            if entities:
                entity_words = [e[0] for e in entities[:5]]
                keywords = list(dict.fromkeys(entity_words + keywords))[:15]
            
            return {
                'original_input': input_data,
                'input_type': 'text',
                'title': extracted_title,  
                'content': normalized,
                'full_text': normalized,
                'keywords': keywords,
                'entities': entities,
                'numbers': numbers,
                'domain': None
            }

    def _process_url(self, url: str) -> Optional[Dict]:
        
        extracted = self.extract_from_url(url)
        if not extracted:
            logger.error("Failed to extract content from URL")
            return None
        
        full_text = f"{extracted['title']} {extracted['description']} {extracted['content']}"
        
        logger.info(f"Extracted {len(full_text)} characters from URL")
        
        keywords = self.extract_keywords(full_text, top_n=15)
        logger.info(f"🔑 Keywords: {keywords[:10]}")

        entities = self.extract_named_entities(full_text)
        if entities:
            logger.info(f"👤 Entities: {entities[:5]}")
            
        numbers = self.extract_numbers_from_text(full_text)
        if numbers:
            logger.info(f"🔢 Numbers: {numbers}")

        if entities:
            entity_words = [e[0] for e in entities[:5]]
            keywords = list(dict.fromkeys(entity_words + keywords))[:15]

        return {
            'original_input': url,
            'input_type': 'url',
            'title': extracted['title'],
            'content': extracted['content'],
            'full_text': full_text,
            'keywords': keywords,       
            'entities': entities,       
            'numbers': numbers,         
            'domain': extracted['domain']
        }


if __name__ == "__main__":
    pass