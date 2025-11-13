import logging
import random
import re
import time
from typing import Dict, Optional
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup, element
from curl_cffi import requests as cffi_requests

from text_utils import normalize_text

logger = logging.getLogger(__name__)


class Crawler:

    def __init__(self):
        """Khởi tạo Crawler."""
        logger.info("Crawler initialized")
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        # Các mẫu URL thường là trang danh mục, không phải bài báo
        self.invalid_patterns = [
            "/topic/",
            "/category/",
            "/tag/",
            "/search",
            "/tim-kiem",
            "/video/",
            "/podcast/",
            "/page/",
            "/chu-de/",
            "/folder/",
            "/gallery/",
            "/photo/",
        ]

    def is_valid_article_url(self, url: str) -> bool:
        try:
            has_number = any(char.isdigit() for char in url)
            is_not_category = not any(
                pattern in url.lower() for pattern in self.invalid_patterns
            )
            url_path = url.split("/")[-1]
            has_sufficient_length = len(url_path) > 15
            return has_number and is_not_category and has_sufficient_length
        except Exception:
            # Nếu URL có định dạng quá tệ (ví dụ: không phải string), trả về False
            return False

    def extract_from_url(self, url: str) -> Optional[Dict[str, str]]:
        if not self.is_valid_article_url(url):
            logger.warning(f"URL may not be a valid article: {url}")

        # Chiến lược 1: Thử trực tiếp bằng curl_cffi (giả mạo trình duyệt)
        result = self._try_requests_method(url)
        if result:
            return result

        # Chiến lược 2: Thử qua proxy của archive.org (nếu trang gốc bị lỗi 404/503)
        logger.warning("Method 1 failed, trying archive.org proxy...")
        result = self._try_archive_method(url)
        if result:
            return result

        # Chiến lược 3: Thử lấy snippet (đoạn trích) từ Google Search
        logger.warning("Method 2 failed, trying search snippet extraction...")
        result = self._try_search_snippet_method(url)
        if result:
            return result

        logger.error(f"All extraction methods failed for: {url}")
        return None

    def _try_requests_method(
        self, url: str, max_retries: int = 3
    ) -> Optional[Dict[str, str]]:
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        domain = urlparse(url).netloc
        if "vnexpress.net" in domain:
            headers["Referer"] = "https://www.google.com/search?q=vnexpress"

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries}: Extracting from {url} using curl_cffi"
                )
                response = cffi_requests.get(
                    url,
                    headers=headers,
                    timeout=30,
                    allow_redirects=True,
                    verify=True,
                    impersonate="chrome120",  # Giả mạo Chrome 120 để vượt qua bot detection
                )

                if response.status_code == 200:
                    # Phân tích HTML nếu tải thành công
                    soup = BeautifulSoup(response.content, "html.parser")

                    # Dọn dẹp các thẻ không chứa nội dung
                    for tag in soup(
                        [
                            "script",
                            "style",
                            "iframe",
                            "noscript",
                            "nav",
                            "footer",
                            "header",
                        ]
                    ):
                        tag.decompose()

                    title = self._extract_title(soup)
                    description = self._extract_description(soup)
                    content = self._extract_content(soup, url)

                    if content and len(content) > 100:
                        logger.info(
                            f"Successfully extracted {len(content)} characters (via curl_cffi)"
                        )
                        return {
                            "title": normalize_text(title),
                            "description": normalize_text(description),
                            "content": normalize_text(content),
                            "url": url,
                            "domain": domain,
                        }

                logger.warning(
                    f"Attempt {attempt + 1} failed: Status {response.status_code}"
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

            if attempt < max_retries - 1:
                # Chờ một khoảng ngẫu nhiên trước khi thử lại
                time.sleep(random.uniform(2, 5))

        return None

    def _try_archive_method(self, url: str) -> Optional[Dict[str, str]]:
        try:
            archive_api = f"http://archive.org/wayback/available?url={url}"
            response = requests.get(archive_api, timeout=10)
            data = response.json()

            if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
                archive_url = data["archived_snapshots"]["closest"]["url"]
                logger.info(f"Found archive.org snapshot: {archive_url}")

                archive_response = requests.get(archive_url, timeout=30)
                if archive_response.status_code == 200:
                    soup = BeautifulSoup(archive_response.content, "html.parser")
                    for tag in soup(
                        [
                            "script",
                            "style",
                            "iframe",
                            "noscript",
                            "nav",
                            "footer",
                            "header",
                        ]
                    ):
                        tag.decompose()

                    title = self._extract_title(soup)
                    description = self._extract_description(soup)
                    content = self._extract_content(soup, url)

                    if content and len(content) > 100:
                        logger.info(
                            f"Archive.org extraction successful: {len(content)} chars"
                        )
                        return {
                            "title": normalize_text(title),
                            "description": normalize_text(description),
                            "content": normalize_text(content),
                            "url": url,
                            "domain": urlparse(url).netloc,
                        }
        except Exception as e:
            logger.warning(f"Archive.org method failed: {str(e)}")

        return None

    def _try_search_snippet_method(self, url: str) -> Optional[Dict[str, str]]:
        try:
            search_query = (
                f"site:{urlparse(url).netloc} {url.split('/')[-1].replace('-', ' ')}"
            )
            search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"

            headers = {"User-Agent": random.choice(self.user_agents)}

            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            search_divs = soup.find_all("div", class_="g")
            for div in search_divs:
                link = div.find("a", href=True)
                if link and url in link["href"]:
                    title_elem = div.find("h3")
                    title = title_elem.get_text(strip=True) if title_elem else ""

                    snippet_elem = div.find(
                        ["div", "span"], class_=re.compile("VwiC3b|s3v9rd")
                    )
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if title and snippet and len(snippet) > 100:
                        logger.info("Search snippet extraction successful")
                        return {
                            "title": normalize_text(title),
                            "description": "",
                            "content": normalize_text(snippet),
                            "url": url,
                            "domain": urlparse(url).netloc,
                        }
        except Exception as e:
            logger.warning(f"Search snippet method failed: {str(e)}")

        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text().strip()

        if not title:
            meta_title = soup.find("meta", property="og:title")
            if meta_title and meta_title.has_attr("content"):
                title = meta_title["content"].strip()

        return title

    def _extract_description(self, soup: BeautifulSoup) -> str:
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if not meta_desc:
            meta_desc = soup.find("meta", attrs={"property": "og:description"})

        if meta_desc and meta_desc.has_attr("content"):
            description = meta_desc["content"].strip()

        return description

    def _extract_paragraphs(self, element: element.Tag) -> str:
        paragraphs = element.find_all("p")
        texts = [
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 30  # Lọc các thẻ <p> rỗng hoặc quá ngắn
        ]
        return " ".join(texts)

    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        content = ""
        domain = urlparse(url).netloc

        # 1. Thử tìm thẻ <article>
        article = soup.find("article")
        if article:
            content = self._extract_paragraphs(article)
            if len(content) > 200:
                logger.info(f"Content found in <article> tag: {len(content)} chars")
                return content

        # 2. Thử tìm các class CSS phổ biến
        if not content or len(content) < 200:
            content_divs = soup.find_all(
                "div",
                class_=re.compile(
                    r"(content|article|body|detail|story|entry|post)", re.I
                ),
            )
            best_content = ""
            for div in content_divs:
                temp_content = self._extract_paragraphs(div)
                if len(temp_content) > len(best_content):
                    best_content = temp_content
            content = best_content  # Cập nhật content ngay cả khi < 200

            if len(content) > 200:
                logger.info(f"Content found in common div class: {len(content)} chars")
                return content

        # 3. Thử theo bộ chọn (selector) cụ thể cho 5 trang báo
        if not content or len(content) < 200:
            domain_content = self._extract_domain_specific(soup, domain)
            if len(domain_content) > len(content):
                content = domain_content
                if len(content) > 200:
                    logger.info(
                        f"Content found via domain-specific rule: {len(content)} chars"
                    )
                    return content

        # 4. Fallback: Lấy tất cả các thẻ <p>
        if not content or len(content) < 200:
            paragraphs = soup.find_all("p")
            texts = [
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 30
            ]
            content = " ".join(texts[:50])  # Lấy 50 đoạn đầu tiên
            logger.info(f"Fallback extraction (all <p>): {len(content)} chars")

        return content

    def _extract_domain_specific(self, soup: BeautifulSoup, domain: str) -> str:
        content = ""
        element = None
        try:
            if "vnexpress.net" in domain:
                element = soup.find(["article", "div"], class_="fck_detail")
            elif "tuoitre.vn" in domain:
                element = soup.find("div", id="main-detail-content")
            elif "thanhnien.vn" in domain:
                element = soup.find(
                    "div", class_=re.compile("content|detail|body", re.I)
                )
            elif "dantri.com.vn" in domain:
                element = soup.find("div", class_=re.compile("detail|content", re.I))
            elif "vietnamnet.vn" in domain:
                element = soup.find(
                    "div", class_=re.compile("main-content|article-content", re.I)
                )

            if element:
                content = self._extract_paragraphs(element)
                logger.info(
                    f"Specific extractor for {domain} found: {len(content)} chars"
                )
        except Exception as e:
            logger.warning(f"Error in domain specific extractor for {domain}: {e}")

        return content
