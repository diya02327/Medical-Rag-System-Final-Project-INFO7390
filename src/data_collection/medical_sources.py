
"""
Medical data collection from reputable sources
"""
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
import logging
from retry import retry
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataCollector:
    """Collect medical information from reputable sources"""
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Medical Research Tool)'
        })
        
    @retry(tries=3, delay=2, backoff=2)
    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with retry logic"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def collect_medlineplus_articles(self, max_articles: int = 500) -> List[Dict]:
        """
        Collect articles from MedlinePlus
        https://medlineplus.gov/
        """
        logger.info("Collecting MedlinePlus articles...")
        articles = []
        
        # MedlinePlus health topics
        base_url = "https://medlineplus.gov"
        topics_url = f"{base_url}/healthtopics.html"
        
        html = self._fetch_url(topics_url)
        if not html:
            return articles
        
        soup = BeautifulSoup(html, 'html.parser')
        topic_links = soup.find_all('a', href=True)
        
        # Filter for condition/symptom pages
        condition_links = [
            link['href'] for link in topic_links 
            if '/ency/article/' in link['href'] or '/healthtopics/' in link['href']
        ][:max_articles]
        
        for link in tqdm(condition_links, desc="MedlinePlus articles"):
            full_url = f"{base_url}{link}" if link.startswith('/') else link
            article = self._parse_medlineplus_article(full_url)
            if article:
                articles.append(article)
                time.sleep(0.5)  # Rate limiting
        
        self._save_articles(articles, "medlineplus")
        return articles
    
    def _parse_medlineplus_article(self, url: str) -> Optional[Dict]:
        """Parse individual MedlinePlus article"""
        html = self._fetch_url(url)
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Unknown"
            
            # Extract main content
            main_content = soup.find('div', {'id': 'topic-content'}) or \
                          soup.find('article') or \
                          soup.find('div', {'class': 'main-content'})
            
            if not main_content:
                return None
            
            # Extract sections
            sections = {}
            current_section = "overview"
            content_parts = []
            
            for element in main_content.find_all(['h2', 'h3', 'p', 'ul', 'ol']):
                if element.name in ['h2', 'h3']:
                    if content_parts:
                        sections[current_section] = ' '.join(content_parts)
                    current_section = element.get_text(strip=True).lower()
                    content_parts = []
                elif element.name == 'p':
                    text = element.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                elif element.name in ['ul', 'ol']:
                    items = [li.get_text(strip=True) for li in element.find_all('li')]
                    content_parts.extend(items)
            
            if content_parts:
                sections[current_section] = ' '.join(content_parts)
            
            return {
                'source': 'MedlinePlus',
                'url': url,
                'title': title_text,
                'sections': sections,
                'full_text': ' '.join(sections.values()),
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def collect_mayo_clinic_articles(self, max_articles: int = 300) -> List[Dict]:
        """
        Collect articles from Mayo Clinic
        https://www.mayoclinic.org/diseases-conditions
        """
        logger.info("Collecting Mayo Clinic articles...")
        articles = []
        
        base_url = "https://www.mayoclinic.org"
        diseases_url = f"{base_url}/diseases-conditions/index"
        
        html = self._fetch_url(diseases_url)
        if not html:
            return articles
        
        soup = BeautifulSoup(html, 'html.parser')
        condition_links = soup.find_all('a', href=True)
        
        # Filter for disease/condition pages
        disease_links = [
            link['href'] for link in condition_links 
            if '/diseases-conditions/' in link['href'] and '/symptoms-causes/' in link['href']
        ][:max_articles]
        
        for link in tqdm(disease_links, desc="Mayo Clinic articles"):
            full_url = f"{base_url}{link}" if link.startswith('/') else link
            article = self._parse_mayo_clinic_article(full_url)
            if article:
                articles.append(article)
                time.sleep(1)  # Rate limiting
        
        self._save_articles(articles, "mayo_clinic")
        return articles
    
    def _parse_mayo_clinic_article(self, url: str) -> Optional[Dict]:
        """Parse individual Mayo Clinic article"""
        html = self._fetch_url(url)
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = soup.find('h1', {'class': 'content-title'}) or soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Unknown"
            
            # Extract sections
            sections = {}
            main_content = soup.find('main') or soup.find('article')
            
            if main_content:
                for section in main_content.find_all('section'):
                    section_title = section.find(['h2', 'h3'])
                    if section_title:
                        section_name = section_title.get_text(strip=True).lower()
                        section_content = []
                        
                        for p in section.find_all('p'):
                            text = p.get_text(strip=True)
                            if text:
                                section_content.append(text)
                        
                        sections[section_name] = ' '.join(section_content)
            
            return {
                'source': 'Mayo Clinic',
                'url': url,
                'title': title_text,
                'sections': sections,
                'full_text': ' '.join(sections.values()),
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def collect_cdc_resources(self, max_articles: int = 200) -> List[Dict]:
        """
        Collect resources from CDC
        https://www.cdc.gov/
        """
        logger.info("Collecting CDC resources...")
        articles = []
        
        # CDC disease categories
        cdc_categories = [
            "https://www.cdc.gov/diseasesconditions/index.html",
            "https://www.cdc.gov/healthyliving/index.html"
        ]
        
        for category_url in cdc_categories:
            html = self._fetch_url(category_url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all('a', href=True)
            
            disease_links = [
                link['href'] for link in links 
                if '/index.html' in link['href'] or '/about/' in link['href']
            ][:max_articles // len(cdc_categories)]
            
            for link in tqdm(disease_links, desc=f"CDC - {category_url}"):
                full_url = f"https://www.cdc.gov{link}" if link.startswith('/') else link
                article = self._parse_cdc_article(full_url)
                if article:
                    articles.append(article)
                    time.sleep(1)
        
        self._save_articles(articles, "cdc")
        return articles
    
    def _parse_cdc_article(self, url: str) -> Optional[Dict]:
        """Parse individual CDC article"""
        html = self._fetch_url(url)
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Unknown"
            
            # Extract main content
            main_content = soup.find('main') or soup.find('div', {'class': 'content'})
            
            sections = {}
            if main_content:
                paragraphs = main_content.find_all('p')
                content_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                sections['overview'] = content_text
            
            return {
                'source': 'CDC',
                'url': url,
                'title': title_text,
                'sections': sections,
                'full_text': sections.get('overview', ''),
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def _save_articles(self, articles: List[Dict], source_name: str):
        """Save collected articles to JSON"""
        if not articles:
            logger.warning(f"No articles to save for {source_name}")
            return
        
        output_file = self.output_dir / f"{source_name}_articles.json"
        
        try:
            # Load existing articles if file exists
            existing_articles = []
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            
            # Merge and deduplicate
            all_articles = existing_articles + articles
            unique_articles = {article['url']: article for article in all_articles}.values()
            
            # Save
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(unique_articles), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(unique_articles)} articles to {output_file}")
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
    
    def collect_all_sources(self) -> Dict[str, List[Dict]]:
        """Collect from all configured sources"""
        all_data = {
            'medlineplus': self.collect_medlineplus_articles(max_articles=500),
            'mayo_clinic': self.collect_mayo_clinic_articles(max_articles=300),
            'cdc': self.collect_cdc_resources(max_articles=200)
        }
        
        total_articles = sum(len(articles) for articles in all_data.values())
        logger.info(f"Total articles collected: {total_articles}")
        
        return all_data


if __name__ == "__main__":
    collector = MedicalDataCollector()
    data = collector.collect_all_sources()
    print(f"Collection complete: {sum(len(v) for v in data.values())} total articles")