import logging
import re
from newspaper import Article

logger = logging.getLogger(__name__)

def extract_article_text(url: str) -> str:
    if not url:
        logger.warning("Empty URL provided for article extraction.")
        return ""
        
    logger.debug(f"Starting article download from URL: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        clean_text = re.sub(r'[\r\n]+', ' ', article.text)
        logger.debug(f"Article successfully parsed. Text length: {len(clean_text)} characters.")
        return clean_text.strip()
        
    except Exception as e:
        logger.warning(f"Failed to parse article at {url}. Reason: {e}")
        return ""