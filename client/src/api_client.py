import logging
import requests

logger = logging.getLogger(__name__)

def fetch_news_feed(ticker: str, api_key: str, limit: int = 10) -> list:
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    logger.info(f"Requesting news feed for ticker '{ticker}' (limit: {limit})...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        feed = data.get("feed", [])
        logger.info(f"Successfully retrieved {len(feed)} articles from Alpha Vantage API.")
        return feed[:limit]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while requesting Alpha Vantage API: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error while parsing API response: {e}", exc_info=True)
        return []