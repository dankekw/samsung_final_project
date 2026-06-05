import os
import sys
import logging
import requests
from dotenv import load_dotenv

from api_client import fetch_news_feed
from parser import extract_article_text
import formatter as fmt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("client_app.log")]
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

def main():
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    server_url = os.getenv("SUMM_SERVER_URL")

    downloaded_articles = []
    cached_texts = {}

    while True:
        fmt.clear_screen()
        fmt.print_header("Welcome to CLI")
        
        server_status = "ONLINE" if server_url else "NOT CONFIGURED"
        print(f"Config: Server -> {server_status} | Loaded Articles: {len(downloaded_articles)}")
        
        if downloaded_articles:
            print("\n")
            fmt.print_news_list(downloaded_articles)

        menu_actions = [
            ("1", "Fetch fresh news (Alpha Vantage API)"),
        ]
        if downloaded_articles:
            menu_actions.extend([
                ("2", "Read full article text (Web Parsing)"),
                ("3", "Run Deep AI Summarization & Sentiment Analysis"),
            ])

        menu_actions.append(("Q", "Exit Application"))
        
        fmt.print_menu(menu_actions)
        choice = input("\nEnter your choice >>> ").strip().upper()

        #fetch data choice
        if choice == "1":
            ticker = input("\nEnter company ticker (default: AAPL): ").strip().upper() or "AAPL"
            limit_str = input("Enter max articles count (default: 5): ").strip()
            limit = int(limit_str) if limit_str.isdigit() else 5
            
            if not api_key:
                print("\nError: ALPHA_VANTAGE_KEY not found in .env files.")
                input("\nPress Enter to continue...")
                continue
                
            print(f"\nRequesting ticker {ticker} from cloud data providers...")
            downloaded_articles = fetch_news_feed(ticker, api_key, limit)
            cached_texts.clear() # Wipe text cache on refresh
            
            print(f"Download complete! Loaded {len(downloaded_articles)} articles.")
            input("\nPress Enter to view table...")

        #web parsing
        elif choice == "2" and downloaded_articles:
            idx_str = input(f"\nEnter article index (1-{len(downloaded_articles)}): ").strip()
            if not idx_str.isdigit() or not (1 <= int(idx_str) <= len(downloaded_articles)):
                print("Invalid article index!")
                input("\nPress Enter...")
                continue
                
            idx = int(idx_str) - 1
            selected_art = downloaded_articles[idx]
            url = selected_art.get('url', '')

            # Check local memory cache first to avoid re-downloading
            if idx not in cached_texts:
                print("\nParsing remote HTML content via newspaper3k engines...")
                cached_texts[idx] = extract_article_text(url)

            fmt.clear_screen()
            fmt.print_full_article(selected_art, cached_texts[idx])
            input("\nReturn to main menu (Press Enter)...")

        #main functions
        elif choice == "3" and downloaded_articles:
            if not server_url:
                print("\nError: SUMM_SERVER_URL is missing in configuration variables.")
                input("\nPress Enter...")
                continue

            idx_str = input(f"\nSelect article to analyze (1-{len(downloaded_articles)}): ").strip()
            if not idx_str.isdigit() or not (1 <= int(idx_str) <= len(downloaded_articles)):
                print("Invalid index!")
                input("\nPress Enter...")
                continue

            idx = int(idx_str) - 1
            selected_art = downloaded_articles[idx]
            
            # Ensure text exists
            if idx not in cached_texts or not cached_texts[idx]:
                print("\nPre-fetching full text assets...")
                cached_texts[idx] = extract_article_text(selected_art.get('url', ''))

            article_text = cached_texts[idx]
            if not article_text:
                print("Critical: Unable to parse this website source code. Execution halted.")
                input("\nPress Enter...")
                continue

            print("\nSending matrix payloads to GPU Colab Server (BART, PEGASUS, FinBERT running)...")
            
            payload = {
                "article_text": article_text,
                "target_summary": selected_art.get("summary", "")
            }
            
            try:
                response = requests.post(f"{server_url}/analyze", json=payload, timeout=90)
                if response.status_code == 200:
                    fmt.clear_screen()
                    fmt.print_analysis_result(selected_art, response.json())
                else:
                    print(f" Server side validation error code: {response.status_code}")
            except Exception as e:
                print(f"Connection pipeline failure: {e}")
                logger.error(f"Failed connection to API endpoint: {e}")
                
            input("\nReturn to main menu (Press Enter)...")

        #exit
        elif choice == "Q":
            fmt.clear_screen()
            print("\nThank you for using Financial NLP CLI application. Goodbye!\n")
            sys.exit(0)

if __name__ == "__main__":
    main()