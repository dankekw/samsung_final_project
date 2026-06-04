import os
import shutil
import textwrap

def clear_screen():
    """clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_terminal_width(default=80):
    """returns current terminal width"""
    return shutil.get_terminal_size((default, 20)).columns

def print_header(title_text: str):
    """prints a styled header block"""
    width = min(get_terminal_width(), 100)
    print("=" * width)
    print(f" {title_text} ".center(width, "═"))
    print("=" * width)

def print_menu(options: list):
    """draws available menu options"""
    width = min(get_terminal_width(), 100)
    print("\n" + "─" * width)
    print(" available functions: ".center(width, "─"))
    for key, desc in options:
        print(f"  [{key}] {desc}")
    print("─" * width)

def print_news_list(articles: list):
    """prints a compact table of downloaded articles"""
    if not articles:
        print("\n[!] No articles downloaded yet. Please fetch news first.")
        return

    width = min(get_terminal_width(), 100)
    print_header("Downloaded financial news")
    
    # Header row
    print(f"{'#':<3} | {'SOURCE':<12} | {'TITLE':<{width-22}}")
    print("-" * width)
    
    for idx, art in enumerate(articles, 1):
        title = art.get('title', 'Untitled')
        source = art.get('source', 'Unknown')
        # Truncate title if it exceeds line width
        max_title_len = width - 22
        display_title = title[:max_title_len-3] + "..." if len(title) > max_title_len else title
        print(f"{idx:<3} | {source:<12} | {display_title}")
    print("-" * width)

def print_full_article(article_data: dict, full_text: str):
    """renders the parsed full text of a web article."""
    width = min(get_terminal_width(), 100)
    print_header("ARTICLE VIEW MODE")
    print(f"📰 Title  : {article_data.get('title')}")
    print(f"🏢 Source : {article_data.get('source')}")
    print(f"🔗 URL    : {article_data.get('url')}")
    print("-" * width)
    
    if not full_text:
        print("\n[!] Full text content is unavailable. Try another article.")
        return

    # Wrap paragraph strings smoothly
    wrapped_text = textwrap.fill(full_text, width=width - 4)
    print(wrapped_text)
    print("-" * width)

def print_analysis_result(article_data: dict, server_response: dict):
    """displays complex BART/PEGASUS summarization data and ROUGE/Sentiment analytics."""
    width = min(get_terminal_width(), 100)
    print_header("Analysis results")
    print(f"Title: {article_data.get('title')}")
    print("-" * width)

    summaries = server_response.get("summaries", {})
    sentiment = server_response.get("sentiment", "UNKNOWN").upper()
    target_summary = article_data.get("summary", "")

    # 1. Output generative summaries
    for model_name, model_data in summaries.items():
        print(f"[Model: {model_name.upper()}] Generated Summary:\n")
        gen_summary = model_data.get("summary", "")
        print(textwrap.fill(gen_summary, width=width - 4, initial_indent="   ", subsequent_indent="   "))
        
        # Display ROUGE
        rouge = model_data.get("rouge", {})
        print(f"\nROUGE Scores -> R1: {rouge.get('rouge1', 0):.4f} | R2: {rouge.get('rouge2', 0):.4f} | RL: {rouge.get('rougeL', 0):.4f}")
        print("-" * width)

    # 2. Target Reference Summary
    if target_summary:
        print("[Reference Summary] (Alpha Vantage baseline):\n")
        print(textwrap.fill(target_summary, width=width - 4, initial_indent="   ", subsequent_indent="   "))
        print("-" * width)

    # 3. Sentiment Classifier
    sentiment_emoji = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(sentiment, "⚪")
    print(f"Market Sentiment: {sentiment_emoji} {sentiment}")
    print("=" * width)