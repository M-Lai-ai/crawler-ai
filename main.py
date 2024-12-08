# main.py

import argparse
import yaml
from webcrawler import WebCrawler

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Ultra Modulaire Web Crawler")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file (YAML)')
    args = parser.parse_args()

    config = load_config(args.config)

    crawler = WebCrawler(config)
    crawler.crawl()

if __name__ == "__main__":
    main()
