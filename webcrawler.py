# webcrawler.py

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
import logging
import time
from collections import defaultdict, deque
import re
from datetime import datetime
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import html2text
from urllib3.exceptions import InsecureRequestWarning
from typing import Tuple, Optional, Set, Dict, List
from xml.etree.ElementTree import Element, SubElement, ElementTree
import json
from concurrent.futures import ThreadPoolExecutor
from math import ceil
import os

from playwright.sync_api import sync_playwright, Page, Browser, Playwright

from llm import LLMClient

# Désactiver les avertissements SSL
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class WebCrawler:
    def __init__(self, config):
        # Configuration générale
        self.start_url = config['start_url']
        self.max_depth = config.get('max_depth', 2)
        self.use_playwright = config.get('use_playwright', False)
        self.download_pdf = config.get('download_pdf', True)
        self.download_doc = config.get('download_doc', True)
        self.download_image = config.get('download_image', True)
        self.download_other = config.get('download_other', True)
        self.max_urls = config.get('max_urls', None)

        # Configuration LLM
        self.llm_config = config.get('llm', {})
        self.llm_client = LLMClient(self.llm_config) if self.llm_config.get('provider') and self.llm_config.get('api_keys') else None
        self.llm_enabled = bool(self.llm_client)

        # Limites des tokens
        self.max_tokens_per_request = self.llm_config.get('max_tokens_per_request', 2048)
        self.chars_per_token = 4
        self.max_chars_per_chunk = self.max_tokens_per_request * self.chars_per_token

        # Initialisation des variables
        self.visited_pages = set()
        self.downloaded_files = set()
        self.domain = urlparse(self.start_url).netloc
        self.site_map: Dict[str, Set[str]] = defaultdict(set)

        # Extraction du pattern de langue depuis l'URL de départ
        self.language_path = re.search(r'/(fr|en)-(ca|us)/', self.start_url)
        self.language_pattern = self.language_path.group(0) if self.language_path else None
        self.language_code = self.language_path.group(1) if self.language_path else None
        self.country_code = self.language_path.group(2) if self.language_path else None

        self.excluded_paths = ['selecteur-de-produits']

        # Création des dossiers de sortie avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path(f"crawler_output_{timestamp}")
        self.create_directories(config)

        # Configuration du logging
        self.setup_logging(config)

        # Statistiques
        self.stats = defaultdict(int)

        # Extensions téléchargeables par catégorie
        self.downloadable_extensions = config.get('downloadable_extensions', {})
        # Retrait des catégories vides
        self.downloadable_extensions = {k: v for k, v in self.downloadable_extensions.items() if v}

        self.all_downloadable_exts = {ext for exts in self.downloadable_extensions.values() for ext in exts}

        # Mapping Content-Type / Extension
        self.content_type_mapping = config.get('content_type_mapping', {})

        self.session = self.setup_session()

        # Convertisseur HTML -> Markdown
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0
        self.html_converter.ignore_images = True
        self.html_converter.single_line_break = False

        # Initialisation de Playwright si nécessaire
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        if self.use_playwright:
            self.init_playwright()

    def init_playwright(self):
        """Initialise Playwright et le navigateur."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

    def setup_session(self) -> requests.Session:
        """Configure une session HTTP avec stratégie de retry et timeouts."""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    def create_directories(self, config):
        """Crée la structure de dossiers nécessaire."""
        directories = ['content', 'content_rewritten', 'logs']
        for category in config.get('downloadable_extensions', {}).keys():
            directories.append(category)
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def setup_logging(self, config):
        """Configure le système de logging."""
        logging_config = config.get('logging', {})
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
        log_file = self.base_dir / log_config.get('file', 'logs/crawler.log')

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting crawler with language pattern: {self.language_pattern}")

    def should_exclude(self, url: str) -> bool:
        """Détermine si une URL doit être exclue."""
        return any(excluded in url for excluded in self.excluded_paths)

    def is_same_language(self, url: str) -> bool:
        """Vérifie si l'URL fait partie du même pattern linguistique."""
        if not self.language_pattern:
            return True
        return self.language_pattern in url

    def is_downloadable_file(self, url: str) -> bool:
        """Vérifie si l'URL pointe vers un fichier téléchargeable."""
        path = urlparse(url).path.lower()
        if not self.all_downloadable_exts:
            return False
        pattern = re.compile(r'\.(' + '|'.join(ext.strip('.') for ext in self.all_downloadable_exts) + r')(\.[a-z0-9]+)?$', re.IGNORECASE)
        return bool(pattern.search(path))

    def head_or_get(self, url: str) -> Optional[requests.Response]:
        """Essaye d'abord HEAD, puis GET."""
        try:
            return self.session.head(url, allow_redirects=True, timeout=10)
        except:
            pass
        try:
            return self.session.get(url, allow_redirects=True, timeout=10, stream=True)
        except:
            return None

    def get_file_type_and_extension(self, url: str, response: requests.Response) -> Tuple[Optional[str], Optional[str]]:
        """Détermine le type de fichier et l'extension."""
        if response is None:
            return None, None

        path = urlparse(url).path.lower()
        content_type = response.headers.get('Content-Type', '').lower()

        # Tentative 1 : Deduire du nom de fichier
        for file_type, extensions in self.downloadable_extensions.items():
            for ext in extensions:
                pattern = re.compile(re.escape(ext) + r'(\.[a-z0-9]+)?$', re.IGNORECASE)
                if pattern.search(path):
                    return file_type, self.content_type_mapping.get(file_type, {}).get(content_type, ext)

        # Tentative 2 : Deduire du Content-Type
        for file_type, mapping in self.content_type_mapping.items():
            if content_type in mapping:
                return file_type, mapping[content_type]

        return None, None

    def sanitize_filename(self, url: str, file_type: str, extension: str, page_number: Optional[int] = None) -> str:
        """Crée un nom de fichier sécurisé."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = url.split('/')[-1] or 'index'
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        name, _ = Path(filename).stem, Path(filename).suffix

        if not extension:
            extension = '.txt'

        if page_number is not None:
            sanitized = f"{name}_page_{page_number:03d}_{url_hash}{extension}"
        else:
            sanitized = f"{name}_{url_hash}{extension}"

        return sanitized

    def download_file(self, url: str) -> bool:
        """Télécharge un fichier si activé."""
        response = self.head_or_get(url)
        if not response or response.status_code != 200:
            logging.warning(f"Failed to retrieve file at {url}")
            return False

        file_type_detected, extension = self.get_file_type_and_extension(url, response)
        if not file_type_detected:
            logging.warning(f"Could not determine the file type for: {url}")
            return False

        if file_type_detected not in self.downloadable_extensions:
            logging.info(f"File type {file_type_detected} not enabled for download.")
            return False

        logging.info(f"Attempting to download {file_type_detected} file from: {url}")

        filename = self.sanitize_filename(url, file_type_detected, extension)
        save_path = self.base_dir / file_type_detected / filename

        if save_path.exists():
            logging.info(f"File already downloaded, skipping: {filename}")
            return False

        try:
            if response.request.method == 'HEAD':
                response = self.session.get(url, stream=True, timeout=20)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self.stats[f'{file_type_detected}_downloaded'] += 1
            self.downloaded_files.add(url)
            logging.info(f"Successfully downloaded {file_type_detected}: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            return False

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Récupère le contenu HTML d'une page."""
        if self.use_playwright and self.page:
            try:
                logging.debug(f"Fetching with Playwright: {url}")
                self.page.goto(url, timeout=20000)
                time.sleep(2)
                return self.page.content()
            except Exception as e:
                logging.error(f"Playwright failed to fetch {url}: {str(e)}")
                return None
        else:
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    return response.text
                else:
                    logging.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                    return None
            except Exception as e:
                logging.error(f"Requests failed to fetch {url}: {str(e)}")
                return None

    def convert_links_to_absolute(self, soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
        """Convertit les liens relatifs en liens absolus."""
        for tag in soup.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            attr = 'href' if tag.name == 'a' else 'src'
            href = tag.get(attr)
            if href:
                absolute_url = urljoin(base_url, href)
                tag[attr] = absolute_url
        return soup

    def clean_text(self, text: str) -> str:
        """Nettoie le texte."""
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_content(self, url: str):
        """Extrait le contenu d'une page HTML -> Markdown."""
        if self.is_downloadable_file(url):
            logging.debug(f"Skipping content extraction for downloadable file: {url}")
            return

        page_content = self.fetch_page_content(url)
        if page_content is None:
            logging.warning(f"Could not retrieve content for: {url}")
            return

        soup = BeautifulSoup(page_content, 'html.parser')
        for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'iframe']):
            element.decompose()

        main_content = (soup.find('main') or soup.find('article') or 
                        soup.find('div', class_='content') or soup.find('div', id='content'))

        if not main_content:
            logging.warning(f"No main content found for: {url}")
            return

        self.convert_links_to_absolute(main_content, url)
        markdown_content = self.html_converter.handle(str(main_content))

        title = soup.find('h1')
        content_parts = []
        if title:
            content_parts.append(f"# {title.get_text().strip()}")
        content_parts.append(f"**Source:** {url}")
        content_parts.append(markdown_content)

        content = self.clean_text('\n\n'.join(content_parts))

        if content:
            filename = self.sanitize_filename(url, 'Doc', '.txt')
            save_path = self.base_dir / 'content' / filename
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.stats['pages_processed'] += 1
            logging.info(f"Successfully saved content to: {filename}")

            # Réécriture LLM immédiate après création du fichier
            if self.llm_enabled and self.llm_client:
                rewritten_text = self.rewrite_text_in_chunks(content)
                if rewritten_text is not None:
                    rewritten_dir = self.base_dir / 'content_rewritten'
                    rewritten_save_path = rewritten_dir / filename
                    with open(rewritten_save_path, 'w', encoding='utf-8') as f:
                        f.write(rewritten_text)
                    logging.info(f"Rewritten content saved to: {rewritten_save_path}")
                else:
                    logging.warning(f"LLM rewriting failed for {filename}")
        else:
            logging.warning(f"No significant content found for: {url}")

        # Détecter et télécharger les fichiers référencés dans la page
        for tag in main_content.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            href = tag.get('href') or tag.get('src')
            if href:
                file_url = urljoin(url, href)
                if self.is_downloadable_file(file_url) and file_url not in self.downloaded_files:
                    self.download_file(file_url)

    def extract_urls(self, start_url: str):
        """
        Extrait récursivement les URLs.
        Construit aussi self.site_map pour cartographier le site.
        """
        queue = deque([(start_url, 0)])
        self.visited_pages.add(start_url)

        crawled_count = 0  # Compteur du nombre d'URL crawlées

        while queue:
            current_url, depth = queue.popleft()

            # Si max_urls est défini et qu'on a déjà atteint le nombre max, on arrête
            if self.max_urls is not None and crawled_count >= self.max_urls:
                logging.info(f"Reached max_urls limit ({self.max_urls}), stopping URL extraction.")
                break

            # Si max_urls n'est pas défini, on respecte max_depth
            if self.max_urls is None and depth > self.max_depth:
                continue

            if self.should_exclude(current_url):
                logging.info(f"Excluded URL: {current_url}")
                continue

            logging.info(f"Extracting URLs from: {current_url} (depth: {depth})")

            crawled_count += 1  # On incrémente après avoir décidé de crawler cette URL

            if self.is_downloadable_file(current_url):
                self.download_file(current_url)
                continue

            page_content = self.fetch_page_content(current_url)
            if page_content is None:
                logging.warning(f"Could not retrieve content for: {current_url}")
                continue

            soup = BeautifulSoup(page_content, 'html.parser')
            child_links = set()
            for tag in soup.find_all(['a', 'link', 'embed', 'iframe', 'object'], href=True):
                href = tag.get('href') or tag.get('src')
                if not href:
                    continue
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)

                if self.is_downloadable_file(absolute_url):
                    self.download_file(absolute_url)
                    continue

                # Vérification des liens internes
                if (self.domain in parsed_url.netloc
                        and self.is_same_language(absolute_url)
                        and not absolute_url.endswith(('#', 'javascript:void(0)', 'javascript:;'))
                        and not self.should_exclude(absolute_url)):
                    child_links.add(absolute_url)
                    if absolute_url not in self.visited_pages:
                        # On ajoute dans la queue uniquement si on n'a pas encore dépassé les limites
                        if self.max_urls is None or crawled_count < self.max_urls:
                            # On ne check la profondeur que si max_urls n'est pas défini
                            if self.max_urls is None and depth + 1 > self.max_depth:
                                continue
                            queue.append((absolute_url, depth + 1))
                            self.visited_pages.add(absolute_url)

            # Mise à jour de la carte du site
            self.site_map[current_url].update(child_links)

    def crawl(self):
        """Méthode principale du crawler."""
        start_time = time.time()
        logging.info(f"Starting crawl of {self.start_url}")
        logging.info(f"Language pattern: {self.language_pattern}")
        logging.info(f"Maximum depth: {self.max_depth}")
        if self.max_urls is not None:
            logging.info(f"Maximum URLs to crawl: {self.max_urls}")

        self.load_downloaded_files()
        error = None
        try:
            # Phase 1: Extraction des URLs
            logging.info("Phase 1: Starting URL extraction")
            self.extract_urls(self.start_url)

            # Phase 2: Extraction du contenu (et réécriture LLM immédiate)
            logging.info("Phase 2: Starting content extraction")
            for i, url in enumerate(self.visited_pages, 1):
                if self.is_downloadable_file(url):
                    continue
                logging.info(f"Processing URL {i}/{len(self.visited_pages)}: {url}")
                self.extract_content(url)
            logging.info("Phase 2: Completed content extraction")

            # Phase 3 (optionnelle): Réécriture du contenu avec LLM sur tous les fichiers restants (si besoin)
            if self.llm_enabled and self.llm_client:
                logging.info("Phase 3: Rewriting content with LLM")
                self.rewrite_all_content()

        except Exception as e:
            error = str(e)
            logging.error(f"Critical error during crawling: {str(e)}")

        end_time = time.time()
        duration = end_time - start_time
        self.generate_report(duration, error=error)
        self.generate_json_report(duration, error=error)
        self.generate_xml_sitemap()

        self.save_downloaded_files()
        if self.use_playwright:
            self.page.close()
            self.browser.close()
            self.playwright.stop()

    def load_downloaded_files(self):
        """Charge les fichiers déjà téléchargés, si disponibles."""
        downloaded_files_path = self.base_dir / 'logs' / 'downloaded_files.txt'
        if downloaded_files_path.exists():
            with open(downloaded_files_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.downloaded_files.add(line.strip())
            logging.info(f"Loaded {len(self.downloaded_files)} downloaded files.")
        else:
            logging.info("No downloaded files tracking file found, starting fresh.")

    def save_downloaded_files(self):
        """Sauvegarde la liste des fichiers téléchargés."""
        downloaded_files_path = self.base_dir / 'logs' / 'downloaded_files.txt'
        try:
            with open(downloaded_files_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.downloaded_files):
                    f.write(url + '\n')
            logging.info(f"Saved {len(self.downloaded_files)} downloaded files.")
        except Exception as e:
            logging.error(f"Error saving downloaded files tracking: {str(e)}")

    def generate_report(self, duration: float, error: Optional[str] = None):
        """Génère un rapport détaillé du crawling (texte)."""
        report_lines = [
            f"Crawler Report",
            f"==============",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Configuration",
            "------------",
            f"Start URL: {self.start_url}",
            f"Language Pattern: {self.language_pattern}",
            f"Max Depth: {self.max_depth}",
            f"Max URLs: {self.max_urls}" if self.max_urls is not None else "",
            f"Duration: {duration:.2f} seconds",
            "",
            "Statistics",
            "---------",
            f"Total URLs found: {len(self.visited_pages)}",
            f"Pages processed: {self.stats['pages_processed']}"
        ]

        # Comptage des fichiers téléchargés par catégorie
        for category in self.downloadable_extensions.keys():
            report_lines.append(f"- {category}: {self.stats.get(category+'_downloaded',0)}")

        report_lines.append("")

        if error:
            report_lines.extend([
                "Errors",
                "------",
                f"Critical Error: {error}",
                ""
            ])

        report_lines.append("Processed URLs")
        report_lines.append("-------------")
        for url in sorted(self.visited_pages):
            report_lines.append(url)

        report_lines.append("")
        report_lines.append("Generated Files")
        report_lines.append("--------------")

        for directory in ['content'] + list(self.downloadable_extensions.keys()):
            dir_path = self.base_dir / directory
            if dir_path.exists():
                files = list(dir_path.iterdir())
                report_lines.append(f"\n{directory} Files ({len(files)}):")
                for file in sorted(files):
                    report_lines.append(f"- {file.name}")

        report_content = "\n".join(report_lines)
        report_path = self.base_dir / 'crawler_report.txt'

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Report generated successfully: {report_path}")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")

        total_downloaded = sum(self.stats.get(k, 0) for k in self.stats if k.endswith('_downloaded'))

        summary = f"""
Crawling Summary
---------------
Start URL: {self.start_url}
Total URLs: {len(self.visited_pages)}
Pages Processed: {self.stats['pages_processed']}
Total Files Downloaded: {total_downloaded}
Duration: {duration:.2f} seconds
Status: {'Completed with errors' if error else 'Completed successfully'}
"""

        try:
            with open(self.base_dir / 'summary.txt', 'w', encoding='utf-8') as f:
                f.write(summary)
            logging.info("Summary generated successfully.")
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")

    def generate_json_report(self, duration: float, error: Optional[str] = None):
        """Génère un rapport détaillé au format JSON."""
        total_downloaded = sum(self.stats.get(k, 0) for k in self.stats if k.endswith('_downloaded'))
        report_data = {
            "configuration": {
                "start_url": self.start_url,
                "language_pattern": self.language_pattern,
                "max_depth": self.max_depth,
                "max_urls": self.max_urls,
                "duration": duration
            },
            "statistics": {
                "total_urls_found": len(self.visited_pages),
                "pages_processed": self.stats.get('pages_processed', 0),
                "files_downloaded": {
                    cat: self.stats.get(cat + '_downloaded', 0) for cat in self.downloadable_extensions.keys()
                },
                "total_files_downloaded": total_downloaded
            },
            "status": "Completed with errors" if error else "Completed successfully",
            "visited_pages": sorted(self.visited_pages),
            "downloaded_files": sorted(self.downloaded_files),
            "error": error if error else None
        }

        json_report_path = self.base_dir / 'report.json'
        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            logging.info(f"JSON report generated successfully: {json_report_path}")
        except Exception as e:
            logging.error(f"Error generating JSON report: {str(e)}")

    def generate_xml_sitemap(self):
        """Génère un fichier XML représentant la structure des liens trouvés."""
        visited = set()

        def add_page_element(parent_elem, url):
            if url in visited:
                return
            visited.add(url)
            page_elem = SubElement(parent_elem, "page", url=url)
            for child_url in sorted(self.site_map[url]):
                add_page_element(page_elem, child_url)

        root = Element("site", start_url=self.start_url)
        add_page_element(root, self.start_url)

        tree = ElementTree(root)
        xml_path = self.base_dir / 'sitemap.xml'
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"XML sitemap generated successfully: {xml_path}")
        except Exception as e:
            logging.error(f"Error generating XML sitemap: {str(e)}")

    def rewrite_all_content(self):
        """Parcourt tous les fichiers .txt dans content/ et les réécrit via le LLM, en chunks si nécessaire."""
        content_dir = self.base_dir / 'content'
        rewritten_dir = self.base_dir / 'content_rewritten'
        txt_files = list(content_dir.glob('*.txt'))

        if not txt_files:
            logging.info("No content files to rewrite.")
            return

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_text = f.read()

            rewritten_text = self.rewrite_text_in_chunks(original_text)
            if rewritten_text is not None:
                save_path = rewritten_dir / txt_file.name
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(rewritten_text)
                logging.info(f"Rewritten content saved to: {save_path}")
            else:
                logging.warning(f"LLM rewriting failed for {txt_file.name}")

    def rewrite_text_in_chunks(self, text: str) -> Optional[str]:
        """
        Divise le texte en chunks si nécessaire, envoie chaque chunk au LLM en parallèle, puis recompose le texte final.
        """
        # Découper le texte en chunks approximatifs
        chunks = self.split_text_into_chunks(text, self.max_chars_per_chunk)
        if not chunks:
            return None

        # Si un seul chunk, traitement direct
        if len(chunks) == 1:
            return self.llm_client.rewrite_text(chunks[0])

        # Sinon, traitement en parallèle
        max_workers = len(self.llm_config.get('api_keys', []))
        if max_workers == 0:
            max_workers = 1

        rewritten_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.llm_client.rewrite_text, chunk) for chunk in chunks]
            for future in futures:
                result = future.result()
                if result is None:
                    logging.error("One of the chunk rewriting failed.")
                    return None
                rewritten_chunks.append(result)

        # Réassembler les chunks
        return "\n".join(rewritten_chunks)

    def split_text_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """
        Divise le texte en chunks d'au plus max_chars caractères.
        On suppose ~4 chars/token, c'est approximatif.
        """
        if len(text) <= max_chars:
            return [text]

        # On découpe en chunks
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            # S'assurer de ne pas couper au milieu d'un mot
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    end = start + last_space
                    chunk = text[start:end]
            chunks.append(chunk)
            start = end
        return chunks
