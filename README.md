# Crawler AI
[![Description de l'image](https://raw.githubusercontent.com/M-Lai-ai/logo/refs/heads/main/favicon.ico)](https://votre-lien-cible.com)

Bienvenue dans **Crawler AI**, un outil ultra modulaire et robuste conçu pour explorer des sites web, extraire du contenu, télécharger des fichiers, et réécrire ce contenu à l’aide d’un Large Language Model (LLM). Ce projet est maintenu par **M-LAI** et se trouve sur le dépôt GitHub suivant : [https://github.com/M-Lai-ai/crawler-ai.git](https://github.com/M-Lai-ai/crawler-ai.git)

## Caractéristiques

- **Exploration Web Ultra Modulaire** :  
  Configurez le crawler via un simple fichier YAML (config.yaml) pour définir :
  - L’URL de départ
  - La profondeur maximale d’exploration
  - Le téléchargement conditionnel de documents, images, PDFs, etc.
  - L’utilisation optionnelle de Playwright pour le rendu de pages dynamiques
  
- **Réécriture via LLM** :  
  Intégrez facilement un modèle de type GPT (OpenAI, Anthropic, Mistral, etc.) pour réécrire le contenu extrait, le rendant plus homogène, clair ou adapté à vos besoins.
  
- **Rapports et Export** :
  - Génération de rapports TXT, JSON et d’un sitemap XML
  - Statistiques détaillées sur les pages traitées, le nombre de fichiers téléchargés et le statut final du crawl
  
- **Résilience et Robustesse** :
  - Stratégie de retries sur les requêtes HTTP
  - Gestion améliorée des erreurs et logs exhaustifs
  - Gestion de l’encodage et des formats de fichiers variés
  

## Prérequis

- Python 3.8+
- Node.js (si vous utilisez Playwright)
- Un ou plusieurs comptes API pour votre LLM préféré (e.g., OpenAI). Clés à renseigner dans `config.yaml`.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/M-Lai-ai/crawler-ai.git
   cd crawler-ai
   ```

2. Créez un environnement virtuel et installez les dépendances :
   ```bash
   python -m venv venv
   source venv/bin/activate  # sous Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. (Optionnel) Installation de Playwright pour le rendu de pages JavaScript :
   ```bash
   playwright install
   ```

## Configuration

Modifiez le fichier `config.yaml` pour adapter le comportement du crawler à vos besoins :

- `start_url` : URL de départ du crawling.
- `max_depth` : Profondeur maximale d’exploration.
- `use_playwright` : `true/false` pour activer ou non Playwright.
- `download_pdf`, `download_doc`, `download_image`, ... : Activer/désactiver le téléchargement de certains types de fichiers.
- Section `llm` : Fournisseur de LLM (OpenAI, Anthropic, Mistral) et clé(s) API correspondante(s).

Exemple minimal :

```yaml
start_url: "https://exemple.com"
max_depth: 1
use_playwright: false
download_pdf: true
max_urls: null

llm:
  provider: "openai"
  api_keys:
    - "votre_cle_api_openai"
  model: "gpt-4"
  system_prompt: "You are a helpful assistant."
  max_tokens_per_request: 2048
  temperature: 1
  top_p: 1

logging:
  level: "INFO"
  file: "logs/crawler.log"
```

## Utilisation

1. Assurez-vous que votre configuration est en place (fichier `config.yaml`).
2. Lancez le crawler :
   ```bash
   python main.py --config config.yaml
   ```
3. Les résultats, logs et rapports seront disponibles dans un dossier nommé `crawler_output_YYYYMMDD_HHMMSS`.

## Assistance

Pour toute question, suggestion ou demande d’amélioration, vous pouvez contacter l’équipe M-LAI à l’adresse suivante :  
**admin@m-lai.ai**

Vous pouvez également créer une [issue sur GitHub](https://github.com/M-Lai-ai/crawler-ai/issues) pour signaler un bug ou proposer une nouvelle fonctionnalité.



**M-LAI** - Des solutions IA modulaires et évolutives pour vos besoins en traitement de l’information.
