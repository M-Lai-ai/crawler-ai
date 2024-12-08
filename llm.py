# llm.py

import requests
import logging
from typing import Optional

class LLMClient:
    def __init__(self, config):
        self.provider = config['provider']
        self.api_keys = config['api_keys']
        self.model = config.get('model', 'gpt-4')
        self.system_prompt = config.get('system_prompt', "You are a helpful assistant.")
        self.max_tokens = config.get('max_tokens_per_request', 2048)
        self.temperature = config.get('temperature', 1)
        self.top_p = config.get('top_p', 1)
        self.current_key = 0  # Pour la rotation des clés API

    def rewrite_text(self, prompt: str) -> Optional[str]:
        if self.provider == "openai":
            return self._rewrite_openai(prompt)
        elif self.provider == "anthropic":
            return self._rewrite_anthropic(prompt)
        elif self.provider == "mistral":
            return self._rewrite_mistral(prompt)
        else:
            logging.error("Unsupported LLM provider.")
            return None

    def _get_api_key(self) -> str:
        key = self.api_keys[self.current_key]
        self.current_key = (self.current_key + 1) % len(self.api_keys)
        return key

    def _rewrite_openai(self, prompt: str) -> Optional[str]:
        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        url = "https://api.openai.com/v1/chat/completions"
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            rewritten = response_data["choices"][0]["message"]["content"].strip()
            return rewritten
        except Exception as e:
            logging.error(f"Error rewriting with OpenAI: {e}")
            return None

    def _rewrite_anthropic(self, prompt: str) -> Optional[str]:
        api_key = self._get_api_key()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        url = "https://api.anthropic.com/v1/messages"
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            response_content = response_data.get("content", [])
            rewritten = "".join(part["text"] for part in response_content if part["type"] == "text").strip()
            return rewritten
        except Exception as e:
            logging.error(f"Error rewriting with Anthropic: {e}")
            return None

    def _rewrite_mistral(self, prompt: str) -> Optional[str]:
        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": min(100, self.max_tokens),  # Exemple de limitation
            "stream": False
        }
        url = "https://api.mistral.ai/v1/chat/completions"
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            rewritten = response_data["choices"][0]["message"]["content"].strip()
            return rewritten
        except Exception as e:
            logging.error(f"Error rewriting with Mistral: {e}")
            return None
