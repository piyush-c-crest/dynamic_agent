"""Thin HTTP client for the Dynamic Agent Orchestrator API."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

DEFAULT_API_BASE = os.getenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:8000")


class OrchestratorClient:
    def __init__(self, base_url: str = DEFAULT_API_BASE, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def get_config(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/config", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def start_chat(
        self,
        prompt: str,
        llm_settings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": prompt}
        if llm_settings:
            payload["llm_settings"] = llm_settings
        r = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/chat/{run_id}", timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def list_history(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/history", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_history(self, run_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/history/{run_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
