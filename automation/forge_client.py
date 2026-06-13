"""Minimal Forge web API client for the automation pipeline."""

import time

import requests


class ForgeClient:
    def __init__(self, base_url, timeout=900):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def alive(self):
        try:
            r = requests.get(f"{self.base_url}/internal/ping", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_alive(self, retries=3, delay=3):
        for _ in range(retries):
            if self.alive():
                return True
            time.sleep(delay)
        return False

    def txt2img(self, payload):
        r = requests.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def unload_checkpoint(self):
        """Ask Forge to drop the image model from VRAM. Best-effort."""
        try:
            r = requests.post(
                f"{self.base_url}/sdapi/v1/unload-checkpoint", timeout=120
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def reload_checkpoint(self):
        try:
            r = requests.post(
                f"{self.base_url}/sdapi/v1/reload-checkpoint", timeout=300
            )
            return r.status_code == 200
        except requests.RequestException:
            return False
