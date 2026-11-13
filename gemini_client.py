"""
Gemini API Client (REST via standard library, IPv4-forced for reliable low-latency on macOS).
Loads GEMINI_API_KEY from environment or .env file.
"""
import http.client
import json
import os
import socket
import ssl
import time


def load_api_key(env_path=".env"):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    return v.strip().strip('"').strip("'")
    return ""


class IPv4HTTPSConnection(http.client.HTTPSConnection):
    """Force IPv4 resolution to avoid macOS IPv6 hang on Google APIs."""

    def connect(self):
        addr_infos = socket.getaddrinfo(
            self.host, self.port, socket.AF_INET, socket.SOCK_STREAM
        )
        last_err = None
        for family, socktype, proto, _, sockaddr in addr_infos:
            sock = None
            try:
                sock = socket.socket(family, socktype, proto)
                sock.settimeout(self.timeout)
                sock.connect(sockaddr)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as err:
                last_err = err
                if sock is not None:
                    sock.close()
        if last_err is not None:
            raise last_err
        raise OSError(f"Could not resolve IPv4 address for {self.host}")


class GeminiClient:
    HOST = "generativelanguage.googleapis.com"
    DEFAULT_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.8-flash"]

    def __init__(self, api_key=None, env_path=".env"):
        self.api_key = api_key or load_api_key(env_path)
        self.last_meta = {
            "provider": "gemini",
            "model": None,
            "used_ai": False,
            "latency_ms": 0,
            "error": None,
        }

    def is_configured(self):
        return bool(self.api_key)

    def _request(self, model, payload, timeout=25):
        path = f"/v1beta/models/{model}:generateContent?key={self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ctx = ssl.create_default_context()
        conn = IPv4HTTPSConnection(self.HOST, 443, timeout=timeout, context=ctx)
        try:
            conn.request(
                "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:300]}")
            return json.loads(raw)
        finally:
            conn.close()

    def generate_json(self, prompt, system_instruction="", timeout=20):
        return self._generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            timeout=timeout,
            expect_json=True,
        )

    def generate_text(self, prompt, system_instruction="", timeout=20):
        return self._generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type="text/plain",
            timeout=timeout,
            expect_json=False,
        )

    def _generate(self, prompt, system_instruction, response_mime_type, timeout, expect_json):
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        last_error = None
        for model in self.DEFAULT_MODELS:
            gen_config = {
                "temperature": 0.2 if expect_json else 0.4,
                "responseMimeType": response_mime_type,
            }
            if "3-flash" in model or "3.8-flash" in model:
                gen_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}

            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_config,
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            t0 = time.time()
            try:
                data = self._request(model, payload, timeout=timeout)
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Empty candidates from Gemini API")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                latency_ms = int((time.time() - t0) * 1000)
                self.last_meta = {
                    "provider": "gemini",
                    "model": model,
                    "used_ai": True,
                    "latency_ms": latency_ms,
                    "error": None,
                }
                if expect_json:
                    cleaned = text
                    if cleaned.startswith("```"):
                        cleaned = cleaned.strip("`")
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()
                    return json.loads(cleaned), self.last_meta
                return text, self.last_meta
            except Exception as exc:
                last_error = str(exc)
                self.last_meta = {
                    "provider": "gemini",
                    "model": model,
                    "used_ai": False,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "error": last_error,
                }

        raise RuntimeError(f"All Gemini models failed: {last_error}")
