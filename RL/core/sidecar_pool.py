"""Request-level load balancing for OpenAI-compatible sidecar pools."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    OpenAI = None

from config.settings import settings

logger = logging.getLogger(__name__)
_EMBEDDINGS_API_UNHEALTHY_ENDPOINTS: set[str] = set()


def _require_openai_sdk() -> None:
    if OpenAI is None:
        raise ModuleNotFoundError("OpenAIEndpointPool requires optional dependency: openai")


def split_endpoint_list(value: Any) -> List[str]:
    """Parse a comma/semicolon separated endpoint list while preserving order."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).replace(";", ",").split(",")
    endpoints: List[str] = []
    seen = set()
    for raw_item in raw_items:
        endpoint = str(raw_item or "").strip()
        if not endpoint or endpoint in seen:
            continue
        endpoints.append(endpoint)
        seen.add(endpoint)
    return endpoints


def format_endpoint_list(value: Any) -> str:
    return ",".join(split_endpoint_list(value))


def openai_base_to_root(base_url: str) -> str:
    endpoint = str(base_url or "").rstrip("/")
    return endpoint[:-3].rstrip("/") if endpoint.endswith("/v1") else endpoint


def extract_embedding_vector(payload: Any) -> List[float]:
    def candidate(value: Any) -> Optional[List[float]]:
        if isinstance(value, list):
            if value and all(isinstance(item, (int, float)) for item in value):
                return [float(item) for item in value]
            for item in value:
                found = candidate(item)
                if found:
                    return found
        if isinstance(value, dict):
            for key in ("embedding", "data", "vector", "outputs"):
                if key in value:
                    found = candidate(value[key])
                    if found:
                        return found
        return None

    vector = candidate(payload)
    if not vector:
        raise RuntimeError("Pooling embedding service returned no vector.")
    return vector


def request_pooling_embedding(
    *,
    api_key: str,
    base_url: str,
    model: str,
    text: str,
    timeout: float = 30.0,
) -> List[float]:
    url = openai_base_to_root(base_url) + "/pooling"
    payloads = (
        {"model": model, "input": text, "task": "embed"},
        {"model": model, "input": [text], "task": "embed"},
    )
    errors: List[str] = []
    for payload in payloads:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1.0, float(timeout))) as response:
                status = int(getattr(response, "status", 200))
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            errors.append(f"HTTP {exc.code}: {body}")
            continue
        if not (200 <= status < 300):
            errors.append(f"HTTP {status}: {body.decode('utf-8', errors='replace')}")
            continue
        return extract_embedding_vector(json.loads(body.decode("utf-8")))
    raise RuntimeError(f"Pooling embedding service failed at {url}: {'; '.join(errors)}")


def mark_embeddings_api_unhealthy(base_url: str) -> None:
    _EMBEDDINGS_API_UNHEALTHY_ENDPOINTS.add(str(base_url or "").rstrip("/"))


def embeddings_api_marked_unhealthy(base_url: str) -> bool:
    return str(base_url or "").rstrip("/") in _EMBEDDINGS_API_UNHEALTHY_ENDPOINTS


def _configured_policy() -> str:
    return str(
        os.environ.get("TOOL_STAR_SIDECAR_LOAD_BALANCE_POLICY", "")
        or getattr(settings, "sidecar_load_balance_policy", "least_inflight")
        or "least_inflight"
    ).strip().lower()


def _failure_cooldown_seconds() -> float:
    try:
        return max(
            0.0,
            float(
                os.environ.get("TOOL_STAR_SIDECAR_ENDPOINT_FAILURE_COOLDOWN_SECONDS", "")
                or getattr(settings, "sidecar_endpoint_failure_cooldown_seconds", 20.0)
                or 0.0
            ),
        )
    except (TypeError, ValueError):
        return 20.0


def _exception_status_code(exc: BaseException) -> Optional[int]:
    for attr_name in ("status_code", "status", "http_status", "http_status_code"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        for attr_name in ("status_code", "status"):
            value = getattr(response, attr_name, None)
            if isinstance(value, int):
                return value

    return None


def _looks_like_endpoint_failure(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    status_code = _exception_status_code(exc)
    if status_code is not None and (500 <= status_code < 600 or status_code == 429):
        return True
    markers = (
        "connection refused",
        "connection error",
        "api connection",
        "connecterror",
        "read timed out",
        "timed out",
        "server disconnected",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "internalservererror",
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
        "status code: 502",
        "status code: 503",
        "status code: 504",
    )
    return any(marker in message for marker in markers)


class OpenAIEndpointPool:
    """Thread-safe pool over several OpenAI-compatible base URLs.

    The pool does not move a running vLLM process between GPUs. It balances
    individual requests over pre-started sidecar replicas.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_urls: Any,
        namespace: str = "openai_sidecar",
        policy: str = "",
    ) -> None:
        endpoints = split_endpoint_list(base_urls)
        if not endpoints:
            raise ValueError(f"{namespace} requires at least one sidecar endpoint.")
        self.api_key = api_key or "EMPTY"
        self.base_urls = endpoints
        self.namespace = str(namespace or "openai_sidecar")
        self.policy = str(policy or _configured_policy()).strip().lower()
        self._clients: Dict[str, OpenAI] = {}
        self._inflight: Dict[str, int] = {endpoint: 0 for endpoint in self.base_urls}
        self._failure_until: Dict[str, float] = {endpoint: 0.0 for endpoint in self.base_urls}
        self._rr_cursor = os.getpid() % max(1, len(self.base_urls))
        self._lock = threading.Lock()

    @property
    def display_base_urls(self) -> str:
        return ",".join(self.base_urls)

    @property
    def first_base_url(self) -> str:
        return self.base_urls[0]

    def _available_endpoints(self, now: float) -> List[str]:
        available = [
            endpoint
            for endpoint in self.base_urls
            if self._failure_until.get(endpoint, 0.0) <= now
        ]
        return available or list(self.base_urls)

    def _select_locked(self) -> str:
        now = time.monotonic()
        candidates = self._available_endpoints(now)
        policy = self.policy
        if policy in {"round_robin", "rr", "rotate"}:
            endpoint = candidates[self._rr_cursor % len(candidates)]
            self._rr_cursor += 1
            return endpoint

        min_inflight = min(self._inflight.get(endpoint, 0) for endpoint in candidates)
        least_busy = [
            endpoint
            for endpoint in candidates
            if self._inflight.get(endpoint, 0) == min_inflight
        ]
        endpoint = least_busy[self._rr_cursor % len(least_busy)]
        self._rr_cursor += 1
        return endpoint

    def _client_for(self, endpoint: str) -> OpenAI:
        _require_openai_sdk()
        client = self._clients.get(endpoint)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=endpoint)
            self._clients[endpoint] = client
        return client

    def _record_failure(self, endpoint: str, exc: BaseException) -> None:
        if len(self.base_urls) <= 1 or not _looks_like_endpoint_failure(exc):
            return
        cooldown = _failure_cooldown_seconds()
        if cooldown <= 0:
            return
        until = time.monotonic() + cooldown
        status_code = _exception_status_code(exc)
        with self._lock:
            self._failure_until[endpoint] = max(self._failure_until.get(endpoint, 0.0), until)
        logger.warning(
            "[SidecarPool][DIAG] namespace=%s endpoint=%s temporarily cooled down for %.1fs after %s status_code=%s",
            self.namespace,
            endpoint,
            cooldown,
            exc.__class__.__name__,
            status_code if status_code is not None else "n/a",
        )

    @contextmanager
    def borrow(self) -> Iterator[Tuple[OpenAI, str]]:
        with self._lock:
            endpoint = self._select_locked()
            self._inflight[endpoint] = self._inflight.get(endpoint, 0) + 1
            inflight = self._inflight[endpoint]
        logger.debug(
            "[SidecarPool][DIAG] namespace=%s selected=%s inflight=%d policy=%s",
            self.namespace,
            endpoint,
            inflight,
            self.policy,
        )
        try:
            yield self._client_for(endpoint), endpoint
        except Exception as exc:
            self._record_failure(endpoint, exc)
            raise
        finally:
            with self._lock:
                self._inflight[endpoint] = max(0, self._inflight.get(endpoint, 1) - 1)
