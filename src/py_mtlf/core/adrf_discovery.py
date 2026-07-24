import threading
import time
from urllib.parse import urlsplit

import httpx

from py_mtlf.config import AdrfSettings
from py_mtlf.core.sync_projection import SyncProjection


def normalize_api_root(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("ADRF API root has an invalid port") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("ADRF API root must be an HTTP(S) origin")
    return value


class AdrfResolver:
    def __init__(
        self,
        settings: AdrfSettings,
        projection: SyncProjection,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._projection = projection
        self._client = client or httpx.Client(timeout=settings.discovery_timeout_seconds)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._cached: str | None = None
        self._expires_at = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def resolve(self) -> str | None:
        if self._settings.mode == "configured":
            return self._settings.configured_endpoint
        now = time.monotonic()
        with self._lock:
            if self._cached is not None and now < self._expires_at:
                return self._cached
        snapshot = self._projection.snapshot()
        if snapshot is None:
            return None
        base = snapshot.containing_nwdaf.internal_callback_base_uri.rstrip("/")
        response = self._client.get(
            f"{base}/internal/v1/nrf/nf-instances",
            params={
                "target-nf-type": "ADRF",
                "requester-nf-type": "NWDAF",
                "service-names": "nadrf-datamanagement",
            },
        )
        response.raise_for_status()
        result = response.json()
        validity = result.get("validityPeriod")
        instances = result.get("nfInstances")
        if not isinstance(validity, int) or validity < 0 or not isinstance(instances, list):
            raise ValueError("NRF discovery returned a malformed SearchResult")
        candidates: list[tuple[str, str, str]] = []
        for profile in instances:
            if profile.get("nfStatus") not in {"", None, "REGISTERED"}:
                continue
            services = [
                (str(service.get("serviceInstanceId", "")), service)
                for service in (profile.get("nfServices") or [])
            ]
            services.extend(
                (str(service.get("serviceInstanceId") or service_id), service)
                for service_id, service in (profile.get("nfServiceList") or {}).items()
            )
            for service_id, service in services:
                if (
                    service.get("serviceName") != "nadrf-datamanagement"
                    or service.get("nfServiceStatus") != "REGISTERED"
                ):
                    continue
                api_root = service.get("apiPrefix") or self._derive_root(profile, service)
                if api_root:
                    candidates.append(
                        (
                            str(profile.get("nfInstanceId", "")),
                            service_id,
                            normalize_api_root(api_root),
                        )
                    )
        candidates.sort()
        selected = candidates[0][2] if candidates else None
        with self._lock:
            self._cached = selected
            self._expires_at = now + validity if validity > 0 else now
        return selected

    @staticmethod
    def _derive_root(profile: dict, service: dict) -> str:
        scheme = service.get("scheme")
        host = service.get("fqdn") or profile.get("fqdn")
        port = None
        endpoints = service.get("ipEndPoints") or []
        if not host and endpoints:
            host = endpoints[0].get("ipv4Address") or endpoints[0].get("ipv6Address")
            port = endpoints[0].get("port")
        if not host:
            addresses = profile.get("ipv4Addresses") or []
            host = addresses[0] if addresses else ""
        if not host or scheme not in {"http", "https"}:
            return ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"{scheme}://{host}{f':{port}' if port else ''}"
