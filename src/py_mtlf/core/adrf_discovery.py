import threading
import time
from urllib.parse import urlsplit

import httpx

from py_mtlf.config import AdrfSettings
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.wire.private import SelectedTarget


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
        nwdaf_context: NwdafContextClient,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(timeout=settings.discovery_timeout_seconds)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._cached: dict[str, tuple[SelectedTarget | None, float]] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def resolve(self, required_nf_instance_id: str = "") -> str | None:
        target = self.resolve_data(required_nf_instance_id)
        return target.api_root if target is not None else None

    def resolve_data(self, required_nf_instance_id: str = "") -> SelectedTarget | None:
        return self._resolve_service(
            "nadrf-datamanagement",
            "data-storage-ind",
            required_nf_instance_id,
        )

    def resolve_model(self, required_nf_instance_id: str = "") -> SelectedTarget | None:
        return self._resolve_service(
            "nadrf-mlmodelmanagement",
            "ml-model-storage-ind",
            required_nf_instance_id,
        )

    def invalidate(self, service_name: str | None = None) -> None:
        with self._lock:
            if service_name is None:
                self._cached.clear()
            else:
                for key in tuple(self._cached):
                    if key == service_name or key.startswith(service_name + "|"):
                        self._cached.pop(key, None)

    def _resolve_service(
        self,
        service_name: str,
        capability_indicator: str,
        required_nf_instance_id: str = "",
    ) -> SelectedTarget | None:
        if self._settings.mode == "configured":
            if (
                required_nf_instance_id
                and required_nf_instance_id != self._settings.configured_nf_instance_id
            ):
                return None
            return SelectedTarget(
                nfInstanceId=self._settings.configured_nf_instance_id,
                nfServiceInstanceId=f"configured-{service_name}",
                serviceName=service_name,
                apiRoot=self._settings.configured_endpoint,
                selectionSource="CONFIG",
            )
        now = time.monotonic()
        cache_key = f"{service_name}|{required_nf_instance_id}"
        with self._lock:
            cached = self._cached.get(cache_key)
            if cached is not None and now < cached[1]:
                return cached[0]
        try:
            context = self._nwdaf_context.get()
        except RuntimeError:
            return None
        base = context.internal_api_root
        params = {
            "target-nf-type": "ADRF",
            "requester-nf-type": "NWDAF",
            "service-names": service_name,
            capability_indicator: "true",
        }
        if required_nf_instance_id:
            params["target-nf-instance-id"] = required_nf_instance_id
        response = self._client.get(
            f"{base}/internal/v1/nrf/nf-instances",
            params=params,
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
            if required_nf_instance_id and profile.get("nfInstanceId") != required_nf_instance_id:
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
                    service.get("serviceName") != service_name
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
        selected = (
            SelectedTarget(
                nfInstanceId=candidates[0][0],
                nfServiceInstanceId=(candidates[0][1] or f"{candidates[0][0]}:{service_name}"),
                serviceName=service_name,
                apiRoot=candidates[0][2],
                selectionSource="NRF",
            )
            if candidates
            else None
        )
        with self._lock:
            self._cached[cache_key] = (
                selected,
                now + validity if validity > 0 else now,
            )
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
