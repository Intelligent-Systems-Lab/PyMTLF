import json
import threading
import time
from urllib.parse import urlsplit

import httpx

from py_mtlf.config import ModelMonitorSettings
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.wire.private import SelectedTarget


class NwdafMonitorResolver:
    def __init__(
        self,
        settings: ModelMonitorSettings,
        nwdaf_context: NwdafContextClient,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(
            timeout=settings.discovery_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._cached: dict[str, SelectedTarget | None] = {}
        self._expires_at: dict[str, float] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def invalidate(self, nf_instance_id: str) -> None:
        with self._lock:
            self._cached.pop(nf_instance_id, None)
            self._expires_at.pop(nf_instance_id, None)

    def resolve(self, nf_instance_id: str) -> SelectedTarget | None:
        try:
            context = self._nwdaf_context.get()
        except RuntimeError:
            return None
        if nf_instance_id == context.nf_instance_id:
            return SelectedTarget(
                nfInstanceId=context.nf_instance_id,
                nfServiceInstanceId="containing-nwdaf-mlmodelmonitor",
                serviceName="nnwdaf-mlmodelmonitor",
                apiRoot=context.api_root,
                selectionSource="CONFIGURED",
            )

        now = time.monotonic()
        with self._lock:
            if nf_instance_id in self._cached and now < self._expires_at.get(
                nf_instance_id,
                0.0,
            ):
                return self._cached[nf_instance_id]
        base_uri = context.internal_api_root
        query_entry = {"mlAnalyticsIds": ["UE_COMMUNICATION"]}
        response = self._client.get(
            f"{base_uri}/internal/v1/nrf/nf-instances",
            params={
                "target-nf-type": "NWDAF",
                "requester-nf-type": "NWDAF",
                "target-nf-instance-id": nf_instance_id,
                "service-names": "nnwdaf-mlmodelmonitor",
                "ml-analytics-info-list": json.dumps(
                    [query_entry],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
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
            if not isinstance(profile, dict):
                raise ValueError("NRF discovery returned a malformed NF profile")
            if profile.get("nfInstanceId") != nf_instance_id:
                continue
            if profile.get("nfStatus") not in {"", None, "REGISTERED"}:
                continue
            if not self._supports_analytics(profile):
                continue
            services = [
                (str(service.get("serviceInstanceId", "")), service)
                for service in (profile.get("nfServices") or [])
                if isinstance(service, dict)
            ]
            services.extend(
                (str(service.get("serviceInstanceId") or service_id), service)
                for service_id, service in (profile.get("nfServiceList") or {}).items()
                if isinstance(service, dict)
            )
            for service_id, service in services:
                if (
                    service.get("serviceName") != "nnwdaf-mlmodelmonitor"
                    or service.get("nfServiceStatus") != "REGISTERED"
                ):
                    continue
                api_root = service.get("apiPrefix") or self._derive_root(profile, service)
                if api_root:
                    candidates.append(
                        (
                            str(profile["nfInstanceId"]),
                            service_id,
                            self._normalize_api_root(str(api_root)),
                        )
                    )
        candidates.sort()
        selected = None
        if candidates:
            selected = SelectedTarget(
                nfInstanceId=candidates[0][0],
                nfServiceInstanceId=candidates[0][1],
                serviceName="nnwdaf-mlmodelmonitor",
                apiRoot=candidates[0][2],
                selectionSource="NRF",
            )
        with self._lock:
            self._cached[nf_instance_id] = selected
            self._expires_at[nf_instance_id] = now + validity if validity > 0 else now
        return selected

    @staticmethod
    def _supports_analytics(profile: dict) -> bool:
        infos = []
        if isinstance(profile.get("nwdafInfo"), dict):
            infos.append(profile["nwdafInfo"])
        info_list = profile.get("nwdafInfoList") or {}
        if isinstance(info_list, dict):
            infos.extend(value for value in info_list.values() if isinstance(value, dict))
        return any(
            "UE_COMMUNICATION" in (entry.get("mlAnalyticsIds") or [])
            for info in infos
            for entry in (info.get("mlAnalyticsList") or [])
            if isinstance(entry, dict)
        )

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

    @staticmethod
    def _normalize_api_root(value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("NWDAF monitor API root must be an HTTP(S) URI")
        return value
