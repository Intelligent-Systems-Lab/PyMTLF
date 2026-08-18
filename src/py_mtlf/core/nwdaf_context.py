import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx


class FLCapabilityType(StrEnum):
    SERVER = "FL_SERVER"
    CLIENT = "FL_CLIENT"
    SERVER_AND_CLIENT = "FL_SERVER_AND_CLIENT"


@dataclass(frozen=True)
class MLAnalyticsCapability:
    ml_analytics_ids: tuple[str, ...]
    fl_capability_type: FLCapabilityType


@dataclass(frozen=True)
class NwdafContext:
    nf_instance_id: str
    api_root: str
    internal_api_root: str
    ml_analytics_capabilities: tuple[MLAnalyticsCapability, ...] = ()

    @property
    def advertised_server(self) -> bool:
        return any(
            capability.fl_capability_type
            in {FLCapabilityType.SERVER, FLCapabilityType.SERVER_AND_CLIENT}
            for capability in self.ml_analytics_capabilities
        )

    @property
    def advertised_client(self) -> bool:
        return any(
            capability.fl_capability_type
            in {FLCapabilityType.CLIENT, FLCapabilityType.SERVER_AND_CLIENT}
            for capability in self.ml_analytics_capabilities
        )


@dataclass(frozen=True)
class CapabilityVerification:
    status: Literal["verified", "mismatch", "unavailable"]
    configured_server: bool
    configured_client: bool
    advertised_server: bool | None
    advertised_client: bool | None

    @property
    def ready(self) -> bool:
        return self.status == "verified"


class NwdafContextClient:
    """Read immutable containing-NWDAF identity and base URIs on demand."""

    def __init__(
        self,
        internal_api_root: str,
        request_timeout_seconds: float,
        *,
        client: httpx.Client | None = None,
        initial: NwdafContext | None = None,
    ) -> None:
        self._internal_api_root = self._validate_origin(internal_api_root)
        if request_timeout_seconds <= 0:
            raise ValueError("containing NWDAF request timeout must be positive")
        self._client = client or httpx.Client(timeout=request_timeout_seconds)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._cached = initial

    def get(self, *, refresh: bool = False) -> NwdafContext:
        with self._lock:
            if self._cached is not None and not refresh:
                return self._cached
        try:
            response = self._client.get(
                self._internal_api_root + "/internal/v1/nwdaf-context",
                follow_redirects=False,
            )
        except httpx.HTTPError as error:
            raise RuntimeError("containing NWDAF context is unavailable") from error
        if response.status_code != 200:
            raise RuntimeError(
                f"containing NWDAF context returned status {response.status_code}"
            )
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("context response must be an object")
            context = NwdafContext(
                nf_instance_id=str(UUID(str(payload["nfInstanceId"]))),
                api_root=self._validate_origin(str(payload["apiRoot"])),
                internal_api_root=self._validate_origin(str(payload["internalApiRoot"])),
                ml_analytics_capabilities=self._parse_capabilities(
                    payload.get("mlAnalyticsCapabilities", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("containing NWDAF context is malformed") from error
        with self._lock:
            self._cached = context
        return context

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _parse_capabilities(value: object) -> tuple[MLAnalyticsCapability, ...]:
        if not isinstance(value, list):
            raise TypeError("mlAnalyticsCapabilities must be an array")
        capabilities = []
        for entry in value:
            if not isinstance(entry, dict):
                raise TypeError("mlAnalyticsCapabilities entry must be an object")
            analytics_ids = entry.get("mlAnalyticsIds")
            if not isinstance(analytics_ids, list) or not analytics_ids:
                raise TypeError("mlAnalyticsIds must be a non-empty array")
            if any(
                not isinstance(analytics_id, str)
                or not analytics_id
                or analytics_id.strip() != analytics_id
                for analytics_id in analytics_ids
            ):
                raise TypeError("mlAnalyticsIds must contain non-empty strings")
            raw_capability = entry.get("flCapabilityType")
            if not isinstance(raw_capability, str):
                raise TypeError("flCapabilityType must be a string")
            capabilities.append(
                MLAnalyticsCapability(
                    ml_analytics_ids=tuple(analytics_ids),
                    fl_capability_type=FLCapabilityType(raw_capability),
                )
            )
        return tuple(capabilities)

    @staticmethod
    def _validate_origin(value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("containing NWDAF API root must be an HTTP(S) origin")
        return normalized


class CapabilityConsistencyChecker:
    def __init__(
        self,
        context_client: NwdafContextClient,
        *,
        configured_server: bool,
        configured_client: bool,
    ) -> None:
        self._context_client = context_client
        self._configured_server = configured_server
        self._configured_client = configured_client

    def check(self) -> CapabilityVerification:
        try:
            context = self._context_client.get(refresh=True)
        except RuntimeError:
            return CapabilityVerification(
                status="unavailable",
                configured_server=self._configured_server,
                configured_client=self._configured_client,
                advertised_server=None,
                advertised_client=None,
            )

        advertised_server = context.advertised_server
        advertised_client = context.advertised_client
        matches = (
            self._configured_server,
            self._configured_client,
        ) == (advertised_server, advertised_client)
        return CapabilityVerification(
            status="verified" if matches else "mismatch",
            configured_server=self._configured_server,
            configured_client=self._configured_client,
            advertised_server=advertised_server,
            advertised_client=advertised_client,
        )
