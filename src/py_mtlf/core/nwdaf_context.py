import threading
from collections.abc import Callable
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
    containing_nwdaf_process_instance_id: str
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
        self._request_timeout_seconds = request_timeout_seconds
        self._client = client or httpx.Client(timeout=request_timeout_seconds)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._cached = initial

    def open(self) -> None:
        with self._lock:
            if self._owns_client and self._client.is_closed:
                self._client = httpx.Client(timeout=self._request_timeout_seconds)

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
                containing_nwdaf_process_instance_id=self._uuid4(
                    payload["processInstanceId"],
                    "processInstanceId",
                ),
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
    def _uuid4(value: object, field_name: str) -> str:
        parsed = UUID(str(value))
        if parsed.version != 4 or str(parsed) != str(value):
            raise ValueError(f"{field_name} must be a canonical UUIDv4")
        return str(parsed)

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

        return self.verify(context)

    def verify(self, context: NwdafContext) -> CapabilityVerification:
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


@dataclass(frozen=True)
class ContainingNwdafGenerationSnapshot:
    ready: bool
    containing_nwdaf_process_instance_id: str | None
    verification: CapabilityVerification
    resetting: bool = False
    failure: str = ""


class ContainingNwdafGenerationMonitor:
    """Own the one network refresh loop for the containing Go NWDAF generation."""

    def __init__(
        self,
        context_client: NwdafContextClient,
        capability_checker: CapabilityConsistencyChecker,
        *,
        configured_server: bool,
        configured_client: bool,
        refresh_interval_seconds: float = 1.0,
    ) -> None:
        if refresh_interval_seconds <= 0:
            raise ValueError("generation refresh interval must be positive")
        self._context_client = context_client
        self._capability_checker = capability_checker
        self._refresh_interval_seconds = refresh_interval_seconds
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset_callback: Callable[[str], None] = lambda _reason: None
        self._snapshot = ContainingNwdafGenerationSnapshot(
            ready=False,
            containing_nwdaf_process_instance_id=None,
            verification=CapabilityVerification(
                status="unavailable",
                configured_server=configured_server,
                configured_client=configured_client,
                advertised_server=None,
                advertised_client=None,
            ),
            failure="containing NWDAF generation has not been observed",
        )

    def set_reset_callback(self, callback: Callable[[str], None]) -> None:
        with self._condition:
            self._reset_callback = callback

    def open(self) -> None:
        self._stop.clear()
        self.refresh_once()
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="containing-nwdaf-generation-monitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        with self._condition:
            self._thread = None

    def ready(self) -> bool:
        with self._condition:
            return self._snapshot.ready

    def snapshot(self) -> ContainingNwdafGenerationSnapshot:
        with self._condition:
            return self._snapshot

    def refresh_once(self) -> ContainingNwdafGenerationSnapshot:
        try:
            context = self._context_client.get(refresh=True)
            verify = getattr(self._capability_checker, "verify", None)
            verification = (
                verify(context) if callable(verify) else self._capability_checker.check()
            )
            if not verification.ready:
                return self._set_unready(
                    verification,
                    f"containing NWDAF capability verification is {verification.status}",
                    reset_active=False,
                )
        except RuntimeError as error:
            current = self.snapshot()
            configured_server = current.verification.configured_server
            configured_client = current.verification.configured_client
            return self._set_unready(
                CapabilityVerification(
                    status="unavailable",
                    configured_server=configured_server,
                    configured_client=configured_client,
                    advertised_server=None,
                    advertised_client=None,
                ),
                str(error),
                reset_active=True,
            )

        generation = context.containing_nwdaf_process_instance_id
        with self._condition:
            previous = self._snapshot.containing_nwdaf_process_instance_id
            changed = previous is not None and previous != generation
            if not changed:
                self._snapshot = ContainingNwdafGenerationSnapshot(
                    ready=True,
                    containing_nwdaf_process_instance_id=generation,
                    verification=verification,
                )
                self._condition.notify_all()
                return self._snapshot
            self._snapshot = ContainingNwdafGenerationSnapshot(
                ready=False,
                containing_nwdaf_process_instance_id=previous,
                verification=verification,
                resetting=True,
                failure="containing NWDAF process generation changed",
            )
            callback = self._reset_callback
        callback("containing NWDAF process generation changed")
        with self._condition:
            self._snapshot = ContainingNwdafGenerationSnapshot(
                ready=True,
                containing_nwdaf_process_instance_id=generation,
                verification=verification,
            )
            self._condition.notify_all()
            return self._snapshot

    def _set_unready(
        self,
        verification: CapabilityVerification,
        failure: str,
        *,
        reset_active: bool,
    ) -> ContainingNwdafGenerationSnapshot:
        with self._condition:
            previous = self._snapshot.containing_nwdaf_process_instance_id
            should_reset = reset_active and previous is not None
            self._snapshot = ContainingNwdafGenerationSnapshot(
                ready=False,
                containing_nwdaf_process_instance_id=None if should_reset else previous,
                verification=verification,
                resetting=should_reset,
                failure=failure,
            )
            callback = self._reset_callback
        if should_reset:
            callback(f"containing NWDAF context unavailable: {failure}")
            with self._condition:
                self._snapshot = ContainingNwdafGenerationSnapshot(
                    ready=False,
                    containing_nwdaf_process_instance_id=None,
                    verification=verification,
                    failure=failure,
                )
                self._condition.notify_all()
        return self.snapshot()

    def _run(self) -> None:
        while not self._stop.wait(self._refresh_interval_seconds):
            try:
                self.refresh_once()
            except Exception:
                # Reset callbacks are lifecycle code: leave admission closed and
                # let the next bounded refresh retry instead of killing the monitor.
                with self._condition:
                    current = self._snapshot
                    self._snapshot = ContainingNwdafGenerationSnapshot(
                        ready=False,
                        containing_nwdaf_process_instance_id=(
                            current.containing_nwdaf_process_instance_id
                        ),
                        verification=current.verification,
                        failure="containing NWDAF generation reset failed",
                    )
                    self._condition.notify_all()
