import threading
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

import httpx


@dataclass(frozen=True)
class NwdafContext:
    nf_instance_id: str
    api_root: str
    internal_api_root: str


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
        payload = response.json()
        try:
            context = NwdafContext(
                nf_instance_id=str(UUID(str(payload["nfInstanceId"]))),
                api_root=self._validate_origin(str(payload["apiRoot"])),
                internal_api_root=self._validate_origin(str(payload["internalApiRoot"])),
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
