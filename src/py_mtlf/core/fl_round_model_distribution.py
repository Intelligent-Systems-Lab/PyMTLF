from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from urllib.parse import quote, unquote, urlsplit

import httpx

from py_mtlf.config import AdrfSettings
from py_mtlf.core.adrf_discovery import AdrfResolver
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.wire.adrf import (
    AllowedConsumer,
    MLModelAddress,
    MLModelInfo,
    NadrfMLModelStoreRecord,
)
from py_mtlf.wire.ml_model import MLModelAdrf

_STORED = "ML_MODEL_FILE_STORED_IN_ADRF"
_MAX_SIGNED_INT64 = (1 << 63) - 1

logger = logging.getLogger(__name__)


class RoundModelDistributionError(RuntimeError):
    """The temporary Root round model could not complete its ADRF lifecycle."""


@dataclass(frozen=True)
class RoundModelRecord:
    ml_correlation_id: str
    round_indicator: int
    model_unique_id: int
    adrf_instance_id: str
    adrf_api_root: str
    store_transaction_id: str
    resource_location: str
    model_url: str
    model_size: int
    source_artifact: ArtifactMetadata
    allowed_consumer_ids: tuple[str, ...]

    @property
    def wire_reference(self) -> MLModelAdrf:
        return MLModelAdrf(
            adrfId=self.adrf_instance_id,
            storTransId=self.store_transaction_id,
        )


@dataclass(frozen=True)
class RetrievedRoundModel:
    model_unique_id: int
    model_url: str
    model_size: int
    adrf_instance_id: str
    store_transaction_id: str


class RoundModelDistribution:
    """Owns temporary Root global-model records for active upper-tier rounds."""

    def __init__(
        self,
        settings: AdrfSettings,
        resolver: AdrfResolver,
        nwdaf_context: NwdafContextClient,
        *,
        client: httpx.Client | None = None,
        model_id_source: Callable[[], int] | None = None,
    ) -> None:
        self._resolver = resolver
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._model_id_source = model_id_source or (
            lambda: secrets.randbelow(_MAX_SIGNED_INT64 + 1)
        )
        self._lock = threading.RLock()
        self._records: dict[tuple[str, int], RoundModelRecord] = {}
        self._allocated_model_ids: set[int] = set()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def abort_generation(self, _reason: str) -> None:
        failures = []
        for record in self.snapshot():
            try:
                self.cleanup(record.ml_correlation_id, record.round_indicator)
            except RoundModelDistributionError as error:
                failures.append(error)
        if failures:
            raise RoundModelDistributionError(
                f"failed to clean {len(failures)} temporary ADRF round record(s)"
            ) from failures[0]

    def snapshot(self) -> tuple[RoundModelRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def store(
        self,
        *,
        ml_correlation_id: str,
        round_indicator: int,
        artifact: ArtifactMetadata,
        allowed_consumer_ids: Iterable[str],
    ) -> RoundModelRecord:
        key = _round_key(ml_correlation_id, round_indicator)
        consumers = _consumer_ids(allowed_consumer_ids)
        with self._lock:
            if key in self._records:
                raise RoundModelDistributionError("round model record already exists")
            model_unique_id = self._allocate_model_id_locked()

        target = self._resolver.resolve_model()
        if target is None:
            self._release_model_id(model_unique_id)
            raise RoundModelDistributionError(
                "no ADRF with ML model storage capability is available"
            )
        owner = self._nwdaf_context.get().nf_instance_id
        request_record = _request_record(
            owner_nf_instance_id=owner,
            model_unique_id=model_unique_id,
            model_url=artifact.url,
            model_size=artifact.size_bytes,
            consumers=consumers,
        )
        try:
            response = self._client.post(
                _go_model_records_url(self._nwdaf_context),
                headers={"Target-Api-Root": target.api_root},
                json=request_record.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
        except httpx.HTTPError as error:
            self._release_model_id(model_unique_id)
            raise RoundModelDistributionError("ADRF round model store transport failed") from error
        if response.status_code != 201:
            self._release_model_id(model_unique_id)
            raise RoundModelDistributionError(
                f"ADRF round model store returned status {response.status_code}"
            )
        location = response.headers.get("Location", "")
        if not location:
            self._release_model_id(model_unique_id)
            raise RoundModelDistributionError("ADRF round model store omitted Location")
        try:
            response_record = NadrfMLModelStoreRecord.model_validate(response.json())
            store_transaction_id = _store_transaction_id(location, target.api_root)
            info = _validate_record(
                response_record,
                owner_nf_instance_id=owner,
                model_unique_id=model_unique_id,
                model_size=artifact.size_bytes,
                consumers=consumers,
                store_transaction_id=store_transaction_id,
                require_store_result=True,
            )
        except (TypeError, ValueError) as error:
            self._release_model_id(model_unique_id)
            raise RoundModelDistributionError(
                "ADRF round model store returned an inconsistent record"
            ) from error
        record = RoundModelRecord(
            ml_correlation_id=key[0],
            round_indicator=key[1],
            model_unique_id=model_unique_id,
            adrf_instance_id=target.nf_instance_id,
            adrf_api_root=target.api_root,
            store_transaction_id=store_transaction_id,
            resource_location=location.rstrip("/"),
            model_url=str(info.model_file_address.model_url),
            model_size=info.model_storage_size,
            source_artifact=artifact,
            allowed_consumer_ids=consumers,
        )
        with self._lock:
            if key in self._records:
                self._allocated_model_ids.discard(model_unique_id)
                raise RoundModelDistributionError("round model record was concurrently replaced")
            self._records[key] = record
        logger.info(
            "Stored temporary ADRF round model ml_corre_id=%s round_ind=%s "
            "adrf_id=%s store_trans_id=%s consumers=%s",
            record.ml_correlation_id,
            record.round_indicator,
            record.adrf_instance_id,
            record.store_transaction_id,
            ",".join(record.allowed_consumer_ids),
        )
        return record

    def update_consumers(
        self,
        ml_correlation_id: str,
        round_indicator: int,
        allowed_consumer_ids: Iterable[str],
    ) -> RoundModelRecord:
        key = _round_key(ml_correlation_id, round_indicator)
        consumers = _consumer_ids(allowed_consumer_ids)
        with self._lock:
            current = self._records.get(key)
        if current is None:
            raise RoundModelDistributionError("round model record was not found")
        owner = self._nwdaf_context.get().nf_instance_id
        request_record = _request_record(
            owner_nf_instance_id=owner,
            model_unique_id=current.model_unique_id,
            model_url=current.model_url,
            model_size=current.model_size,
            consumers=consumers,
        )
        try:
            response = self._client.put(
                _go_model_record_url(self._nwdaf_context, current.store_transaction_id),
                headers={"Target-Api-Root": current.adrf_api_root},
                json=request_record.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
        except httpx.HTTPError as error:
            raise RoundModelDistributionError("ADRF allowlist update transport failed") from error
        if response.status_code not in {200, 204}:
            raise RoundModelDistributionError(
                f"ADRF allowlist update returned status {response.status_code}"
            )
        if response.status_code == 200:
            try:
                response_record = NadrfMLModelStoreRecord.model_validate(response.json())
                _validate_record(
                    response_record,
                    owner_nf_instance_id=owner,
                    model_unique_id=current.model_unique_id,
                    model_size=current.model_size,
                    consumers=consumers,
                    store_transaction_id=current.store_transaction_id,
                    require_store_result=False,
                )
            except (TypeError, ValueError) as error:
                raise RoundModelDistributionError(
                    "ADRF allowlist update returned an inconsistent record"
                ) from error
        updated = replace(current, allowed_consumer_ids=consumers)
        with self._lock:
            if self._records.get(key) is not current:
                raise RoundModelDistributionError("round model record changed during update")
            self._records[key] = updated
        return updated

    def cleanup(self, ml_correlation_id: str, round_indicator: int) -> None:
        key = _round_key(ml_correlation_id, round_indicator)
        with self._lock:
            current = self._records.get(key)
        if current is None:
            return
        try:
            response = self._client.delete(
                _go_model_record_url(self._nwdaf_context, current.store_transaction_id),
                headers={"Target-Api-Root": current.adrf_api_root},
            )
        except httpx.HTTPError as error:
            raise RoundModelDistributionError("ADRF round model delete transport failed") from error
        if response.status_code not in {200, 204, 404}:
            raise RoundModelDistributionError(
                f"ADRF round model delete returned status {response.status_code}"
            )
        with self._lock:
            if self._records.get(key) is current:
                self._records.pop(key, None)
                self._allocated_model_ids.discard(current.model_unique_id)
        logger.info(
            "Deleted temporary ADRF round model ml_corre_id=%s round_ind=%s "
            "adrf_id=%s store_trans_id=%s",
            current.ml_correlation_id,
            current.round_indicator,
            current.adrf_instance_id,
            current.store_transaction_id,
        )

    def retrieve(
        self,
        *,
        reference: MLModelAdrf,
        model_unique_id: int,
        consumer_nf_instance_id: str,
    ) -> RetrievedRoundModel:
        if reference.adrf_id is None or not reference.storage_transaction_id:
            raise RoundModelDistributionError(
                "round model reference requires adrfId and storTransId"
            )
        target = self._resolver.resolve_model(reference.adrf_id)
        if target is None or target.nf_instance_id != reference.adrf_id:
            raise RoundModelDistributionError("referenced ADRF could not be resolved exactly")
        try:
            response = self._client.get(
                _go_model_records_url(self._nwdaf_context),
                headers={"Target-Api-Root": target.api_root},
                params={"store-trans-id": reference.storage_transaction_id},
            )
        except httpx.HTTPError as error:
            raise RoundModelDistributionError(
                "ADRF round model retrieval transport failed"
            ) from error
        if response.status_code != 200:
            raise RoundModelDistributionError(
                f"ADRF round model retrieval returned status {response.status_code}"
            )
        try:
            record = NadrfMLModelStoreRecord.model_validate(response.json())
            info = _validate_retrieved_record(
                record,
                model_unique_id=model_unique_id,
                consumer_nf_instance_id=consumer_nf_instance_id,
                store_transaction_id=reference.storage_transaction_id,
            )
        except (TypeError, ValueError) as error:
            raise RoundModelDistributionError(
                "ADRF round model retrieval returned an inconsistent record"
            ) from error
        retrieved = RetrievedRoundModel(
            model_unique_id=model_unique_id,
            model_url=str(info.model_file_address.model_url),
            model_size=info.model_storage_size,
            adrf_instance_id=target.nf_instance_id,
            store_transaction_id=reference.storage_transaction_id,
        )
        logger.info(
            "Retrieved temporary ADRF round model consumer_nf_instance_id=%s "
            "adrf_id=%s store_trans_id=%s model_unique_id=%s",
            consumer_nf_instance_id,
            retrieved.adrf_instance_id,
            retrieved.store_transaction_id,
            retrieved.model_unique_id,
        )
        return retrieved

    def _allocate_model_id_locked(self) -> int:
        for _ in range(1024):
            model_id = self._model_id_source()
            if not isinstance(model_id, int) or isinstance(model_id, bool):
                raise RoundModelDistributionError("model ID source returned a non-integer")
            if model_id < 0 or model_id > _MAX_SIGNED_INT64:
                raise RoundModelDistributionError("model ID source returned an out-of-range value")
            if model_id not in self._allocated_model_ids:
                self._allocated_model_ids.add(model_id)
                return model_id
        raise RoundModelDistributionError("could not allocate a unique active round model ID")

    def _release_model_id(self, model_unique_id: int) -> None:
        with self._lock:
            self._allocated_model_ids.discard(model_unique_id)


def _round_key(ml_correlation_id: str, round_indicator: int) -> tuple[str, int]:
    normalized = ml_correlation_id.strip()
    if not normalized:
        raise ValueError("mlCorreId must not be blank")
    if round_indicator < 0:
        raise ValueError("roundInd must be non-negative")
    return normalized, round_indicator


def _consumer_ids(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({value.strip() for value in values if value.strip()}))
    if not normalized:
        raise ValueError("at least one allowed consumer is required")
    return normalized


def _request_record(
    *,
    owner_nf_instance_id: str,
    model_unique_id: int,
    model_url: str,
    model_size: int,
    consumers: tuple[str, ...],
) -> NadrfMLModelStoreRecord:
    return NadrfMLModelStoreRecord(
        nfInstanceId=owner_nf_instance_id,
        mlModelInfo=[
            MLModelInfo(
                modelUniqueId=model_unique_id,
                mlFileAddr=MLModelAddress(mLModelUrl=model_url),
                mlStorageSize=model_size,
                allowConsumerList=[
                    AllowedConsumer(nfInstanceId=consumer) for consumer in consumers
                ],
            )
        ],
    )


def _validate_record(
    record: NadrfMLModelStoreRecord,
    *,
    owner_nf_instance_id: str,
    model_unique_id: int,
    model_size: int,
    consumers: tuple[str, ...],
    store_transaction_id: str,
    require_store_result: bool,
) -> MLModelInfo:
    if record.nf_instance_id != owner_nf_instance_id:
        raise ValueError("record owner changed")
    info = record.ml_model_info[0]
    if info.model_unique_id != model_unique_id:
        raise ValueError("modelUniqueId changed")
    if info.model_storage_size != model_size:
        raise ValueError("mlStorageSize changed")
    returned_consumers = tuple(
        sorted(
            consumer.nf_instance_id
            for consumer in info.allowed_consumers
            if consumer.nf_instance_id is not None
        )
    )
    if returned_consumers != consumers or len(returned_consumers) != len(info.allowed_consumers):
        raise ValueError("allowConsumerList changed")
    _validate_model_url(info, store_transaction_id)
    if require_store_result:
        result = record.model_store_result
        if (
            result is None
            or result.model_unique_id != model_unique_id
            or result.store_result != _STORED
        ):
            raise ValueError("modelStoreResult did not confirm storage")
    return info


def _validate_retrieved_record(
    record: NadrfMLModelStoreRecord,
    *,
    model_unique_id: int,
    consumer_nf_instance_id: str,
    store_transaction_id: str,
) -> MLModelInfo:
    info = record.ml_model_info[0]
    if info.model_unique_id != model_unique_id:
        raise ValueError("modelUniqueId does not match the subscription")
    _validate_model_url(info, store_transaction_id)
    allowed = {
        consumer.nf_instance_id
        for consumer in info.allowed_consumers
        if consumer.nf_instance_id is not None
    }
    if consumer_nf_instance_id not in allowed:
        raise ValueError("consumer is absent from allowConsumerList")
    return info


def _validate_model_url(info: MLModelInfo, store_transaction_id: str) -> None:
    model_url = str(info.model_file_address.model_url or "")
    parsed = urlsplit(model_url)
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or len(segments) < 2
        or segments[-2:] != [store_transaction_id, "model"]
    ):
        raise ValueError("mlFileAddr does not identify the requested ADRF record")


def _store_transaction_id(location: str, target_api_root: str) -> str:
    parsed = urlsplit(location)
    target = urlsplit(target_api_root)
    if parsed.scheme != target.scheme or parsed.netloc != target.netloc:
        raise ValueError("Location is outside the selected ADRF")
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if len(segments) < 2 or segments[-2] != "mlmodel-store-records":
        raise ValueError("Location does not identify an ML model store record")
    store_transaction_id = segments[-1].strip()
    if not store_transaction_id:
        raise ValueError("Location has no store transaction ID")
    return store_transaction_id


def _go_model_records_url(nwdaf_context: NwdafContextClient) -> str:
    return (
        nwdaf_context.get().internal_api_root
        + "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records"
    )


def _go_model_record_url(
    nwdaf_context: NwdafContextClient,
    store_transaction_id: str,
) -> str:
    return _go_model_records_url(nwdaf_context) + "/" + quote(store_transaction_id, safe="")
