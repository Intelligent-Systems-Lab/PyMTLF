from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from urllib.parse import urlsplit

import httpx

from py_mtlf.config import FederatedLearningSettings
from py_mtlf.core.fl_hierarchy import normalize_nf_instance_id
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.core.workloads import IMAGE_MODEL_INTEROPERABILITY
from py_mtlf.wire.ml_model_training import (
    InvalidParameter,
    NwdafMLModelTrainSubsc,
    RequirementsError,
)
from py_mtlf.wire.private import SelectedTarget


class HierarchyDiscoveryError(RuntimeError):
    pass


class HierarchyNodeRole(StrEnum):
    BRANCH = "BRANCH"
    LEAF = "LEAF"


@dataclass(frozen=True)
class ResolvedHierarchyNode:
    nf_instance_id: str
    role: HierarchyNodeRole
    target: SelectedTarget
    discovery_scope: HierarchyDiscoveryScope | None = None
    observed_at: datetime | None = None
    valid_until: datetime | None = None


@dataclass(frozen=True)
class HierarchyDiscoveryScope:
    containing_nf_instance_id: str
    role: HierarchyNodeRole
    ml_event: str
    model_interoperability: str
    tracking_areas: tuple[tuple[str, str, str], ...]
    target_nf_instance_id: str | None = None


@dataclass(frozen=True)
class HierarchyDiscoverySnapshot:
    scope: HierarchyDiscoveryScope
    nodes: tuple[ResolvedHierarchyNode, ...]
    observed_at: datetime
    validity_period: int
    returned_count: int
    complete_nf_instance_count: int | None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if (
            not isinstance(self.validity_period, int)
            or isinstance(self.validity_period, bool)
            or self.validity_period < 0
        ):
            raise ValueError("validity_period must be a non-negative integer")
        if (
            not isinstance(self.returned_count, int)
            or isinstance(self.returned_count, bool)
            or self.returned_count < len(self.nodes)
        ):
            raise ValueError("returned_count must include all resolved nodes")
        if (
            self.complete_nf_instance_count is not None
            and (
                not isinstance(self.complete_nf_instance_count, int)
                or isinstance(self.complete_nf_instance_count, bool)
                or self.complete_nf_instance_count < self.returned_count
            )
        ):
            raise ValueError("complete_nf_instance_count is inconsistent")

    @property
    def valid_until(self) -> datetime:
        return self.observed_at + timedelta(seconds=self.validity_period)

    @property
    def is_complete(self) -> bool:
        return (
            self.complete_nf_instance_count is None
            or self.complete_nf_instance_count == self.returned_count
        )


@dataclass(frozen=True)
class _ParsedSearchResult:
    profiles: tuple[object, ...]
    validity_period: int
    returned_count: int
    complete_nf_instance_count: int | None


@dataclass(frozen=True)
class HierarchyDiscoveryRequirements:
    ml_event: str
    model_interoperability: str
    tracking_areas: tuple[dict[str, object], ...]


class HierarchyNodeResolver:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        nwdaf_context: NwdafContextClient,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._clock = clock or (lambda: datetime.now(UTC))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def resolve(
        self,
        *,
        nf_instance_id: str,
        role: HierarchyNodeRole,
        ml_event: str,
        model_interoperability: str,
    ) -> ResolvedHierarchyNode:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        if not isinstance(role, HierarchyNodeRole):
            raise ValueError("role must be a HierarchyNodeRole")
        if not ml_event.strip() or ml_event.strip() != ml_event:
            raise ValueError("ml_event must be a non-empty canonical value")
        if not model_interoperability.strip():
            raise ValueError("model_interoperability must not be blank")
        profile_interoperability = _profile_interoperability(model_interoperability)

        capability = (
            "FL_SERVER_AND_CLIENT"
            if role is HierarchyNodeRole.BRANCH
            else "FL_CLIENT"
        )
        query_entry = _query_entry(
            ml_event=ml_event,
            capability=capability,
            model_interoperability=profile_interoperability,
            tracking_areas=(),
        )
        try:
            context = self._nwdaf_context.get()
            result = _parse_search_result(
                self._search(
                    context.internal_api_root,
                    query_entry=query_entry,
                    target_nf_instance_id=nf_id,
                )
            )
        except (httpx.HTTPError, ValueError) as error:
            raise HierarchyDiscoveryError(
                f"NRF discovery failed for {role.value} {nf_id}"
            ) from error
        observed_at = _observed_at(self._clock())
        scope = _scope(
            containing_nf_instance_id=context.nf_instance_id,
            role=role,
            ml_event=ml_event,
            model_interoperability=model_interoperability,
            tracking_areas=(),
            target_nf_instance_id=nf_id,
        )

        exact_profiles = []
        for profile in result.profiles:
            if not isinstance(profile, dict):
                raise HierarchyDiscoveryError("NRF discovery returned a malformed NF profile")
            if profile.get("nfInstanceId") == nf_id:
                exact_profiles.append(profile)
        if len(exact_profiles) != 1:
            raise HierarchyDiscoveryError(
                f"NRF discovery did not uniquely resolve {role.value} {nf_id}"
            )
        profile = exact_profiles[0]
        if profile.get("nfStatus") != "REGISTERED":
            raise HierarchyDiscoveryError(f"{role.value} {nf_id} is not REGISTERED")
        if not _supports_requirement(
            profile,
            role=role,
            ml_event=ml_event,
            model_interoperability=profile_interoperability,
        ):
            raise HierarchyDiscoveryError(
                f"{role.value} {nf_id} does not advertise the required FL capability"
            )

        targets = _service_targets(profile, nf_id)
        if len(targets) != 1:
            raise HierarchyDiscoveryError(
                f"NRF discovery did not uniquely resolve the Training service for {nf_id}"
            )
        return ResolvedHierarchyNode(
            nf_instance_id=nf_id,
            role=role,
            target=targets[0],
            discovery_scope=scope,
            observed_at=observed_at,
            valid_until=observed_at + timedelta(seconds=result.validity_period),
        )

    def discover(
        self,
        *,
        role: HierarchyNodeRole,
        ml_event: str,
        model_interoperability: str,
        tracking_areas: Iterable[Mapping[str, object]],
        excluded_nf_instance_ids: Iterable[str] = (),
    ) -> HierarchyDiscoverySnapshot:
        if not isinstance(role, HierarchyNodeRole):
            raise ValueError("role must be a HierarchyNodeRole")
        if not ml_event.strip() or ml_event.strip() != ml_event:
            raise ValueError("ml_event must be a non-empty canonical value")
        if not model_interoperability.strip():
            raise ValueError("model_interoperability must not be blank")
        profile_interoperability = _profile_interoperability(model_interoperability)
        tais = _normalize_tais(tracking_areas)
        if not tais:
            raise ValueError("bounded hierarchy discovery requires tracking areas")
        excluded = {normalize_nf_instance_id(value) for value in excluded_nf_instance_ids}
        capability = (
            "FL_SERVER_AND_CLIENT"
            if role is HierarchyNodeRole.BRANCH
            else "FL_CLIENT"
        )
        query_entry = _query_entry(
            ml_event=ml_event,
            capability=capability,
            model_interoperability=profile_interoperability,
            tracking_areas=tais,
        )
        try:
            context = self._nwdaf_context.get()
            excluded.add(normalize_nf_instance_id(context.nf_instance_id))
            result = _parse_search_result(
                self._search(context.internal_api_root, query_entry=query_entry)
            )
        except (httpx.HTTPError, ValueError) as error:
            raise HierarchyDiscoveryError(
                f"NRF list discovery failed for {role.value} candidates"
            ) from error

        observed_at = _observed_at(self._clock())
        scope = _scope(
            containing_nf_instance_id=context.nf_instance_id,
            role=role,
            ml_event=ml_event,
            model_interoperability=model_interoperability,
            tracking_areas=tais,
        )
        valid_until = observed_at + timedelta(seconds=result.validity_period)
        resolved: list[ResolvedHierarchyNode] = []
        seen: set[str] = set()
        for profile in result.profiles:
            if not isinstance(profile, dict):
                raise HierarchyDiscoveryError("NRF discovery returned a malformed NF profile")
            raw_id = profile.get("nfInstanceId")
            try:
                nf_id = normalize_nf_instance_id(str(raw_id))
            except ValueError:
                continue
            if nf_id in excluded or nf_id in seen or profile.get("nfStatus") != "REGISTERED":
                continue
            if not _supports_requirement(
                profile,
                role=role,
                ml_event=ml_event,
                model_interoperability=profile_interoperability,
                tracking_areas=tais,
            ):
                continue
            targets = _service_targets(profile, nf_id)
            if len(targets) != 1:
                continue
            seen.add(nf_id)
            resolved.append(
                ResolvedHierarchyNode(
                    nf_instance_id=nf_id,
                    role=role,
                    target=targets[0],
                    discovery_scope=scope,
                    observed_at=observed_at,
                    valid_until=valid_until,
                )
            )
        return HierarchyDiscoverySnapshot(
            scope=scope,
            nodes=tuple(
                sorted(
                    resolved,
                    key=lambda item: (
                        item.nf_instance_id,
                        item.target.nf_service_instance_id,
                        item.target.api_root,
                    ),
                )
            ),
            observed_at=observed_at,
            validity_period=result.validity_period,
            returned_count=result.returned_count,
            complete_nf_instance_count=result.complete_nf_instance_count,
        )

    def discover_for_subscription(
        self,
        value: NwdafMLModelTrainSubsc,
        *,
        role: HierarchyNodeRole,
        excluded_nf_instance_ids: Iterable[str] = (),
    ) -> HierarchyDiscoverySnapshot:
        try:
            requirements = hierarchy_discovery_requirements(value)
        except ValueError as error:
            raise RequirementsError(
                [
                    InvalidParameter(
                        "mLEventSubscs[0].mLEventFilter.networkArea.tais",
                        str(error),
                    )
                ]
            ) from error
        return self.discover(
            role=role,
            ml_event=requirements.ml_event,
            model_interoperability=requirements.model_interoperability,
            tracking_areas=requirements.tracking_areas,
            excluded_nf_instance_ids=excluded_nf_instance_ids,
        )

    def discovery_scope_for_subscription(
        self,
        value: NwdafMLModelTrainSubsc,
        *,
        role: HierarchyNodeRole,
    ) -> HierarchyDiscoveryScope:
        try:
            requirements = hierarchy_discovery_requirements(value)
        except ValueError as error:
            raise RequirementsError(
                [
                    InvalidParameter(
                        "mLEventSubscs[0].mLEventFilter.networkArea.tais",
                        str(error),
                    )
                ]
            ) from error
        context = self._nwdaf_context.get()
        return _scope(
            containing_nf_instance_id=context.nf_instance_id,
            role=role,
            ml_event=requirements.ml_event,
            model_interoperability=requirements.model_interoperability,
            tracking_areas=requirements.tracking_areas,
        )

    def _search(
        self,
        internal_api_root: str,
        *,
        query_entry: dict[str, object],
        target_nf_instance_id: str | None = None,
    ) -> object:
        params = {
            "target-nf-type": "NWDAF",
            "requester-nf-type": "NWDAF",
            "service-names": "nnwdaf-mlmodeltraining",
            "ml-analytics-info-list": json.dumps(
                [query_entry],
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        if target_nf_instance_id is not None:
            params["target-nf-instance-id"] = target_nf_instance_id
        response = self._client.get(
            internal_api_root + "/internal/v1/nrf/nf-instances",
            params=params,
        )
        response.raise_for_status()
        return response.json()


def _supports_requirement(
    profile: dict,
    *,
    role: HierarchyNodeRole,
    ml_event: str,
    model_interoperability: str,
    tracking_areas: tuple[dict[str, object], ...] = (),
) -> bool:
    accepted_capabilities = (
        {"FL_SERVER_AND_CLIENT"}
        if role is HierarchyNodeRole.BRANCH
        else {"FL_CLIENT", "FL_SERVER_AND_CLIENT"}
    )
    for info in _nwdaf_infos(profile):
        entries = info.get("mlAnalyticsList") or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            interoperability = entry.get("mlModelInterInfo") or {}
            entry_tais: tuple[dict[str, object], ...] = ()
            if tracking_areas:
                try:
                    entry_tais = tuple(
                        _normalize_tai(value)
                        for value in entry.get("trackingAreaList") or ()
                        if isinstance(value, dict)
                    )
                except ValueError:
                    continue
            matches_tracking_area = not tracking_areas or bool(
                {_tai_key(value) for value in entry_tais}
                & {_tai_key(value) for value in tracking_areas}
            )
            if (
                ml_event in (entry.get("mlAnalyticsIds") or [])
                and entry.get("flCapabilityType") in accepted_capabilities
                and isinstance(interoperability, dict)
                and model_interoperability in (interoperability.get("vendorList") or [])
                and matches_tracking_area
            ):
                return True
    return False


def hierarchy_discovery_requirements(
    value: NwdafMLModelTrainSubsc,
) -> HierarchyDiscoveryRequirements:
    if not value.ml_event_subscriptions:
        raise ValueError("hierarchy discovery requires an ML event subscription")
    event = value.ml_event_subscriptions[0]
    if not event.ml_event.strip() or event.ml_event.strip() != event.ml_event:
        raise ValueError("hierarchy discovery requires a canonical ML event")
    if not event.model_interoperability.strip():
        raise ValueError("hierarchy discovery requires model interoperability")
    area = event.ml_event_filter.get("networkArea")
    if not isinstance(area, Mapping):
        raise ValueError("bounded hierarchy discovery requires networkArea")
    raw_tais = area.get("tais")
    if not isinstance(raw_tais, list) or not raw_tais:
        raise ValueError("bounded hierarchy discovery requires tracking areas")
    tais = tuple(
        _normalize_tai(item)
        for item in raw_tais
        if isinstance(item, Mapping)
    )
    if len(tais) != len(raw_tais):
        raise ValueError("hierarchy discovery tracking areas are invalid")
    return HierarchyDiscoveryRequirements(
        ml_event=event.ml_event,
        model_interoperability=event.model_interoperability,
        tracking_areas=tais,
    )


def _nwdaf_infos(profile: dict) -> tuple[dict, ...]:
    values: list[dict] = []
    if isinstance(profile.get("nwdafInfo"), dict):
        values.append(profile["nwdafInfo"])
    info_list = profile.get("nwdafInfoList") or {}
    if isinstance(info_list, dict):
        values.extend(value for value in info_list.values() if isinstance(value, dict))
    elif isinstance(info_list, list):
        values.extend(value for value in info_list if isinstance(value, dict))
    return tuple(values)


def _services(profile: dict) -> tuple[tuple[str, dict], ...]:
    values = [
        (str(item.get("serviceInstanceId", "")), item)
        for item in profile.get("nfServices") or []
        if isinstance(item, dict)
    ]
    service_list = profile.get("nfServiceList") or {}
    if isinstance(service_list, dict):
        values.extend(
            (str(item.get("serviceInstanceId") or key), item)
            for key, item in service_list.items()
            if isinstance(item, dict)
        )
    return tuple(values)


def _derive_root(profile: dict, service: dict) -> str:
    scheme = service.get("scheme")
    endpoints = service.get("ipEndPoints") or []
    endpoint = endpoints[0] if endpoints and isinstance(endpoints[0], dict) else {}
    host = (
        service.get("fqdn")
        or profile.get("fqdn")
        or endpoint.get("ipv4Address")
        or endpoint.get("ipv6Address")
    )
    port = endpoint.get("port")
    if scheme not in {"http", "https"} or not host:
        return ""
    host = str(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}{f':{port}' if port else ''}"


def _normalize_api_root(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise HierarchyDiscoveryError("Training service API root must be an HTTP(S) URI")
    return normalized


def _parse_search_result(result: object) -> _ParsedSearchResult:
    if not isinstance(result, dict):
        raise HierarchyDiscoveryError("NRF discovery returned a malformed SearchResult")
    validity = result.get("validityPeriod")
    profiles = result.get("nfInstances")
    complete_count = result.get("numNfInstComplete")
    if (
        not isinstance(validity, int)
        or isinstance(validity, bool)
        or validity < 0
        or not isinstance(profiles, list)
        or (
            complete_count is not None
            and (
                not isinstance(complete_count, int)
                or isinstance(complete_count, bool)
                or complete_count < len(profiles)
            )
        )
    ):
        raise HierarchyDiscoveryError("NRF discovery returned a malformed SearchResult")
    return _ParsedSearchResult(
        profiles=tuple(profiles),
        validity_period=validity,
        returned_count=len(profiles),
        complete_nf_instance_count=complete_count,
    )


def _observed_at(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("discovery clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _scope(
    *,
    containing_nf_instance_id: str,
    role: HierarchyNodeRole,
    ml_event: str,
    model_interoperability: str,
    tracking_areas: tuple[dict[str, object], ...],
    target_nf_instance_id: str | None = None,
) -> HierarchyDiscoveryScope:
    return HierarchyDiscoveryScope(
        containing_nf_instance_id=normalize_nf_instance_id(containing_nf_instance_id),
        role=role,
        ml_event=ml_event,
        model_interoperability=model_interoperability,
        tracking_areas=tuple(sorted({_tai_key(value) for value in tracking_areas})),
        target_nf_instance_id=(
            normalize_nf_instance_id(target_nf_instance_id)
            if target_nf_instance_id is not None
            else None
        ),
    )


def _query_entry(
    *,
    ml_event: str,
    capability: str,
    model_interoperability: str,
    tracking_areas: tuple[dict[str, object], ...],
) -> dict[str, object]:
    entry: dict[str, object] = {
        "mlAnalyticsIds": [ml_event],
        "flCapabilityType": capability,
        "mlModelInterInfo": {"vendorList": [model_interoperability]},
    }
    if tracking_areas:
        entry["trackingAreaList"] = list(tracking_areas)
    return entry


def _profile_interoperability(value: str) -> str:
    if re.fullmatch(r"[0-9]{6}", value):
        return value
    if value in IMAGE_MODEL_INTEROPERABILITY.values():
        return "001122"
    raise ValueError(
        "model interoperability has no configured NRF VendorId mapping"
    )


_MCC = re.compile(r"^[0-9]{3}$")
_MNC = re.compile(r"^[0-9]{2,3}$")
_TAC = re.compile(r"^(?:[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6})$")


def _normalize_tai(value: Mapping[str, object]) -> dict[str, object]:
    plmn = value.get("plmnId")
    tac = value.get("tac")
    if not isinstance(plmn, Mapping) or not isinstance(tac, str):
        raise ValueError("tracking area must contain plmnId and tac")
    mcc = plmn.get("mcc")
    mnc = plmn.get("mnc")
    if (
        not isinstance(mcc, str)
        or not isinstance(mnc, str)
        or not _MCC.fullmatch(mcc)
        or not _MNC.fullmatch(mnc)
        or not _TAC.fullmatch(tac)
    ):
        raise ValueError("tracking area is invalid")
    return {"plmnId": {"mcc": mcc, "mnc": mnc}, "tac": tac.upper()}


def _normalize_tais(
    values: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for value in values:
        normalized = _normalize_tai(value)
        by_key[_tai_key(normalized)] = normalized
    return tuple(by_key[key] for key in sorted(by_key))


def _tai_key(value: Mapping[str, object]) -> tuple[str, str, str]:
    plmn = value["plmnId"]
    assert isinstance(plmn, Mapping)
    return str(plmn["mcc"]), str(plmn["mnc"]), str(value["tac"]).upper()


def _service_targets(profile: dict, nf_instance_id: str) -> tuple[SelectedTarget, ...]:
    candidates: list[SelectedTarget] = []
    for service_id, service in _services(profile):
        if (
            not service_id
            or service.get("serviceName") != "nnwdaf-mlmodeltraining"
            or service.get("nfServiceStatus") != "REGISTERED"
        ):
            continue
        root = service.get("apiPrefix") or _derive_root(profile, service)
        if not root:
            continue
        try:
            api_root = _normalize_api_root(str(root))
        except HierarchyDiscoveryError:
            continue
        candidates.append(
            SelectedTarget(
                nfInstanceId=nf_instance_id,
                nfServiceInstanceId=service_id,
                serviceName="nnwdaf-mlmodeltraining",
                apiRoot=api_root,
                selectionSource="NRF",
            )
        )
    unique = {
        (item.nf_service_instance_id, item.api_root): item for item in candidates
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.nf_service_instance_id, item.api_root),
        )
    )
