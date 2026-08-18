from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

import httpx

from py_mtlf.config import FederatedLearningSettings
from py_mtlf.core.fl_hierarchy import normalize_nf_instance_id
from py_mtlf.core.nwdaf_context import NwdafContextClient
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


class HierarchyNodeResolver:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        nwdaf_context: NwdafContextClient,
        client: httpx.Client | None = None,
    ) -> None:
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None

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

        capability = (
            "FL_SERVER_AND_CLIENT"
            if role is HierarchyNodeRole.BRANCH
            else "FL_CLIENT"
        )
        query_entry = {
            "mlAnalyticsIds": [ml_event],
            "flCapabilityType": capability,
            "mlModelInterInfo": {"vendorList": [model_interoperability]},
        }
        try:
            context = self._nwdaf_context.get()
            response = self._client.get(
                context.internal_api_root + "/internal/v1/nrf/nf-instances",
                params={
                    "target-nf-type": "NWDAF",
                    "requester-nf-type": "NWDAF",
                    "target-nf-instance-id": nf_id,
                    "service-names": "nnwdaf-mlmodeltraining",
                    "ml-analytics-info-list": json.dumps(
                        [query_entry],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HierarchyDiscoveryError(
                f"NRF discovery failed for {role.value} {nf_id}"
            ) from error
        if not isinstance(result, dict):
            raise HierarchyDiscoveryError("NRF discovery returned a malformed SearchResult")
        validity = result.get("validityPeriod")
        profiles = result.get("nfInstances")
        if not isinstance(validity, int) or validity < 0 or not isinstance(profiles, list):
            raise HierarchyDiscoveryError("NRF discovery returned a malformed SearchResult")

        exact_profiles = []
        for profile in profiles:
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
            model_interoperability=model_interoperability,
        ):
            raise HierarchyDiscoveryError(
                f"{role.value} {nf_id} does not advertise the required FL capability"
            )

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
            candidates.append(
                SelectedTarget(
                    nfInstanceId=nf_id,
                    nfServiceInstanceId=service_id,
                    serviceName="nnwdaf-mlmodeltraining",
                    apiRoot=_normalize_api_root(str(root)),
                    selectionSource="NRF",
                )
            )
        unique = {
            (
                item.nf_service_instance_id,
                item.api_root,
            ): item
            for item in candidates
        }
        if len(unique) != 1:
            raise HierarchyDiscoveryError(
                f"NRF discovery did not uniquely resolve the Training service for {nf_id}"
            )
        target = next(iter(unique.values()))
        return ResolvedHierarchyNode(nf_instance_id=nf_id, role=role, target=target)


def _supports_requirement(
    profile: dict,
    *,
    role: HierarchyNodeRole,
    ml_event: str,
    model_interoperability: str,
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
            if (
                ml_event in (entry.get("mlAnalyticsIds") or [])
                and entry.get("flCapabilityType") in accepted_capabilities
                and isinstance(interoperability, dict)
                and model_interoperability in (interoperability.get("vendorList") or [])
            ):
                return True
    return False


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
