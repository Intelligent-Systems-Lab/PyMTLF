from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from py_mtlf.config import PrivateCollectionProfileSettings
from py_mtlf.core.adrf_discovery import AdrfResolver, normalize_api_root
from py_mtlf.core.nwdaf_context import NwdafContextClient


class CollectionRelayError(RuntimeError):
    def __init__(
        self,
        cause: str,
        detail: str,
        *,
        retryable: bool = True,
        provisional_subscription: AcceptedSubscription | None = None,
    ) -> None:
        super().__init__(detail)
        self.cause = cause
        self.detail = detail
        self.retryable = retryable
        self.provisional_subscription = provisional_subscription


@dataclass(frozen=True)
class ServingSmfTarget:
    supi: str
    smf_nf_instance_id: str
    api_root: str
    pdu_session_id: int
    dnn: str
    snssai: dict | None


@dataclass(frozen=True)
class AcceptedSubscription:
    subscription_id: str
    location: str
    target: ServingSmfTarget
    correlation_id: str
    representation: dict


@dataclass(frozen=True)
class StorageReceipt:
    transport: str
    adrf_instance_id: str = ""


class CollectionRelayClient:
    def __init__(
        self,
        context: NwdafContextClient,
        adrf_resolver: AdrfResolver,
        timeout_seconds: float,
        *,
        mongo_settings=None,
        client: httpx.Client | None = None,
    ) -> None:
        self._context = context
        self._adrf_resolver = adrf_resolver
        self._timeout = timeout_seconds
        self._mongo = mongo_settings
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=False)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def resolve_profile(
        self,
        profile: PrivateCollectionProfileSettings,
    ) -> tuple[ServingSmfTarget, ...]:
        targets: list[ServingSmfTarget] = []
        for group_id in profile.target_ue.int_group_ids:
            udm_id, sdm_root = self._discover_one(
                "UDM",
                "nudm-sdm",
                {"internal-group-identity": group_id},
            )
            group = self._get(
                "/internal/v1/udm-sdm/group-data/group-identifiers",
                sdm_root,
                params={"int-group-id": group_id, "ue-id-ind": "true"},
            )
            if group.get("intGroupId") not in {None, "", group_id}:
                raise CollectionRelayError(
                    "INVALID_PEER_RESPONSE",
                    "UDM returned a different Internal Group ID",
                    retryable=False,
                )
            members = group.get("ueIdList")
            if not isinstance(members, list) or not members:
                raise CollectionRelayError("TARGET_UNAVAILABLE", "UDM group has no UE members")
            _, uecm_root = self._discover_one(
                "UDM",
                "nudm-uecm",
                {"target-nf-instance-id": udm_id},
            )
            for member in members:
                supi = member.get("supi") if isinstance(member, dict) else None
                if not isinstance(supi, str) or not supi:
                    raise CollectionRelayError(
                        "INVALID_PEER_RESPONSE",
                        "UDM returned malformed group membership",
                        retryable=False,
                    )
                registrations = self._get(
                    f"/internal/v1/udm-uecm/{supi}/registrations/smf-registrations",
                    uecm_root,
                ).get("smfRegistrationList")
                if not isinstance(registrations, list):
                    raise CollectionRelayError(
                        "INVALID_PEER_RESPONSE",
                        "UDM returned malformed SMF registrations",
                        retryable=False,
                    )
                matched = [item for item in registrations if self._matches(item, profile)]
                if not matched:
                    raise CollectionRelayError(
                        "TARGET_UNAVAILABLE",
                        "no serving SMF registration matches the collection profile",
                    )
                for registration in matched:
                    smf_id = str(registration.get("smfInstanceId", ""))
                    _, smf_root = self._discover_one(
                        "SMF",
                        "nsmf-event-exposure",
                        {"target-nf-instance-id": smf_id},
                    )
                    targets.append(
                        ServingSmfTarget(
                            supi=supi,
                            smf_nf_instance_id=smf_id,
                            api_root=smf_root,
                            pdu_session_id=int(registration["pduSessionId"]),
                            dnn=str(registration.get("dnn", "")),
                            snssai=registration.get("singleNssai"),
                        )
                    )
        identities = {
            (
                item.supi,
                item.smf_nf_instance_id,
                item.api_root,
                item.pdu_session_id,
                item.dnn,
                str(item.snssai),
            ): item
            for item in targets
        }
        return tuple(identities[key] for key in sorted(identities))

    def create_subscription(
        self,
        target: ServingSmfTarget,
        correlation_id: str,
        payload: dict,
    ) -> AcceptedSubscription:
        try:
            response = self._client.post(
                self._internal_root()
                + "/internal/v1/smf-event-exposure/subscriptions",
                headers={"Target-Api-Root": target.api_root},
                json=payload,
            )
        except httpx.HTTPError as error:
            raise CollectionRelayError(
                "SMF_SUBSCRIPTION_FAILED",
                "SMF subscription create transport failed",
            ) from error
        location = response.headers.get("Location", "").strip()
        if response.status_code != 201 or not location:
            raise CollectionRelayError(
                "SMF_SUBSCRIPTION_FAILED",
                f"SMF subscription create failed with status {response.status_code}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
            )
        subscription_id = location.rstrip("/").rsplit("/", 1)[-1]
        try:
            representation = response.json()
        except ValueError as error:
            provisional = AcceptedSubscription(
                subscription_id,
                location,
                target,
                correlation_id,
                payload,
            )
            raise CollectionRelayError(
                "INVALID_PEER_RESPONSE",
                "SMF subscription create returned malformed JSON",
                retryable=False,
                provisional_subscription=provisional,
            ) from error
        if not isinstance(representation, dict):
            representation = {}
        provisional = AcceptedSubscription(
            subscription_id,
            location,
            target,
            correlation_id,
            representation or payload,
        )
        collection_fields = (
            "supi",
            "pduSeId",
            "dnn",
            "snssai",
            "nfId",
            "notifId",
            "notifUri",
            "eventSubs",
            "notifMethod",
            "repPeriod",
        )
        if (
            not subscription_id
            or representation.get("notifId") != correlation_id
            or representation.get("subId") != subscription_id
            or any(
                field in payload and representation.get(field) != payload[field]
                for field in collection_fields
            )
        ):
            raise CollectionRelayError(
                "INVALID_PEER_RESPONSE",
                "SMF subscription create returned an invalid accepted resource",
                retryable=False,
                provisional_subscription=provisional,
            )
        if representation.get("expiry") is not None:
            raise CollectionRelayError(
                "UNSUPPORTED_FINITE_LEASE",
                "finite SMF subscription expiry is not supported",
                retryable=False,
                provisional_subscription=provisional,
            )
        return AcceptedSubscription(
            subscription_id,
            location,
            target,
            correlation_id,
            representation,
        )

    def delete_subscription(self, subscription: AcceptedSubscription) -> bool:
        try:
            response = self._client.delete(
                self._internal_root()
                + "/internal/v1/smf-event-exposure/subscriptions/"
                + subscription.subscription_id,
                headers={"Target-Api-Root": subscription.target.api_root},
            )
        except httpx.HTTPError:
            return False
        if response.status_code in {204, 404}:
            return True
        if response.status_code in {429, 500, 502, 503, 504}:
            return False
        raise CollectionRelayError(
            "SMF_CLEANUP_FAILED",
            f"SMF subscription delete failed with status {response.status_code}",
            retryable=False,
        )

    def store_record(self, record: dict, *, supi: str, measurement_time) -> StorageReceipt:
        try:
            adrf = self._adrf_resolver.resolve_data()
        except Exception as error:
            raise CollectionRelayError(
                "STORAGE_UNAVAILABLE",
                "ADRF target resolution failed",
            ) from error
        if adrf is not None:
            try:
                response = self._client.post(
                    self._internal_root()
                    + "/internal/v1/adrf-data-management/data-store-records",
                    headers={"Target-Api-Root": adrf.api_root},
                    json=record,
                )
            except httpx.HTTPError as error:
                if self._mongo is None:
                    raise CollectionRelayError(
                        "STORAGE_UNAVAILABLE",
                        "ADRF data-store transport failed",
                    ) from error
                response = None
            location = response.headers.get("Location", "") if response is not None else ""
            if (
                response is not None
                and response.status_code == 201
                and self._valid_resource_location(adrf.api_root, location)
            ):
                return StorageReceipt("adrf", adrf.nf_instance_id)
            if (
                response is not None
                and 400 <= response.status_code < 500
                and response.status_code != 429
            ):
                raise CollectionRelayError(
                    "ADRF_RECORD_REJECTED",
                    f"ADRF rejected data-store record with status {response.status_code}",
                    retryable=False,
                )
        if self._mongo is None:
            raise CollectionRelayError("STORAGE_UNAVAILABLE", "durable storage is unavailable")
        from pymongo import ASCENDING, MongoClient
        from pymongo.errors import PyMongoError

        try:
            client = MongoClient(
                self._mongo.url,
                serverSelectionTimeoutMS=self._mongo.connect_timeout_ms,
                connectTimeoutMS=self._mongo.connect_timeout_ms,
                socketTimeoutMS=self._mongo.read_timeout_ms,
                tz_aware=True,
            )
            try:
                collection = client[self._mongo.database][self._mongo.collection]
                collection.create_index(
                    [("supi", ASCENDING), ("measurementTime", ASCENDING)]
                )
                document = {
                    "supi": supi,
                    "measurementTime": measurement_time,
                    "dataSub": record["dataSub"],
                    "dataNotif": record["dataNotif"],
                }
                collection.insert_one(document)
            finally:
                client.close()
        except PyMongoError as error:
            raise CollectionRelayError(
                "STORAGE_UNAVAILABLE",
                "MongoDB durable storage failed",
            ) from error
        return StorageReceipt("mongodb")

    @staticmethod
    def _valid_resource_location(api_root: str, location: str) -> bool:
        if not location.strip():
            return False
        parsed = urlsplit(urljoin(api_root.rstrip("/") + "/", location))
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )

    def _discover_one(
        self,
        nf_type: str,
        service_name: str,
        extra: dict[str, str],
    ) -> tuple[str, str]:
        params = {
            "target-nf-type": nf_type,
            "requester-nf-type": "NWDAF",
            "service-names": service_name,
            **extra,
        }
        result = self._get("/internal/v1/nrf/nf-instances", "", params=params)
        candidates: list[tuple[str, str]] = []
        required_nf_instance_id = extra.get("target-nf-instance-id", "")
        for profile in result.get("nfInstances") or []:
            if profile.get("nfStatus") not in {None, "", "REGISTERED"}:
                continue
            nf_instance_id = str(profile.get("nfInstanceId", ""))
            if not nf_instance_id or (
                required_nf_instance_id
                and nf_instance_id != required_nf_instance_id
            ):
                continue
            for service in [
                *(profile.get("nfServices") or []),
                *(profile.get("nfServiceList") or {}).values(),
            ]:
                if (
                    service.get("serviceName") != service_name
                    or service.get("nfServiceStatus") != "REGISTERED"
                ):
                    continue
                root = service.get("apiPrefix") or self._derive_service_root(
                    profile,
                    service,
                )
                if root:
                    candidates.append(
                        (nf_instance_id, normalize_api_root(root))
                    )
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise CollectionRelayError(
                "TARGET_AMBIGUOUS" if candidates else "TARGET_UNAVAILABLE",
                f"expected exactly one registered {service_name} endpoint",
            )
        return candidates[0]

    @staticmethod
    def _derive_service_root(profile: dict, service: dict) -> str:
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
        return normalize_api_root(f"{scheme}://{host}{f':{port}' if port else ''}")

    def _get(
        self,
        path: str,
        target_api_root: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict:
        headers = {"Target-Api-Root": target_api_root} if target_api_root else None
        try:
            response = self._client.get(
                self._internal_root() + path,
                params=params,
                headers=headers,
            )
        except httpx.HTTPError as error:
            raise CollectionRelayError(
                "PEER_REQUEST_FAILED",
                "collection relay transport failed",
            ) from error
        if response.status_code != 200:
            raise CollectionRelayError(
                "PEER_REQUEST_FAILED",
                f"collection relay failed with status {response.status_code}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
            )
        try:
            value = response.json()
        except ValueError as error:
            raise CollectionRelayError(
                "INVALID_PEER_RESPONSE",
                "collection relay returned malformed JSON",
                retryable=False,
            ) from error
        if not isinstance(value, dict):
            raise CollectionRelayError(
                "INVALID_PEER_RESPONSE",
                "collection relay returned an invalid representation",
                retryable=False,
            )
        return value

    def _internal_root(self) -> str:
        try:
            value = self._context.get().internal_api_root
        except RuntimeError as error:
            raise CollectionRelayError(
                "CONTAINING_NWDAF_UNAVAILABLE",
                "containing NWDAF context is unavailable",
            ) from error
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CollectionRelayError(
                "CONTAINING_NWDAF_UNAVAILABLE",
                "containing NWDAF internal API root is invalid",
            )
        return value.rstrip("/")

    @staticmethod
    def _matches(registration: dict, profile: PrivateCollectionProfileSettings) -> bool:
        if profile.dnns and str(registration.get("dnn", "")).lower() not in profile.dnns:
            return False
        if not profile.snssais:
            return True
        value = registration.get("singleNssai") or {}
        return any(
            value.get("sst") == expected.sst and str(value.get("sd", "")).upper() == expected.sd
            for expected in profile.snssais
        )
