from datetime import UTC, datetime
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pymongo.errors import AutoReconnect

from py_mtlf.app import create_app
from py_mtlf.config import AccuracyPolicySettings, DatasetSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, ScopeReference
from py_mtlf.core.dataset import (
    AdrfRoute,
    DatasetCoordinator,
    DatasetJob,
    DatasetJobState,
)
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.models import BackendSyncRequest
from py_mtlf.wire.adrf import (
    DataNotification,
    DataSubscription,
    NadrfDataRetrievalNotification,
    NadrfDataStoreRecord,
    TimeWindow,
)
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorNotification,
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)


class CatalogStub:
    provider_namespace = "local-mtlf"
    family_key = ("local-mtlf", "ue-communication-default")
    version_key = ("local-mtlf", 1)

    def version_key_for_id(self, model_id):
        return self.provider_namespace, model_id

    def family_for_version(self, version_key):
        return self.family_key if version_key == self.version_key else None

    def current(self, family_key):
        if family_key != self.family_key:
            return None
        return type("Current", (), {"version_key": self.version_key})()


def monitor(group: str) -> MLModelMonitorSubscription:
    return MLModelMonitorSubscription(
        modelIds=[1],
        notificationUri="http://go.example/notify",
        notifCorrId=f"monitor-{group}",
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def registration(group: str) -> MLModelMonitorRegistration:
    return MLModelMonitorRegistration(
        consumerId="11111111-1111-4111-8111-111111111111",
        modelId=1,
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def report(group: str, deviation: float) -> MLModelMonitorNotification:
    return MLModelMonitorNotification(
        notifCorrId=f"monitor-{group}",
        modelAccuInfos=[{"modelId": 1, "deviation": deviation}],
    )


def retrain_intent():
    policy = AccuracyPolicy(
        AccuracyPolicySettings(
            reference_buffer_size=2,
            min_reference_samples=1,
            min_std=0.1,
            fixed_floor=0.05,
            z_score_threshold=1,
            decision_window_size=1,
            required_hits=1,
        ),
        CatalogStub(),
    )
    for group in ("group-a", "group-b"):
        policy.observe(monitor(group), report(group, 0.1), registration(group))
    policy.observe(monitor("group-a"), report("group-a", 0.5), registration("group-a"))
    return policy, policy.take_intents()[0]


def sync_snapshot() -> BackendSyncRequest:
    return BackendSyncRequest.model_validate(
        {
            "containingNwdaf": {
                "nfInstanceId": "nwdaf-1",
                "apiBaseUri": "http://go.example",
                "internalCallbackBaseUri": "http://go-internal.example",
            },
            "eventsSubscriptions": [
                {
                    "subscriptionId": f"events-{group}",
                    "subscription": {
                        "eventSubscriptions": [
                            {
                                "event": "UE_COMMUNICATION",
                                "tgtUe": {"intGroupIds": [group]},
                            }
                        ]
                    },
                    "externalNotificationUri": "http://consumer.example",
                }
                for group in ("group-a", "group-b")
            ],
            "smfResources": [
                {
                    "correlationId": f"smf-{group}",
                    "resourceLocation": (
                        f"http://smf.example/nsmf-event-exposure/v1/subscriptions/{group}"
                    ),
                    "targetApiRoot": "http://smf.example",
                    "nwdafSubscriptionIds": [f"events-{group}"],
                    "pendingCleanup": False,
                    "subscription": {
                        "supi": f"imsi-{index}",
                        "notifId": f"smf-{group}",
                        "notifUri": "http://anlf.example/callback",
                        "eventSubs": [{"event": "UPF_EVENT"}],
                    },
                }
                for index, group in enumerate(("group-a", "group-b"), start=1)
            ],
            "trainingDataSource": "mongodb",
        }
    )


def test_scope_resolution_keeps_all_active_groups_and_peer_identity():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )

    resources = coordinator._resolve(intent, projection.snapshot())

    assert {resource.supi for resource in resources} == {"imsi-1", "imsi-2"}
    assert {scope for resource in resources for scope in resource.scope_keys} == set(
        intent.active_scope_keys
    )
    assert all(resource.identity.startswith("http://smf.example|") for resource in resources)
    coordinator._client.close()


def test_training_scope_matches_standard_event_subscription_network_area():
    tai = {
        "plmnId": {"mcc": "466", "mnc": "92"},
        "tac": "000001",
    }
    scope = ScopeReference(
        scope_key="scope-a",
        consumer_id="",
        model_ids=(),
        ml_event="UE_COMMUNICATION",
        ml_event_filter={"networkArea": {"tais": [tai]}},
        target_ue={"intGroupIds": ["group-a"]},
    )
    subscription = {
        "eventSubscriptions": [
            {
                "event": "UE_COMMUNICATION",
                "tgtUe": {"intGroupIds": ["group-a"]},
                "networkArea": {"tais": [tai]},
            }
        ]
    }

    assert DatasetCoordinator._matches_scope(scope, subscription)

    subscription["eventSubscriptions"][0]["networkArea"] = {
        "tais": [
            {
                "plmnId": {"mcc": "466", "mnc": "92"},
                "tac": "000002",
            }
        ]
    }
    assert not DatasetCoordinator._matches_scope(scope, subscription)


def test_dataset_ready_requires_each_scope_and_deduplicates_native_identity():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )
    window = TimeWindow(
        startTime=datetime(2026, 7, 24, tzinfo=UTC),
        stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
    )
    job = DatasetJob("job-1", intent, window, "mongodb")
    coordinator._jobs[job.job_id] = job
    job.resources = coordinator._resolve(intent, projection.snapshot())
    for resource in job.resources:
        record = NadrfDataStoreRecord(
            dataSub=[DataSubscription(smfDataSub=resource.smf_data_sub)],
            dataNotif=DataNotification(upfEventNotifs=[{"sample": resource.supi}]),
        )
        coordinator._append_record(
            job,
            resource,
            record,
            "mongodb",
            window.start_time,
            resource.supi,
        )
        coordinator._append_record(
            job,
            resource,
            record,
            "mongodb",
            window.start_time,
            resource.supi,
        )

    coordinator._complete(job)

    assert job.state == DatasetJobState.READY
    assert job.snapshot is not None
    assert len(job.snapshot.records) == 2
    assert set(job.snapshot.scope_record_counts.values()) == {1}
    coordinator._client.close()


def test_ready_snapshot_can_only_be_claimed_once_and_records_terminal_outcome():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )
    window = TimeWindow(
        startTime=datetime(2026, 7, 24, tzinfo=UTC),
        stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
    )
    job = DatasetJob("job-1", intent, window, "mongodb")
    coordinator._jobs[job.job_id] = job
    job.resources = coordinator._resolve(intent, projection.snapshot())
    for resource in job.resources:
        coordinator._append_record(
            job,
            resource,
            NadrfDataStoreRecord(
                dataSub=[DataSubscription(smfDataSub=resource.smf_data_sub)],
                dataNotif=DataNotification(upfEventNotifs=[{"sample": resource.supi}]),
            ),
            "mongodb",
            window.start_time,
            resource.supi,
        )
    coordinator._complete(job)

    claimed = coordinator.claim_ready(job.job_id)

    assert claimed is not None
    assert coordinator.claim_ready(job.job_id) is None
    coordinator.finish_claim(job.job_id, success=True)
    assert job.state == DatasetJobState.COMPLETED
    coordinator.shutdown()


def test_adrf_fetch_uses_instruction_uri_and_bounded_same_origin_redirect():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    requests: list[str] = []
    resource = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )._resolve(intent, projection.snapshot())[0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/data-store-records"):
            return httpx.Response(
                307,
                headers={"Location": "/redirected-records"},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "dataSub": [{"smfDataSub": resource.smf_data_sub}],
                "dataNotif": {"upfEventNotifs": [{"sample": resource.supi}]},
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
        client=client,
    )
    route = AdrfRoute(
        "corr-1",
        resource,
        fetch_uri="http://adrf.example/nadrf-datamanagement/v1/data-store-records",
    )
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "adrf",
    )

    coordinator._fetch_one(job, route, "http://adrf.example", "fetch-1")

    assert requests == [
        "http://adrf.example/nadrf-datamanagement/v1/data-store-records?fetch-correlation-ids=fetch-1",
        "http://adrf.example/redirected-records",
    ]
    assert len(job.records) == 1
    client.close()


def test_adrf_fetch_rejects_cross_origin_instruction():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )
    resource = coordinator._resolve(intent, projection.snapshot())[0]
    route = AdrfRoute(
        "corr-1",
        resource,
        fetch_uri="http://other.example/nadrf-datamanagement/v1/data-store-records",
    )
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "adrf",
    )

    try:
        coordinator._fetch_one(job, route, "http://adrf.example", "fetch-1")
    except RuntimeError as error:
        assert "selected ADRF origin" in str(error)
    else:
        raise AssertionError("cross-origin fetchUri was accepted")
    coordinator._client.close()


def test_adrf_fetch_retries_transport_failure():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    resource = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )._resolve(intent, projection.snapshot())[0]
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary failure", request=request)
        return httpx.Response(
            200,
            json={
                "dataSub": [{"smfDataSub": resource.smf_data_sub}],
                "dataNotif": {"upfEventNotifs": [{"sample": resource.supi}]},
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    coordinator = DatasetCoordinator(
        DatasetSettings(max_retry_attempts=2, retry_initial_backoff_seconds=0),
        projection,
        policy,
        Mock(close=Mock()),
        client=client,
    )
    route = AdrfRoute(
        "corr-1",
        resource,
        fetch_uri="http://adrf.example/nadrf-datamanagement/v1/data-store-records",
    )
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "adrf",
    )

    coordinator._fetch_one(job, route, "http://adrf.example", "fetch-1")

    assert attempts == 2
    assert len(job.records) == 1
    coordinator.shutdown()
    client.close()


def test_adrf_callback_rejects_fetch_ids_above_job_limit():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(max_records_per_job=1),
        projection,
        policy,
        Mock(close=Mock()),
    )
    resource = coordinator._resolve(intent, projection.snapshot())[0]
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "adrf",
    )
    job.routes["corr-1"] = AdrfRoute("corr-1", resource)
    coordinator._jobs[job.job_id] = job
    coordinator._routes["corr-1"] = job.job_id

    with pytest.raises(RuntimeError, match="exceed"):
        coordinator.receive_notification(
            NadrfDataRetrievalNotification.model_validate(
                {
                    "notifCorrId": "corr-1",
                    "timeStamp": "2026-07-24T00:00:00Z",
                    "fetchInstruct": {
                        "fetchUri": (
                            "http://adrf.example/nadrf-datamanagement/v1/data-store-records"
                        ),
                        "fetchCorrIds": ["fetch-1", "fetch-2"],
                    },
                }
            )
        )

    assert job.failure == "ADRF fetch IDs exceed max_records_per_job"
    assert job.routes["corr-1"].pending_fetch_ids == ["fetch-1"]
    coordinator.shutdown()


@patch("pymongo.MongoClient")
def test_mongo_skips_malformed_document_and_keeps_valid_records(mongo_client):
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(),
        projection,
        policy,
        Mock(close=Mock()),
    )
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "mongodb",
    )
    job.resources = coordinator._resolve(intent, projection.snapshot())
    resource = job.resources[0]
    collection = mongo_client.return_value.__getitem__.return_value.__getitem__.return_value
    collection.find.return_value.sort.return_value = [
        {
            "_id": "bad",
            "supi": resource.supi,
            "measurementTime": job.time_window.start_time,
            "dataSub": None,
            "dataNotif": None,
        },
        {
            "_id": "good",
            "supi": resource.supi,
            "measurementTime": job.time_window.start_time,
            "dataSub": [{"smfDataSub": resource.smf_data_sub}],
            "dataNotif": {"upfEventNotifs": [{"sample": resource.supi}]},
        },
    ]

    records, malformed = coordinator._read_mongo_once(job)

    assert mongo_client.call_args.kwargs["tz_aware"] is True
    assert malformed == 1
    assert len(records) == 1
    assert records[0][3] == "good"
    coordinator.shutdown()


def test_mongo_query_retries_without_committing_partial_attempt():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    coordinator = DatasetCoordinator(
        DatasetSettings(max_retry_attempts=2, retry_initial_backoff_seconds=0),
        projection,
        policy,
        Mock(close=Mock()),
    )
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "mongodb",
    )
    job.resources = coordinator._resolve(intent, projection.snapshot())
    resource = job.resources[0]
    record = NadrfDataStoreRecord(
        dataSub=[DataSubscription(smfDataSub=resource.smf_data_sub)],
        dataNotif=DataNotification(upfEventNotifs=[{"sample": resource.supi}]),
    )
    coordinator._read_mongo_once = Mock(
        side_effect=[
            AutoReconnect("temporary"),
            (
                [
                    (
                        resource,
                        record,
                        job.time_window.start_time,
                        "record-1",
                    )
                ],
                0,
            ),
        ]
    )

    coordinator._retrieve_mongo(job)

    assert coordinator._read_mongo_once.call_count == 2
    assert [item.identity for item in job.records] == ["record-1"]
    coordinator.shutdown()


def test_adrf_cleanup_retries_and_is_idempotent():
    policy, intent = retrain_intent()
    projection = SyncProjection()
    projection.replace(sync_snapshot())
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 204, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    coordinator = DatasetCoordinator(
        DatasetSettings(max_retry_attempts=2, retry_initial_backoff_seconds=0),
        projection,
        policy,
        Mock(close=Mock()),
        client=client,
    )
    resource = coordinator._resolve(intent, projection.snapshot())[0]
    job = DatasetJob(
        "job-1",
        intent,
        TimeWindow(
            startTime=datetime(2026, 7, 24, tzinfo=UTC),
            stopTime=datetime(2026, 7, 24, 1, tzinfo=UTC),
        ),
        "adrf",
    )
    job.routes["corr-1"] = AdrfRoute(
        "corr-1",
        resource,
        peer_subscription_id="peer-sub-1",
    )

    coordinator._cleanup(job)
    coordinator._cleanup(job)

    assert attempts == 2
    assert job.routes["corr-1"].cleanup_complete is True
    assert job.routes["corr-1"].peer_subscription_id == ""
    assert job.cleanup_failure == ""
    coordinator.shutdown()
    client.close()


def test_adrf_callback_validation_uses_standard_problem_details(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/adrf-data-management/retrieval-notifications",
            json={"notifCorrId": ""},
        )

    assert response.status_code == 400
    assert response.json()["cause"] == "INVALID_MSG_FORMAT"
