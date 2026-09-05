from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.config import (
    AdrfSettings,
    ArtifactSettings,
    FederatedLearningSettings,
    FederatedStrategySettings,
    FittingSettings,
    FLClientSettings,
    FLLifecycleSettings,
    FLServerSettings,
    ModelProvisionSettings,
    RuntimeSettings,
    SeedModelSettings,
    Settings,
    StorageSettings,
    load_settings,
)
from py_mtlf.core.fl_topology import StaticTopologyPlanner


def private_network_area(tac: str = "001101") -> dict:
    return {
        "tais": [
            {
                "plmn_id": {"mcc": "466", "mnc": "92"},
                "tac": tac,
            }
        ]
    }


def test_defaults_use_confirmed_phase_one_values():
    settings = Settings()

    assert settings.server.binding_host == "127.0.0.1"
    assert settings.server.port == 9092
    assert settings.storage.artifact_root == Path("data/artifacts")
    assert settings.artifact.max_compressed_bytes == 256 * 1024 * 1024
    assert settings.artifact.max_extracted_bytes == 1024 * 1024 * 1024
    assert settings.artifact.max_single_file_bytes == 512 * 1024 * 1024
    assert settings.artifact.max_entries == 32
    assert FittingSettings().validation_ratio == 0.10
    assert FLServerSettings().preparation_data_window_seconds == 3600
    assert FLServerSettings().client_training.epochs == 18
    assert FLLifecycleSettings().terminal_status_ttl_seconds == 3600
    assert FLLifecycleSettings().tombstone_ttl_seconds == 3600


def test_workspace_must_not_overlap_durable_roots(tmp_path):
    workspace_root = tmp_path / "fl-workspaces"

    with pytest.raises(ValidationError, match="must not overlap durable storage"):
        Settings(
            storage=StorageSettings(artifact_root=workspace_root / "artifacts"),
            federated_learning=FederatedLearningSettings(
                workspace_root=workspace_root,
            ),
        )


@pytest.mark.parametrize(
    "workspace_root",
    [Path("/"), Path.cwd(), Path(__file__).resolve().parents[2]],
)
def test_workspace_rejects_broad_cleanup_roots(workspace_root):
    with pytest.raises(ValidationError, match="workspace_root is unsafe"):
        Settings(
            federated_learning=FederatedLearningSettings(
                workspace_root=workspace_root,
            )
        )


def test_federated_epochs_are_server_owned() -> None:
    server = FLServerSettings(client_training={"epochs": 4})
    assert server.client_training.epochs == 4

    with pytest.raises(ValidationError):
        FLClientSettings.model_validate({"training": {"epochs": 4}})

    for epochs in (0, -1, 1.5, "4"):
        with pytest.raises(ValidationError):
            FLServerSettings.model_validate({"client_training": {"epochs": epochs}})


def test_load_settings_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  port: 9092\n  unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(path)


@pytest.mark.parametrize(
    "value",
    [
        "file:///tmp/artifacts",
        "http://user@example.com",
        "http://example.com/path",
        "http://example.com?query=yes",
        "http://example.com:invalid",
        "http://example.com:0",
        "http://example.com:65536",
    ],
)
def test_artifact_public_base_url_requires_origin(value):
    with pytest.raises(ValidationError):
        ArtifactSettings(public_base_url=value)


def test_single_file_limit_must_not_exceed_total_limit():
    with pytest.raises(ValidationError):
        ArtifactSettings(max_single_file_bytes=11, max_extracted_bytes=10)


def test_storage_path_must_not_be_blank():
    with pytest.raises(ValidationError):
        StorageSettings(artifact_root=" ")


def test_seed_model_ids_and_artifact_keys_are_unique():
    seed = SeedModelSettings(
        family_id="ue-communication-default",
        model_id=1,
        artifact_key="a" * 64,
        event="UE_COMMUNICATION",
    )
    with pytest.raises(ValidationError, match="IDs must be unique"):
        ModelProvisionSettings(seed_models=(seed, seed))


@pytest.mark.parametrize(
    "payload",
    [
        {"storage": {"database_path": "data/old.sqlite3"}},
        {"reconciliation": {"retry_interval_seconds": 1}},
    ],
)
def test_removed_state_machine_config_is_rejected(payload):
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "other"},
        {"mode": "configured", "configured_endpoint": ""},
        {"mode": "configured", "configured_endpoint": "http://example.com/path"},
        {"mode": "configured", "configured_endpoint": "http://example.com:invalid"},
        {"retry_initial_backoff_seconds": 3, "retry_max_backoff_seconds": 2},
    ],
)
def test_adrf_settings_reject_invalid_mode_endpoint_and_backoff(payload):
    with pytest.raises(ValidationError):
        AdrfSettings.model_validate(payload)


def test_adrf_configured_endpoint_is_normalized():
    settings = AdrfSettings(
        mode="configured",
        configured_endpoint="http://adrf.example:9888/",
        configured_nf_instance_id="00000000-0000-4000-8000-000000000010",
    )

    assert settings.configured_endpoint == "http://adrf.example:9888"


@pytest.mark.parametrize("mode", ["local", "federated"])
def test_runtime_accepts_supported_modes(mode):
    assert RuntimeSettings(mode=mode.upper()).mode == mode


@pytest.mark.parametrize(
    "mode",
    ["coordinator", "fl_server", "fl_client", "root", "branch", "leaf"],
)
def test_runtime_rejects_unknown_and_role_modes(mode):
    with pytest.raises(ValidationError, match="runtime.mode"):
        RuntimeSettings(mode=mode)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("cpu", "cpu"), ("CUDA", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:12", "cuda:12")],
)
def test_training_device_is_validated_and_canonicalized(value, expected):
    assert FittingSettings(device=value).device == expected


@pytest.mark.parametrize("value", ["", "auto", "gpu", "cuda:-1", "cuda:01", "mps"])
def test_invalid_training_device_is_rejected(value):
    with pytest.raises(ValidationError, match="training.device"):
        FittingSettings(device=value)


@pytest.mark.parametrize(
    "payload",
    [
        {"workspace_root": ""},
        {"workspace_ttl_seconds": 0},
        {"public_base_url": "relative"},
        {"artifact_download": {"timeout_seconds": 0}},
        {"artifact_download": {"allowed_origins": ["http://peer.example", "http://peer.example/"]}},
    ],
)
def test_federated_learning_settings_fail_fast(payload):
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "server", "client", "local_training", "valid"),
    [
        ("local", None, None, {}, True),
        ("local", FLServerSettings(), None, None, False),
        (
            "local",
            None,
            FLClientSettings(
                training_data={"collection_trigger": "consumer_subscription"}
            ),
            None,
            False,
        ),
        ("federated", None, None, None, False),
        ("federated", FLServerSettings(), None, None, False),
        (
            "federated",
            None,
            FLClientSettings(
                training_data={"collection_trigger": "consumer_subscription"}
            ),
            None,
            True,
        ),
        (
            "federated",
            FLServerSettings(),
            FLClientSettings(
                training_data={"collection_trigger": "consumer_subscription"}
            ),
            None,
            True,
        ),
        ("federated", FLServerSettings(), None, {}, False),
    ],
)
def test_runtime_engine_configuration_matrix(
    mode,
    server,
    client,
    local_training,
    valid,
):
    payload = {
        "runtime": {"mode": mode},
        "federated_learning": {"server": server, "client": client},
        "local_training": local_training,
    }
    if valid:
        settings = Settings.model_validate(payload)
        assert (settings.federated_learning.server is not None) is (server is not None)
        assert (settings.federated_learning.client is not None) is (client is not None)
    else:
        with pytest.raises(ValidationError):
            Settings.model_validate(payload)


def test_fl_client_requires_explicit_training_data_collection_trigger(tmp_path):
    with pytest.raises(ValidationError, match="training_data"):
        FLClientSettings()

    client = FLClientSettings.model_validate(
        {
            "training_data": {"collection_trigger": "consumer_subscription"},
        }
    )

    assert client.training_data.collection_trigger == "consumer_subscription"

    private = FLClientSettings.model_validate(
        {
            "training_data": {
                "collection_trigger": "private_api",
                "callback_base_uri": "http://127.0.0.1:9092",
                "state_directory": str(tmp_path / "collections"),
                "consent": {
                    "purpose": "model_training",
                    "policy": "not_required_by_local_policy",
                },
                "collection_profiles": [
                    {
                        "profile_id": "ue-communication-default",
                        "ml_event": "UE_COMMUNICATION",
                        "ml_event_filter": {},
                        "target_ue": {"intGroupIds": ["group-a.example"]},
                        "network_area": private_network_area(),
                        "dnns": ["internet"],
                        "snssais": [{"sst": 1, "sd": "010203"}],
                        "sampling_interval_seconds": 2,
                    }
                ],
            }
        }
    )

    assert private.training_data.collection_trigger == "private_api"
    assert private.training_data.collection_profiles[0].target_ue.int_group_ids == (
        "group-a.example",
    )

    with pytest.raises(ValidationError):
        FLClientSettings.model_validate(
            {"training_data": {"collection_trigger": "local_file"}}
        )


def test_fl_client_accepts_only_matching_local_image_workload(tmp_path):
    client = FLClientSettings.model_validate(
        {
            "workload": {"profile": "image_classification"},
            "training_data": {
                "collection_trigger": "local",
                "dataset": "mnist",
                "shard_path": str((tmp_path / "train.npz").resolve()),
            },
        }
    )

    assert client.workload.profile == "image_classification"
    assert client.training_data.dataset == "mnist"

    aggregation_only = FLClientSettings.model_validate(
        {"workload": {"profile": "image_classification"}}
    )
    assert aggregation_only.training_data is None

    with pytest.raises(ValidationError, match="must use the local source"):
        FLClientSettings.model_validate(
            {
                "workload": {"profile": "image_classification"},
                "training_data": {"collection_trigger": "consumer_subscription"},
            }
        )
    with pytest.raises(ValidationError, match="requires consumer_subscription"):
        FLClientSettings.model_validate(
            {
                "training_data": {
                    "collection_trigger": "local",
                    "dataset": "cifar10",
                    "shard_path": str((tmp_path / "train.npz").resolve()),
                },
            }
        )


def test_load_settings_resolves_local_image_shard_path(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
runtime:
  mode: federated
federated_learning:
  client:
    workload:
      profile: image_classification
    training_data:
      collection_trigger: local
      dataset: cifar10
      shard_path: datasets/client-01.npz
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.federated_learning.client is not None
    assert settings.federated_learning.client.training_data.shard_path == (
        tmp_path / "datasets" / "client-01.npz"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"consent": {"purpose": "model_training", "policy": "required"}},
        {"collection_profiles": []},
        {"callback_base_uri": "http://127.0.0.1:9092/callback"},
        {"state_directory": ""},
        {"retry_initial_backoff_seconds": 2, "retry_max_backoff_seconds": 1},
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "ml_event_filter": {"networkArea": {"tais": []}},
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area(),
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"supis": ["imsi-001"]},
                    "network_area": private_network_area(),
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {
                        "intGroupIds": ["group-a.example", "group-a.example"]
                    },
                    "network_area": private_network_area(),
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area(),
                    "dnns": ["invalid_dnn"],
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area(),
                    "snssais": [
                        {"sst": 1, "sd": "010203"},
                        {"sst": 1, "sd": "010203"},
                    ],
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area(),
                    "sampling_interval_seconds": 0,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": {"tais": []},
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area("not-a-tac"),
                    "sampling_interval_seconds": 2,
                }
            ]
        },
        {
            "collection_profiles": [
                {
                    "profile_id": "ue-communication-default",
                    "ml_event": "UE_COMMUNICATION",
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "network_area": private_network_area(),
                    "sampling_interval_seconds": 2,
                    "minimum_observation_count": 1,
                }
            ]
        },
    ],
)
def test_private_collection_settings_fail_closed(tmp_path, override):
    payload = {
        "collection_trigger": "private_api",
        "callback_base_uri": "http://127.0.0.1:9092",
        "state_directory": str(tmp_path / "collections"),
        "consent": {
            "purpose": "model_training",
            "policy": "not_required_by_local_policy",
        },
        "collection_profiles": [
            {
                "profile_id": "ue-communication-default",
                "ml_event": "UE_COMMUNICATION",
                "target_ue": {"intGroupIds": ["group-a.example"]},
                "network_area": private_network_area(),
                "sampling_interval_seconds": 2,
            }
        ],
    }
    payload.update(override)

    with pytest.raises(ValidationError):
        FLClientSettings.model_validate({"training_data": payload})


def test_consumer_subscription_rejects_private_collection_fields(tmp_path):
    with pytest.raises(ValidationError):
        FLClientSettings.model_validate(
            {
                "training_data": {
                    "collection_trigger": "consumer_subscription",
                    "state_directory": str(tmp_path / "collections"),
                }
            }
        )


def test_load_settings_resolves_private_collection_state_directory(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
runtime:
  mode: federated
federated_learning:
  client:
    training_data:
      collection_trigger: private_api
      callback_base_uri: http://127.0.0.1:9092
      state_directory: state/collections
      consent:
        purpose: model_training
        policy: not_required_by_local_policy
      collection_profiles:
        - profile_id: ue-communication-default
          ml_event: UE_COMMUNICATION
          target_ue:
            intGroupIds: [group-a.example]
          network_area:
            tais:
              - plmn_id:
                  mcc: "466"
                  mnc: "92"
                tac: "001101"
          sampling_interval_seconds: 2
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.federated_learning.client is not None
    assert settings.federated_learning.client.training_data.state_directory == (
        tmp_path / "state" / "collections"
    )


def test_flat_monitor_owner_requires_explicit_orchestration_and_degradation_trigger():
    federated = FederatedLearningSettings.model_validate(
        {
            "server": {},
            "orchestration": {
                "mode": "flat",
                "participant_source": "monitor_scopes",
            },
            "training_trigger": {
                "degradation": {"enabled": True},
                "private_api": {"enabled": False},
            },
        }
    )

    assert federated.orchestration.mode == "flat"
    assert federated.orchestration.participant_source == "monitor_scopes"
    assert federated.training_trigger.degradation.enabled is True


@pytest.mark.parametrize(
    "payload",
    [
        {
            "server": {},
            "orchestration": {
                "mode": "flat",
                "participant_source": "monitor_scopes",
            },
        },
        {
            "server": {},
            "training_trigger": {"degradation": {"enabled": True}},
        },
        {
            "client": {
                "training_data": {"collection_trigger": "consumer_subscription"}
            },
            "orchestration": {
                "mode": "flat",
                "participant_source": "monitor_scopes",
            },
            "training_trigger": {"degradation": {"enabled": True}},
        },
    ],
)
def test_invalid_autonomous_owner_combinations_fail_fast(payload):
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate(payload)


def test_federated_server_requires_single_active_process():
    with pytest.raises(ValidationError, match="max_active_processes"):
        Settings(
            runtime=RuntimeSettings(mode="federated"),
            federated_learning=FederatedLearningSettings(
                server=FLServerSettings(max_active_processes=2),
                orchestration={
                    "mode": "flat",
                    "participant_source": "monitor_scopes",
                },
                training_trigger={"degradation": {"enabled": True}},
            ),
        )


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        ("local.yaml", "local"),
        ("fl-server.yaml", "federated"),
        ("fl-client.yaml", "federated"),
        ("fl-server-client.yaml", "federated"),
        ("fl-client-image-classification.yaml", "federated"),
        ("fl-client-private-collection.yaml", "federated"),
        ("fl-server-client-private-collection.yaml", "federated"),
        ("fl-server-hierarchy.yaml", "federated"),
    ],
)
def test_tracked_engine_profiles_are_valid(profile, mode):
    path = Path(__file__).parents[1] / "config" / profile

    assert load_settings(path).runtime.mode == mode


def test_tracked_hierarchy_profile_references_a_valid_topology():
    path = Path(__file__).parents[1] / "config" / "fl-server-hierarchy.yaml"
    loaded = load_settings(path)

    assert loaded.federated_learning.topology is not None
    planner = StaticTopologyPlanner.load(loaded.federated_learning.topology.config_file)
    assignment = planner.build(
        root_nf_instance_id="00000000-0000-4000-8000-000000000001"
    )
    assert len(assignment.branches) == 1


def test_removed_flat_training_and_fl_role_keys_are_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate({"training": {"epochs": 1}})
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate({"round_count": 2})


def test_load_settings_resolves_topology_from_main_config_directory(tmp_path, monkeypatch):
    config_directory = tmp_path / "deployment"
    topology_directory = config_directory / "topology"
    topology_directory.mkdir(parents=True)
    topology_path = topology_directory / "hierarchy.yaml"
    topology_path.write_text(
        "version: 1\nadmission:\n  mode: complete_required\n",
        encoding="utf-8",
    )
    config_path = config_directory / "mtlf.yaml"
    config_path.write_text(
        """
runtime:
  mode: federated
federated_learning:
  server: {}
  orchestration:
    mode: hierarchical
    participant_source: static
  training_trigger:
    degradation:
      enabled: true
  strategy:
    algorithm:
      name: fedprox
      proximal_mu: 0.01
    participant_selection: all
    waiting_policy: all
    aggregation: sample_weighted
  topology:
    strategy: static
    config_file: ./topology/hierarchy.yaml
""".strip(),
        encoding="utf-8",
    )
    unrelated_directory = tmp_path / "unrelated"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    loaded = load_settings(config_path)

    assert loaded.federated_learning.topology is not None
    assert loaded.federated_learning.topology.config_file == topology_path.resolve()


def test_hierarchy_configuration_requires_server_strategy_and_topology_together(tmp_path):
    topology = {"strategy": "static", "config_file": tmp_path / "topology.yaml"}
    strategy = FederatedStrategySettings.model_validate(
        {
            "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        }
    )

    orchestration = {"mode": "hierarchical", "participant_source": "static"}
    trigger = {"degradation": {"enabled": True}}
    with pytest.raises(ValidationError, match="requires server engine"):
        FederatedLearningSettings(
            orchestration=orchestration,
            topology=topology,
            strategy=strategy,
            training_trigger=trigger,
        )
    with pytest.raises(ValidationError, match="requires strategy"):
        FederatedLearningSettings(
            server=FLServerSettings(),
            orchestration=orchestration,
            topology=topology,
            training_trigger=trigger,
        )
    with pytest.raises(ValidationError, match="requires topology"):
        FederatedLearningSettings(
            server=FLServerSettings(),
            orchestration=orchestration,
            strategy=strategy,
            training_trigger=trigger,
        )
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate(
            {
                "server": {},
                "training_trigger": {"private_api": {"enabled": True}},
            }
        )


@pytest.mark.parametrize(
    "strategy",
    [
        {
            "algorithm": {"name": "fedavg", "proximal_mu": 0.01},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        },
        {
            "algorithm": {"name": "fedprox", "proximal_mu": 0},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        },
        {
            "algorithm": {"name": "fedprox", "proximal_mu": float("inf")},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        },
        {
            "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
            "participant_selection": "fixed_count",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        },
        {
            "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
            "participant_selection": "all",
            "waiting_policy": "minimum_results",
            "aggregation": "sample_weighted",
        },
        {
            "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "uniform",
        },
    ],
)
def test_hierarchy_strategy_rejects_unsupported_values(strategy):
    with pytest.raises(ValidationError):
        FederatedStrategySettings.model_validate(strategy)
