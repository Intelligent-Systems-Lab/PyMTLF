from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from py_mtlf.config import FittingSettings
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.trainer import LoadedBundle
from py_mtlf.core.training_data import FEATURE_ORDER, ScopeTrainingData, TrainingDataset


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, value):
        return self.linear(value[:, :, -1])


def bundle(*, fill: float | None = None) -> LoadedBundle:
    model = TinyModel()
    if fill is not None:
        for parameter in model.parameters():
            parameter.data.fill_(fill)
    scaler = StandardScaler().fit(np.arange(800, dtype=float).reshape(80, 10))
    return LoadedBundle(
        manifest={
            "workload_profile": "ue_communication_forecasting",
            "model": {"input_size": 10, "output_size": 2},
            "inference": {
                "feature_order": list(FEATURE_ORDER),
                "output_fields": ["ul_vol", "dl_vol"],
                "seq_length": 4,
                "out_seq_len": 1,
            },
        },
        model=model,
        scaler=scaler,
        model_source=b"trusted",
    )


def training_dataset() -> TrainingDataset:
    observations = np.asarray(
        [[float(index + feature) for feature in range(10)] for index in range(1, 81)]
    )
    inputs = np.asarray([observations[index : index + 4] for index in range(60)])
    targets = np.asarray([observations[index + 4, [1, 2]] for index in range(60)])
    scope = ScopeTrainingData(
        scope_key="scope-a",
        observation_count=80,
        observation_timestamps=tuple(
            datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=index)
            for index in range(80)
        ),
        observations=observations,
        training_inputs=inputs,
        training_targets=targets,
        validation_inputs=inputs[:10],
        validation_targets=targets[:10],
        training_observations=observations,
        training_eligible=True,
        evaluation_eligible=True,
        exclusion_reason="",
    )
    return TrainingDataset(
        feature_order=FEATURE_ORDER,
        output_fields=("ul_vol", "dl_vol"),
        output_indices=(1, 2),
        seq_length=4,
        out_seq_len=1,
        triggering_scope_key="scope-a",
        scopes=(scope,),
    )


@pytest.mark.parametrize("proximal_mu", [0.0, 0.25])
def test_fedprox_non_negative_mu_executes_real_optimization(proximal_mu):
    base = bundle()
    before = {
        name: value.detach().clone() for name, value in base.model.named_parameters()
    }

    result = FederatedTrainer(
        FittingSettings(batch_size=8, learning_rate=0.001, random_seed=7)
    ).train(
        base,
        training_dataset(),
        epochs=1,
        proximal_mu=proximal_mu,
    )

    assert result.training_sample_count == 60
    assert np.isfinite(result.final_loss)
    assert any(
        not torch.equal(before[name], value)
        for name, value in result.model.named_parameters()
    )


def test_sample_weighted_aggregation_uses_participant_sample_counts():
    base = bundle(fill=0)
    first = bundle(fill=2)
    second = bundle(fill=8)

    result = FederatedTrainer.aggregate(base, ((first, 3), (second, 1)))

    for value in result.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 3.5))
