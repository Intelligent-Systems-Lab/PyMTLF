import math
from dataclasses import replace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from py_mtlf.config import TrainingSettings
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.trainer import LoadedBundle, LocalTrainer, wape_sums
from py_mtlf.core.training_data import FEATURE_ORDER, ScopeTrainingData, TrainingDataset


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, value):
        return self.linear(value[:, :, -1])


def training_dataset() -> TrainingDataset:
    observations = np.asarray(
        [[float(index + feature) for feature in range(10)] for index in range(1, 81)]
    )
    inputs = np.asarray([observations[index : index + 4] for index in range(60)])
    targets = np.asarray([observations[index + 4, [1, 2]] for index in range(60)])
    validation_inputs = inputs[:10]
    validation_targets = targets[:10]
    scope = ScopeTrainingData(
        scope_key="scope-a",
        scope_digest="a" * 64,
        observation_count=80,
        training_inputs=inputs,
        training_targets=targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
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


def bundle(model: TinyModel) -> LoadedBundle:
    scaler = StandardScaler().fit(np.log1p(np.arange(800).reshape(80, 10) + 1))
    return LoadedBundle(
        manifest={
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


def test_local_trainer_warm_starts_and_always_evaluates_candidate():
    torch.manual_seed(1)
    current = bundle(TinyModel())
    candidate = bundle(TinyModel())
    candidate.model.load_state_dict(current.model.state_dict())
    before = {name: value.detach().clone() for name, value in candidate.model.state_dict().items()}
    result = LocalTrainer(
        TrainingSettings(
            epochs=3,
            batch_size=8,
            enforce_performance_gate=False,
        )
    ).train(current, candidate, training_dataset())

    assert math.isfinite(result.final_loss)
    assert result.evaluation.accepted
    assert len(result.evaluation.scopes) == 1
    assert math.isfinite(result.evaluation.aggregate_candidate.value)
    assert any(
        not torch.equal(before[name], value) for name, value in result.model.state_dict().items()
    )


def test_wape_zero_denominator_matches_accuracy_policy():
    assert wape_sums(0, 0).value == 0
    assert wape_sums(2, 0).value == 1
    assert wape_sums(2, 10).value == 0.2


def test_per_scope_regression_rejects_candidate_even_when_aggregate_improves(
    monkeypatch,
):
    base = training_dataset()
    trigger = replace(
        base.scopes[0],
        validation_targets=np.full((2, 2), 10.0),
    )
    peer = replace(
        trigger,
        scope_key="scope-b",
        scope_digest="b" * 64,
    )
    dataset = replace(base, scopes=(trigger, peer))
    current = bundle(TinyModel())
    candidate = bundle(TinyModel())

    def predict(model, _scaler, scope, _dataset):
        if model is current.model:
            return np.full((2, 2), 5.0 if scope.scope_key == "scope-a" else 9.0)
        return np.full((2, 2), 8.0 if scope.scope_key == "scope-a" else 7.0)

    monkeypatch.setattr(LocalTrainer, "_predict", staticmethod(predict))
    evaluation = LocalTrainer(
        TrainingSettings(
            enforce_performance_gate=True,
            max_scope_wape_regression=0.02,
        )
    )._evaluate(current, candidate.model, candidate.scaler, dataset)

    assert evaluation.aggregate_candidate.value < evaluation.aggregate_current.value
    assert not evaluation.accepted
    assert evaluation.rejection_reasons == (f"scope_regression_exceeded:{'b' * 64}",)


def test_disabled_performance_gate_keeps_evaluation_but_accepts_regression(
    monkeypatch,
):
    dataset = training_dataset()
    current = bundle(TinyModel())
    candidate = bundle(TinyModel())

    def predict(model, _scaler, scope, _dataset):
        assert scope.validation_targets is not None
        offset = 1 if model is current.model else 3
        return scope.validation_targets + offset

    monkeypatch.setattr(LocalTrainer, "_predict", staticmethod(predict))
    evaluation = LocalTrainer(TrainingSettings(enforce_performance_gate=False))._evaluate(
        current, candidate.model, candidate.scaler, dataset
    )

    assert evaluation.aggregate_candidate.value > evaluation.aggregate_current.value
    assert evaluation.accepted
    assert evaluation.rejection_reasons == ()


def test_fedavg_uses_exact_training_sample_counts():
    base = bundle(TinyModel())
    first = bundle(TinyModel())
    second = bundle(TinyModel())
    with torch.no_grad():
        for value in base.model.parameters():
            value.zero_()
        for value in first.model.parameters():
            value.fill_(1.0)
        for value in second.model.parameters():
            value.fill_(3.0)

    aggregate = FederatedTrainer.aggregate(
        base,
        ((first, 1), (second, 3)),
    )

    for value in aggregate.parameters():
        assert torch.allclose(value, torch.full_like(value, 2.5))
