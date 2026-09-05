import gzip
import io
import json
import tarfile
from pathlib import Path

import joblib
import numpy as np

from py_mtlf.core.dataset import DatasetSnapshot
from py_mtlf.core.seed_catalog import ModelVersionKey
from py_mtlf.core.trainer import TrainingResult
from py_mtlf.core.training_data import TrainingDataset, eligibility_manifest


class CandidateBundleBuilder:
    def build(
        self,
        directory: Path,
        *,
        result: TrainingResult,
        dataset: TrainingDataset,
        snapshot: DatasetSnapshot,
        generation: int,
        parent_artifact_key: str,
        model_version_key: ModelVersionKey,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.npy"
        scaler_path = directory / "scaler.pkl"
        model_source_path = directory / "model.py"
        state_values = [
            value.detach().cpu().numpy() for value in result.model.state_dict().values()
        ]
        weights = np.empty(len(state_values), dtype=object)
        weights[:] = state_values
        np.save(model_path, weights, allow_pickle=True)
        joblib.dump(result.scaler, scaler_path)
        model_source_path.write_bytes(result.model_source)

        manifest = result.manifest
        manifest["model_identity"] = {"model_unique_id": model_version_key}
        manifest["model_generation"] = generation
        manifest["parent_artifact_key"] = parent_artifact_key
        manifest["created_at"] = snapshot.time_window.stop_time.isoformat()
        manifest["training"] = {
            **(manifest.get("training") if isinstance(manifest.get("training"), dict) else {}),
            "trainer": "local",
            "single_pass": True,
            "final_loss": result.final_loss,
            "training_data_source": snapshot.source,
            "time_window": snapshot.time_window.model_dump(
                by_alias=True,
                mode="json",
            ),
            "scope_eligibility": eligibility_manifest(dataset),
        }
        manifest["validation_summary"] = {
            "performance_gate_accepted": result.evaluation.accepted,
            "rejection_reasons": list(result.evaluation.rejection_reasons),
            "aggregate": {
                "current_wape": result.evaluation.aggregate_current.value,
                "candidate_wape": result.evaluation.aggregate_candidate.value,
                "absolute_delta": result.evaluation.aggregate_delta,
            },
            "scopes": [
                {
                    "scope_key": item.scope_key,
                    "triggering_scope": item.triggering_scope,
                    "current_wape": item.current.value,
                    "candidate_wape": item.candidate.value,
                    "absolute_delta": item.delta,
                }
                for item in result.evaluation.scopes
            ],
        }
        components = {
            "model.py": model_source_path.read_bytes(),
            "model.npy": model_path.read_bytes(),
            "scaler.pkl": scaler_path.read_bytes(),
        }
        files = {
            "config.json": json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            **components,
        }
        bundle = directory / "candidate.tar.gz"
        with (
            bundle.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
            tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive,
        ):
            for name in sorted(files):
                content = files[name]
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(content))
        return bundle
