import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from py_mtlf.wire.ml_model_training import NwdafMLModelTrainSubsc


class TrainingScopeDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_subscription: dict[str, Any] = Field(alias="eventSubscription")
    target_reporting_ue: dict[str, Any] | None = Field(
        default=None,
        alias="targetReportingUe",
    )
    requested_time_windows: list[dict[str, Any]] = Field(
        default_factory=list,
        alias="requestedTimeWindows",
    )
    scope_digest: str = Field(alias="scopeDigest", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_training_request(
        cls,
        request: NwdafMLModelTrainSubsc,
        event_index: int,
    ) -> "TrainingScopeDescriptor":
        try:
            event = request.ml_event_subscriptions[event_index]
        except IndexError as exc:
            raise ValueError("event_index is outside mLEventSubscs") from exc

        event_subscription = event.model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
        )
        target_reporting_ue = request.target_reporting_ue
        requested_time_windows: list[dict[str, Any]] = []
        for info in request.ml_model_training_infos or []:
            data_requirement = info.data_availability_requirement
            if data_requirement is None:
                continue
            for window in data_requirement.time_windows or []:
                requested_time_windows.append(
                    window.model_dump(
                        by_alias=True,
                        exclude_none=True,
                        mode="json",
                    )
                )

        canonical = {
            "eventSubscription": event_subscription,
            "requestedTimeWindows": requested_time_windows,
        }
        if target_reporting_ue is not None:
            canonical["targetReportingUe"] = target_reporting_ue
        digest = hashlib.sha256(_canonical_json(canonical)).hexdigest()
        return cls(
            eventSubscription=event_subscription,
            targetReportingUe=target_reporting_ue,
            requestedTimeWindows=requested_time_windows,
            scopeDigest=digest,
        )

    def canonical_payload(self) -> bytes:
        value = {
            "eventSubscription": self.event_subscription,
            "requestedTimeWindows": self.requested_time_windows,
        }
        if self.target_reporting_ue is not None:
            value["targetReportingUe"] = self.target_reporting_ue
        return _canonical_json(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
