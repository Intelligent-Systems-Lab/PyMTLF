from pydantic import BaseModel, ConfigDict, Field


class SelectedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    nf_instance_id: str = Field(min_length=1, alias="nfInstanceId")
    nf_service_instance_id: str = Field(min_length=1, alias="nfServiceInstanceId")
    service_name: str = Field(min_length=1, alias="serviceName")
    api_root: str = Field(min_length=1, alias="apiRoot")
    selection_source: str = Field(min_length=1, alias="selectionSource")


TARGET_NF_INSTANCE_ID_HEADER = "X-NWDAF-Target-Nf-Instance-Id"
TARGET_NF_SERVICE_INSTANCE_ID_HEADER = "X-NWDAF-Target-Nf-Service-Instance-Id"
TARGET_API_ROOT_HEADER = "X-NWDAF-Target-Api-Root"
TARGET_SELECTION_SOURCE_HEADER = "X-NWDAF-Target-Selection-Source"


def selected_target_headers(target: SelectedTarget | None) -> dict[str, str]:
    if target is None:
        return {}
    return {
        TARGET_NF_INSTANCE_ID_HEADER: target.nf_instance_id,
        TARGET_NF_SERVICE_INSTANCE_ID_HEADER: target.nf_service_instance_id,
        TARGET_API_ROOT_HEADER: target.api_root,
        TARGET_SELECTION_SOURCE_HEADER: target.selection_source,
    }
