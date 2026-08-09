from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from py_mtlf.models import TrainingDataDescriptor

router = APIRouter(prefix="/internal/v1/anlf/training-data-descriptors", tags=["training-data"])


@router.put("/{descriptor_id}", status_code=status.HTTP_204_NO_CONTENT)
def put_training_data_descriptor(
    descriptor_id: str,
    descriptor: TrainingDataDescriptor,
    request: Request,
) -> Response:
    normalized_id = str(UUID(descriptor_id))
    request.app.state.dataset_coordinator.put_training_data_descriptor(
        normalized_id,
        descriptor,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{descriptor_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_training_data_descriptor(descriptor_id: str, request: Request) -> Response:
    request.app.state.dataset_coordinator.delete_training_data_descriptor(str(UUID(descriptor_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
