from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from py_mtlf.core.artifacts import (
    ArtifactNotFoundError,
    InvalidArtifactError,
)
from py_mtlf.models import PrivateError

router = APIRouter(prefix="/internal/v1", tags=["artifacts"])


@router.get("/artifacts/{artifact_key}", response_model=None)
async def get_artifact(artifact_key: str, request: Request) -> Response:
    repository = request.app.state.artifacts
    try:
        metadata = repository.metadata(artifact_key)
    except (InvalidArtifactError, ArtifactNotFoundError):
        error = PrivateError(
            code="ARTIFACT_NOT_FOUND",
            message="artifact was not found",
            retryable=False,
            correlation_id=artifact_key,
        )
        return JSONResponse(error.model_dump(mode="json"), status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(
        metadata.path,
        media_type=metadata.media_type,
        headers={
            "ETag": f'"sha256:{metadata.key}"',
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
        content_disposition_type="inline",
    )


@router.get(
    "/fl-artifacts/{process_id}/{participant_id}/{round_indicator}/{role}/{digest}",
    response_model=None,
)
async def get_fl_artifact(
    process_id: str,
    participant_id: str,
    round_indicator: int,
    role: str,
    digest: str,
    request: Request,
) -> Response:
    reader = request.app.state.fl_workspace.open_artifact(
        process_id, participant_id, round_indicator, role, digest
    )
    if reader is None:
        return JSONResponse(
            PrivateError(
                code="ARTIFACT_NOT_FOUND",
                message="federated learning artifact was not found",
                retryable=False,
                correlation_id=digest,
            ).model_dump(mode="json"),
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return StreamingResponse(
        reader.iter_bytes(),
        media_type="application/gzip",
        headers={
            "ETag": f'"sha256:{digest}"',
            "Cache-Control": "public, max-age=3600, immutable",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
        },
    )
