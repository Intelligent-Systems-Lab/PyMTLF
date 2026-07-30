from fastapi.responses import JSONResponse

from py_mtlf.wire.ml_model import ProblemDetails


def problem_response(
    status_code: int,
    title: str,
    detail: str,
    *,
    cause: str = "",
    invalid_params: list[dict[str, str]] | None = None,
) -> JSONResponse:
    problem = ProblemDetails(
        status=status_code,
        title=title,
        detail=detail,
        cause=cause,
        invalidParams=invalid_params or [],
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
        ),
        media_type="application/problem+json",
    )
