from fastapi.responses import JSONResponse

from py_mtlf.wire.ml_model import ProblemDetails


def validation_error_invalid_params(errors: list[dict]) -> list[dict[str, str]]:
    invalid_params: list[dict[str, str]] = []
    for error in errors:
        location = tuple(part for part in error.get("loc", ()) if part != "body")
        path = ""
        for part in location:
            if isinstance(part, int):
                path += f"[{part}]"
            elif path:
                path += f".{part}"
            else:
                path = str(part)
        invalid_params.append(
            {
                "param": path,
                "reason": str(error.get("msg", "request validation failed")),
            }
        )
    return invalid_params


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
