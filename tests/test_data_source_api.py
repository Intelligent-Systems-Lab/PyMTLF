import pytest
from fastapi.testclient import TestClient

from py_mtlf.app import create_app


@pytest.mark.parametrize(
    ("mode", "available"),
    [
        ("adrf", ["adrf"]),
        ("mongodb", ["mongodb"]),
        ("dual", ["adrf", "mongodb"]),
    ],
)
def test_selects_configured_storage_mode(settings, mode, available):
    configured = settings.model_copy(update={
        "data_source": settings.data_source.model_copy(update={"storage_mode": mode})
    })
    app = create_app(configured)

    with TestClient(app) as client:
        first = client.post(
            "/internal/v1/data-source-selection",
            json={"availableDataSources": available},
        )
        second = client.post(
            "/internal/v1/data-source-selection",
            json={"availableDataSources": list(reversed(available))},
        )

    assert first.status_code == 200
    assert first.json() == {"storageMode": mode}
    assert second.json() == first.json()
    assert app.state.runtime.selected_storage_mode == mode


@pytest.mark.parametrize(
    ("mode", "available"),
    [
        ("adrf", []),
        ("mongodb", ["adrf"]),
        ("dual", ["mongodb"]),
    ],
)
def test_missing_required_source_returns_retryable_conflict(settings, mode, available):
    configured = settings.model_copy(update={
        "data_source": settings.data_source.model_copy(update={"storage_mode": mode})
    })

    with TestClient(create_app(configured)) as client:
        response = client.post(
            "/internal/v1/data-source-selection",
            json={"availableDataSources": available},
        )

    assert response.status_code == 409
    assert response.json() == {
        "code": "DATA_SOURCE_REQUIREMENT_UNSATISFIED",
        "message": "configured storage mode requires unavailable data source",
        "retryable": True,
    }


@pytest.mark.parametrize(
    "available",
    [
        ["mongodb", "mongodb"],
        ["unknown"],
    ],
)
def test_invalid_source_inventory_is_rejected(settings, available):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/internal/v1/data-source-selection",
            json={"availableDataSources": available},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
