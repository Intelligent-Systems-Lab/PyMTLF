import pytest
from pydantic import ValidationError

from py_mtlf.models import ModelIdentity


def test_model_identity_trims_and_rejects_blank_provider():
    assert ModelIdentity(provider_id=" local ", model_unique_id=1).provider_id == "local"
    with pytest.raises(ValidationError):
        ModelIdentity(provider_id=" ", model_unique_id=1)
