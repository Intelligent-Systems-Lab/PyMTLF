import pytest
from pydantic import ValidationError

from py_mtlf.models import ModelIdentity


def test_model_identity_uses_only_numeric_standard_identity():
    assert ModelIdentity(model_unique_id=1).model_unique_id == 1
    with pytest.raises(ValidationError):
        ModelIdentity(provider_id="local", model_unique_id=1)
