import pytest

from server.schemas import EstimateRequest, SearchRequest, TraceRequest


def test_request_schemas_accept_username():
    assert TraceRequest(model_id="m", username="alice").username == "alice"
    assert EstimateRequest(config_content="x", username="bob").username == "bob"
    assert SearchRequest(config_content="x", username="cara").username == "cara"


def test_username_defaults_to_none():
    assert EstimateRequest(config_content="x").username is None
