from __future__ import annotations

import json
import math
import random
import time
import unittest
from collections import namedtuple
from typing import Any, ClassVar, Literal, cast

import pytest
import httpx
import requests

from cognite.client import CogniteClient, utils
from cognite.client._api_client import APIClient
from cognite.client.config import ClientConfig
from cognite.client.credentials import Token
from cognite.client.data_classes import TimeSeries, TimeSeriesUpdate
from cognite.client.data_classes._base import (
    CogniteFilter,
    CognitePrimitiveUpdate,
    CogniteResource,
    CogniteResourceList,
    CogniteUpdate,
    PropertySpec,
)
from cognite.client.data_classes.hosted_extractors import MQTT5SourceUpdate, MQTT5SourceWrite
from cognite.client.exceptions import CogniteAPIError, CogniteNotFoundError
from cognite.client.utils._identifier import Identifier, IdentifierSequence
from tests.utils import jsgz_load, set_request_limit

BASE_URL = "http://localtest.com/api/1.0/projects/test-project"
URL_PATH = "/someurl"

RESPONSE = {"any": "ok"}


@pytest.fixture(scope="class")
def api_client_with_token_factory(cognite_client):
    return APIClient(
        ClientConfig(
            client_name="any",
            project="test-project",
            base_url=BASE_URL,
            max_workers=1,
            headers={"x-cdp-app": "python-sdk-integration-tests"},
            credentials=Token(lambda: "abc"),
        ),
        api_version=None,
        cognite_client=cognite_client,
    )


@pytest.fixture(scope="class")
def api_client_with_token(cognite_client):
    return APIClient(
        ClientConfig(
            client_name="any",
            project="test-project",
            base_url=BASE_URL,
            max_workers=1,
            headers={"x-cdp-app": "python-sdk-integration-tests"},
            credentials=Token("abc"),
        ),
        api_version=None,
        cognite_client=cognite_client,
    )


RequestCase = namedtuple("RequestCase", ["name", "method", "kwargs"])


class TestBasicRequests:

    request_cases: ClassVar = [
        lambda api_client: RequestCase(
            name="post", method=api_client._post, kwargs={"url_path": URL_PATH, "json": {"any": "ok"}}
        ),
        lambda api_client: RequestCase(name="get", method=api_client._get, kwargs={"url_path": URL_PATH}),
        lambda api_client: RequestCase(name="delete", method=api_client._delete, kwargs={"url_path": URL_PATH}),
        lambda api_client: RequestCase(
            name="put", method=api_client._put, kwargs={"url_path": URL_PATH, "json": {"any": "ok"}}
        ),
    ]

    @pytest.mark.parametrize("fn", request_cases)
    def test_requests_ok(self, fn, httpx_mock, api_client_with_token):
        name, method, kwargs = fn(api_client_with_token)

        httpx_mock.add_response(method=name.upper(), url=BASE_URL + URL_PATH, status_code=200, json=RESPONSE)

        response = method(**kwargs)
        assert response.status_code == 200
        assert response.json() == RESPONSE

        request = httpx_mock.get_requests()[0]
        assert "application/json" == request.headers["content-type"]
        assert "application/json" == request.headers["accept"]
        assert api_client_with_token._config.credentials.authorization_header()[1] == request.headers["Authorization"]
        assert "python-sdk-integration-tests" == request.headers["x-cdp-app"]
        assert "User-Agent" in request.headers

    @pytest.mark.parametrize("fn", request_cases)
    def test_requests_fail(self, fn, httpx_mock, api_client_with_token):
        name, method, kwargs = fn(api_client_with_token)

        httpx_mock.add_response(
            method=name.upper(), url=BASE_URL + URL_PATH, status_code=400, json={"error": "Client error"}
        )
        httpx_mock.add_response(method=name.upper(), url=BASE_URL + URL_PATH, status_code=500, text="Server error")
        httpx_mock.add_response(
            method=name.upper(), url=BASE_URL + URL_PATH, status_code=500, json={"error": "Server error"}
        )
        httpx_mock.add_response(
            method=name.upper(),
            url=BASE_URL + URL_PATH,
            status_code=400,
            json={"error": {"code": 400, "message": "Client error"}},
        )

        with pytest.raises(CogniteAPIError, match="Client error") as e:
            method(**kwargs)
        assert e.value.code == 400

        with pytest.raises(CogniteAPIError, match="Server error") as e:
            method(**kwargs)
        assert e.value.code == 500

        with pytest.raises(CogniteAPIError, match="Server error") as e:
            method(**kwargs)
        assert e.value.code == 500

        with pytest.raises(CogniteAPIError, match="Client error | code: 400 | X-Request-ID:") as e:
            method(**kwargs)
        assert e.value.code == 400
        assert e.value.message == "Client error"

    @pytest.mark.usefixtures("disable_gzip")
    def test_request_gzip_disabled(self, httpx_mock, api_client_with_token):
        def check_gzip_disabled(request):
            assert "Content-Encoding" not in request.headers
            assert {"any": "OK"} == json.loads(request.content)
            return httpx.Response(status_code=200, json=RESPONSE)

        for method in ["PUT", "POST"]:
            httpx_mock.add_callback(check_gzip_disabled, url=BASE_URL + URL_PATH, method=method)

        api_client_with_token._post(URL_PATH, {"any": "OK"}, headers={})
        api_client_with_token._put(URL_PATH, {"any": "OK"}, headers={})

    def test_request_gzip_enabled(self, httpx_mock, api_client_with_token):
        def check_gzip_enabled(request):
            assert "Content-Encoding" in request.headers
            assert {"any": "OK"} == jsgz_load(request.content)
            return httpx.Response(status_code=200, json=RESPONSE)

        for method in ["PUT", "POST"]:
            httpx_mock.add_callback(check_gzip_enabled, url=BASE_URL + URL_PATH, method=method)

        api_client_with_token._post(URL_PATH, {"any": "OK"}, headers={})
        api_client_with_token._put(URL_PATH, {"any": "OK"}, headers={})

    def test_headers_correct(self, httpx_mock, api_client_with_token):
        httpx_mock.add_response(method="POST", url=BASE_URL + URL_PATH, status_code=200, json=RESPONSE)
        api_client_with_token._post(URL_PATH, {"any": "OK"}, headers={"additional": "stuff"})
        request = httpx_mock.get_requests()[0]
        headers = request.headers

        assert "gzip, deflate" in headers["accept-encoding"]
        assert "gzip" == headers["content-encoding"]
        assert f"CognitePythonSDK:{utils._auxiliary.get_current_sdk_version()}" == headers["x-cdp-sdk"]
        assert "Bearer abc" == headers["Authorization"]
        assert "stuff" == headers["additional"]

    def test_headers_correct_with_token_factory(self, httpx_mock, api_client_with_token_factory):
        httpx_mock.add_response(method="POST", url=BASE_URL + URL_PATH, status_code=200, json=RESPONSE)
        api_client_with_token_factory._post(URL_PATH, {"any": "OK"})
        request = httpx_mock.get_requests()[0]
        headers = request.headers

        assert "api-key" not in headers
        assert api_client_with_token_factory._config.credentials.authorization_header()[1] == headers["Authorization"]

    def test_headers_correct_with_token(self, httpx_mock, api_client_with_token):
        httpx_mock.add_response(method="POST", url=BASE_URL + URL_PATH, status_code=200, json=RESPONSE)
        api_client_with_token._post(URL_PATH, {"any": "OK"})
        request = httpx_mock.get_requests()[0]
        headers = request.headers

        assert "api-key" not in headers
        assert api_client_with_token._config.credentials.authorization_header()[1] == headers["Authorization"]

    @pytest.mark.parametrize("payload", [math.nan, math.inf, -math.inf, {"foo": {"bar": {"baz": [[[math.nan]]]}}}])
    def test__do_request_raises_more_verbose_exception(self, api_client_with_token, payload):
        with pytest.raises(ValueError, match=r"contain NaN\(s\) or \+/\- Inf\!"):
            api_client_with_token._do_request("POST", URL_PATH, json=payload)

    def test__do_request_raises_unmodified_exception(self, api_client_with_token):
        # Create circular ref in payload to raise an arbitrary ValueError
        # we want to make sure we _don't_ modify:
        payload = []
        payload.append(payload)
        with pytest.raises(ValueError) as exc_info:
            api_client_with_token._do_request("POST", URL_PATH, json=payload)
        exc_msg = exc_info.value.args[0]
        assert "contain NaN(s) or +/- Inf!" not in exc_msg


class SomeUpdate(CogniteUpdate):
    @property
    def y(self):
        return PrimitiveUpdate(self, "y")

    @property
    def external_id(self):
        return PrimitiveUpdate(self, "externalId")

    @classmethod
    def _get_update_properties(cls, item: CogniteResource | None = None) -> list[PropertySpec]:
        return [PropertySpec("y", is_nullable=False), PropertySpec("external_id", is_nullable=False)]


class PrimitiveUpdate(CognitePrimitiveUpdate):
    def set(self, value: Any) -> SomeUpdate:
        return self._set(value)


class SomeResource(CogniteResource):
    def __init__(self, x=None, y=None, id=None, external_id=None, cognite_client=None):
        self.x = x
        self.y = y
        self.id = id
        self.external_id = external_id
        self._cognite_client = cast("CogniteClient", cognite_client)


class SomeResourceList(CogniteResourceList):
    _RESOURCE = SomeResource


class SomeFilter(CogniteFilter):
    def __init__(self, var_x, var_y):
        self.var_x = var_x
        self.var_y = var_y


class SomeAggregation(CogniteResource):
    def __init__(self, count):
        self.count = count

    @classmethod
    def _load(cls, resource: dict[str, Any], cognite_client: CogniteClient | None = None) -> SomeAggregation:
        return cls(count=resource["count"])


class TestStandardRetrieve:
    def test_standard_retrieve_OK(self, api_client_with_token, httpx_mock):
        httpx_mock.add_response(method="GET", url=BASE_URL + URL_PATH + "/1", status_code=200, json={"x": 1, "y": 2})
        assert SomeResource(1, 2) == api_client_with_token._retrieve(
            cls=SomeResource, resource_path=URL_PATH, identifier=Identifier(1)
        )

    def test_standard_retrieve_not_found(self, api_client_with_token, httpx_mock):
        httpx_mock.add_response(
            method="GET", url=BASE_URL + URL_PATH + "/1", status_code=404, json={"error": {"message": "Not Found."}}
        )
        assert (
            api_client_with_token._retrieve(cls=SomeResource, resource_path=URL_PATH, identifier=Identifier(1)) is None
        )

    def test_standard_retrieve_fail(self, api_client_with_token, httpx_mock):
        httpx_mock.add_response(
            method="GET", url=BASE_URL + URL_PATH + "/1", status_code=400, json={"error": {"message": "Client Error"}}
        )
        with pytest.raises(CogniteAPIError, match="Client Error") as e:
            api_client_with_token._retrieve(cls=SomeResource, resource_path=URL_PATH, identifier=Identifier(1))
        assert "Client Error" == e.value.message
        assert 400 == e.value.code

    def test_cognite_client_is_set(self, cognite_client, api_client_with_token, httpx_mock):
        httpx_mock.add_response(method="GET", url=BASE_URL + URL_PATH + "/1", status_code=200, json={"x": 1, "y": 2})
        res = api_client_with_token._retrieve(cls=SomeResource, resource_path=URL_PATH, identifier=Identifier(1))
        assert cognite_client == res._cognite_client
