from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.client import (
    McpServerIdentity,
    McpToolPage,
    SdkConnectedMcpClient,
)
from nemotron.steps.byob.runtime.mcp.config import LoadedMcpOracleConfig, McpOracleConfig
from nemotron.steps.byob.runtime.mcp.discovery import catalog_identity_document
from nemotron.steps.byob.runtime.mcp.gateway import (
    GatewayArtifacts,
    GatewayError,
    GatewayService,
    create_gateway_app,
)
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    BFCL_ORACLE_PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.mcp.gateway.result_mapping import (
    control_result_object,
    map_call_result,
)
from nemotron.steps.byob.runtime.mcp.normalization import normalize_catalog

ZERO_DIGEST = "sha256:" + "0" * 64
SERVER_DIGEST = "sha256:" + "a" * 64
GATEWAY_DIGEST = "sha256:" + "b" * 64

TOOLS = [
    {
        "name": "bfcl.describe",
        "description": "Describe this oracle.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "bfcl.reset",
        "description": "Reset one episode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fixtures": {"type": ["object", "null"]},
                "context": {"type": "object"},
            },
        },
    },
    {
        "name": "bfcl.state",
        "description": "Get state.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "bfcl.end",
        "description": "End episode.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "inventory.lookup",
        "description": "Look up one item.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "episode_id": {"type": "string"},
            },
            "required": ["item_id", "episode_id"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    },
]


def _raw_config(*, tool_timeout_s: float = 1.0) -> dict[str, Any]:
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "A",
        "mcp_protocol_versions": ["2026-07-28"],
        "transport": {
            "kind": "streamable_http",
            "url": "https://mcp.example.test/mcp",
        },
        "expected": {
            "server_name": "catalog",
            "server_version": "1.0.0",
            "tool_catalog_digest": ZERO_DIGEST,
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
            "server_content_digest": SERVER_DIGEST,
        },
        "control": {
            "reset_strategy": "control_tool",
            "state_strategy": "control_tool",
            "describe_oracle": "bfcl.describe",
            "reset_episode": "bfcl.reset",
            "get_episode_state": "bfcl.state",
            "end_episode": "bfcl.end",
            "episode_binding": "argument",
            "episode_argument": "episode_id",
        },
        "fixtures": {"direction": "pushed"},
        "tools": {
            "include": ["inventory.lookup"],
            "aliases": {"inventory.lookup": "inventory_lookup"},
            "mutates": [],
            "requires_confirmation": [],
            "trust_annotations": False,
        },
        "isolation": "namespace_per_episode",
        "limits": {
            "connect_timeout_s": 1,
            "handshake_timeout_s": 1,
            "tool_timeout_s": tool_timeout_s,
            "reset_timeout_s": 1,
            "episode_timeout_s": 30,
            "max_response_bytes": 100_000,
            "max_tools": 16,
            "max_catalog_pages": 4,
            "max_concurrent_episodes": 2,
            "session_idle_ttl_s": 10,
        },
    }


def _loaded(*, tool_timeout_s: float = 1.0) -> LoadedMcpOracleConfig:
    config = McpOracleConfig.model_validate(_raw_config(tool_timeout_s=tool_timeout_s))
    catalog = normalize_catalog(TOOLS, config)
    document = catalog_identity_document(
        config,
        negotiated_mcp_version="2026-07-28",
        server_name="catalog",
        server_version="1.0.0",
        catalog=catalog,
    )
    digest = "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
    config = config.model_copy(update={"expected": config.expected.model_copy(update={"tool_catalog_digest": digest})})
    return LoadedMcpOracleConfig(
        path=Path("/tmp/mcp_oracle.yaml"),
        value=config,
        raw_document=config.model_dump(mode="json", exclude_none=False),
    )


class _FakeClient:
    sdk_version = "2.1.0-test"
    protocol_version = "2026-07-28"
    server_identity = McpServerIdentity("catalog", "1.0.0")
    capabilities = {"tools": {"listChanged": False}}

    def __init__(
        self,
        index: int,
        *,
        hang_business_call: bool = False,
        factory: _Factory | None = None,
    ):
        self.index = index
        self.hang_business_call = hang_business_call
        self.factory = factory
        self.closed = False
        self.business_calls = 0
        self.episode_id: str | None = None
        self.state: dict[str, Any] = {}
        self.reset_context: dict[str, Any] | None = None

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        assert cursor is None
        if self.factory is not None and self.factory.hang_discovery:
            assert self.factory.discovery_started is not None
            assert self.factory.release_discovery is not None
            self.factory.discovery_started.set()
            await self.factory.release_discovery.wait()
        return McpToolPage(tuple(TOOLS), None)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name == "bfcl.describe":
            return {
                "structuredContent": {
                    "oracle_id": "catalog-oracle",
                    "oracle_version": "1.0.0",
                    "content_digest": SERVER_DIGEST,
                }
            }
        if name == "bfcl.reset":
            if self.factory is not None and self.factory.hang_reset:
                assert self.factory.reset_started is not None
                self.factory.reset_started.set()
                await asyncio.sleep(60)
            self.episode_id = f"episode-{self.index}"
            self.reset_context = arguments["context"]
            self.state = {"calls": 0, "fixtures": arguments["fixtures"]}
            if self.factory is not None and self.factory.pause_reset_return:
                assert self.factory.reset_ready is not None
                assert self.factory.release_reset is not None
                self.factory.reset_ready.set()
                await self.factory.release_reset.wait()
            return {"structuredContent": {"episode_id": self.episode_id}}
        if name == "bfcl.state":
            assert arguments["episode_id"] == self.episode_id
            return {"structuredContent": dict(self.state)}
        if name == "bfcl.end":
            assert arguments["episode_id"] == self.episode_id
            return {"structuredContent": {"closed": True}}
        if name == "inventory.lookup":
            self.business_calls += 1
            assert arguments["episode_id"] == self.episode_id
            assert meta is None
            if self.factory is not None and self.factory.block_business_call:
                assert self.factory.business_started is not None
                assert self.factory.release_business is not None
                self.factory.business_started.set()
                await self.factory.release_business.wait()
            if self.hang_business_call:
                await asyncio.sleep(60)
            self.state["calls"] += 1
            return {"structuredContent": {"item": f"{self.episode_id}:{arguments['item_id']}"}}
        raise AssertionError(f"unexpected tool {name}")


class _Factory:
    def __init__(
        self,
        *,
        hang_business_call: bool = False,
        hang_reset: bool = False,
        block_business_call: bool = False,
        hang_discovery: bool = False,
        pause_reset_return: bool = False,
        close_delay_s: float = 0.0,
    ):
        self.hang_business_call = hang_business_call
        self.hang_reset = hang_reset
        self.block_business_call = block_business_call
        self.hang_discovery = hang_discovery
        self.pause_reset_return = pause_reset_return
        self.close_delay_s = close_delay_s
        self.reset_started: asyncio.Event | None = None
        self.business_started: asyncio.Event | None = None
        self.release_business: asyncio.Event | None = None
        self.discovery_started: asyncio.Event | None = None
        self.release_discovery: asyncio.Event | None = None
        self.reset_ready: asyncio.Event | None = None
        self.release_reset: asyncio.Event | None = None
        self.clients: list[_FakeClient] = []
        self.context_owners: list[tuple[asyncio.Task[Any] | None, asyncio.Task[Any] | None]] = []

    def __call__(self, config: McpOracleConfig):
        return self._open(config)

    @asynccontextmanager
    async def _open(self, config: McpOracleConfig):
        entered_by = asyncio.current_task()
        client = _FakeClient(
            len(self.clients),
            hang_business_call=self.hang_business_call,
            factory=self,
        )
        self.clients.append(client)
        try:
            yield client
        finally:
            # Stands in for an episode transport whose shutdown is itself an await, such
            # as the DELETE a Streamable HTTP session sends under terminate_on_close.
            # Client 0 is the startup discovery connection and closes immediately.
            if self.close_delay_s and client.index > 0:
                await asyncio.sleep(self.close_delay_s)
            client.closed = True
            self.context_owners.append((entered_by, asyncio.current_task()))


def _service(
    *,
    factory: _Factory | None = None,
    tool_timeout_s: float = 1.0,
) -> tuple[GatewayService, _Factory]:
    factory = factory or _Factory()
    return (
        GatewayService(
            _loaded(tool_timeout_s=tool_timeout_s),
            artifacts=GatewayArtifacts(GATEWAY_DIGEST),
            connection_factory=factory,
        ),
        factory,
    )


def _context(task_id: str = "task-1") -> dict[str, Any]:
    return {
        "clock": "2026-08-25T12:00:00+00:00",
        "seed": 7,
        "timeout_s": 5,
        "task_id": task_id,
    }


def test_result_mapping_covers_success_error_and_confirmation() -> None:
    config = _loaded().value.results
    assert map_call_result(
        {"structuredContent": {"item": "A"}},
        config=config,
        output_schema=None,
    ) == {"item": "A"}
    assert map_call_result(
        {
            "isError": True,
            "structuredContent": {"error": {"code": "not_found", "id": "A"}},
        },
        config=config,
        output_schema=None,
    ) == {"error": {"code": "not_found", "id": "A"}}
    assert map_call_result(
        {"structuredContent": {"status": "awaiting_confirmation"}},
        config=config,
        output_schema=None,
    ) == {"status": "awaiting_confirmation"}
    # The SDK always dumps the default discriminator, so it must stay mappable.
    assert map_call_result(
        {"resultType": "complete", "isError": False, "structuredContent": {"item": "A"}},
        config=config,
        output_schema=None,
    ) == {"item": "A"}


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"resultType": "input_required"}, "mcp_input_required_unsupported"),
        ({"task": {"taskId": "t-1"}}, "mcp_async_task_unsupported"),
        (
            {"resultType": "vendorExtension", "structuredContent": {"episode_id": "e"}},
            "mcp_unsupported_result_type",
        ),
    ],
)
def test_control_results_refuse_unsupported_shapes(
    payload: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(GatewayError) as caught:
        control_result_object(payload, operation="bfcl.reset")
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"content": [{"type": "text", "text": "{}"}]}, "mcp_result_not_object"),
        (
            {"structuredContent": {"error": {"code": "bad"}}},
            "mcp_error_flag_inconsistent",
        ),
        (
            {"isError": True, "structuredContent": {"error": {"message": "bad"}}},
            "mcp_unstructured_error",
        ),
        (
            {"resultType": "input_required", "structuredContent": {}},
            "mcp_input_required_unsupported",
        ),
        (
            {"task": {"taskId": "t-1", "status": "working"}},
            "mcp_async_task_unsupported",
        ),
        (
            {"resultType": "streamingChunk", "structuredContent": {"item": "A"}},
            "mcp_unsupported_result_type",
        ),
    ],
)
def test_result_mapping_fails_closed(
    payload: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            payload,
            config=_loaded().value.results,
            output_schema=None,
        )
    assert caught.value.code == code
    assert caught.value.http_status >= 500


def test_result_mapping_enforces_output_schema() -> None:
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            {"structuredContent": {"wrong": "shape"}},
            config=_loaded().value.results,
            output_schema=TOOLS[-1]["outputSchema"],
        )
    assert caught.value.code == "mcp_output_schema_mismatch"


def test_gateway_identity_is_deterministic_and_protocol_compatible() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            first = service.metadata()
            second = service.metadata()
            assert first == second
            assert first["protocol_version"] == BFCL_ORACLE_PROTOCOL_VERSION
            assert first["oracle_id"] == "catalog-oracle"
            assert first["content_digest"].startswith("sha256:")
        finally:
            await service.shutdown()
        assert factory.context_owners
        assert all(entered is exited for entered, exited in factory.context_owners)

    asyncio.run(run())


def test_two_sessions_are_isolated_and_cleanup_is_idempotent() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            first = await service.create_session(
                context=_context("first"),
                fixtures={"items": [{"id": "A"}]},
            )
            second = await service.create_session(
                context=_context("second"),
                fixtures={"items": [{"id": "B"}]},
            )
            first_result, second_result = await asyncio.gather(
                service.call_tool(
                    first["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                ),
                service.call_tool(
                    second["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "B"},
                    turn_index=0,
                ),
            )
            assert first_result["item"] != second_result["item"]
            assert (await service.get_state(first["session_id"]))["calls"] == 1
            assert (await service.get_state(second["session_id"]))["calls"] == 1
            await service.delete_session(first["session_id"])
            await service.delete_session(first["session_id"])
            assert factory.clients[1].closed is True
            assert factory.clients[2].closed is False
        finally:
            await service.shutdown()
        assert factory.context_owners
        assert all(entered is exited for entered, exited in factory.context_owners)

    asyncio.run(run())


def test_timeout_poisoning_never_retries_a_business_call() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True),
            tool_timeout_s=0.01,
        )
        await service.start()
        try:
            created = await service.create_session(
                context=_context(),
                fixtures=None,
            )
            with pytest.raises(GatewayError) as caught:
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert caught.value.code == "mcp_call_timeout"
            session_client = factory.clients[1]
            assert session_client.business_calls == 1
            assert session_client.closed is True
            with pytest.raises(GatewayError) as poisoned:
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=1,
                )
            assert poisoned.value.code == "mcp_session_poisoned"
            assert session_client.business_calls == 1
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_poisoning_lets_the_transport_finish_its_own_shutdown() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True, close_delay_s=0.01),
            tool_timeout_s=0.05,
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            with pytest.raises(GatewayError):
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert factory.clients[1].closed is True
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_a_transport_that_will_not_close_is_cancelled_within_the_grace_period() -> None:
    async def run() -> None:
        service, factory = _service(
            factory=_Factory(hang_business_call=True, close_delay_s=60.0),
            tool_timeout_s=0.01,
        )
        await service.start()
        try:
            created = await service.create_session(context=_context(), fixtures=None)
            session = service._sessions[created["session_id"]]
            with pytest.raises(GatewayError):
                await service.call_tool(
                    created["session_id"],
                    name="inventory_lookup",
                    arguments={"item_id": "A"},
                    turn_index=0,
                )
            assert session.closed is True
            assert session.worker.cancelled() is True
            assert factory.clients[1].closed is False
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_reset_receives_the_bfcl_context_verbatim() -> None:
    async def run() -> None:
        service, factory = _service()
        await service.start()
        try:
            context = _context()
            await service.create_session(context=context, fixtures=None)
            assert factory.clients[1].reset_context == context
        finally:
            await service.shutdown()

    asyncio.run(run())


def test_http_adapter_matches_bfcl_oracle_v1_routes() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        metadata = client.get("/v1/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["protocol_version"] == BFCL_ORACLE_PROTOCOL_VERSION
        assert client.get("/v1/tools").json() == {"tools": ["inventory_lookup"]}

        created = client.post(
            "/v1/sessions",
            json={"context": _context(), "fixtures": None},
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        result = client.post(
            f"/v1/sessions/{session_id}/calls",
            json={
                "name": "inventory_lookup",
                "arguments": {"item_id": "A"},
                "turn_index": 0,
            },
        )
        assert result.status_code == 200
        assert result.json()["item"].endswith(":A")
        assert client.get(f"/v1/sessions/{session_id}/state").json()["calls"] == 1
        assert client.delete(f"/v1/sessions/{session_id}").status_code == 204
        assert client.delete(f"/v1/sessions/{session_id}").status_code == 204


def test_http_adapter_rejects_unknown_sessions_and_duplicate_json_keys() -> None:
    service, _ = _service()
    app = create_gateway_app(service)
    with TestClient(app) as client:
        unknown = client.get("/v1/sessions/missing/state")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "mcp_session_unknown"
        duplicate = client.post(
            "/v1/sessions",
            content='{"context":{},"context":{},"fixtures":null}',
            headers={"Content-Type": "application/json"},
        )
        assert duplicate.status_code == 400
        assert duplicate.json()["error"]["code"] == "mcp_request_invalid"


def test_http_adapter_enforces_optional_client_authentication() -> None:
    service, _ = _service()
    app = create_gateway_app(service, client_bearer_token="gateway-secret")
    with TestClient(app) as client:
        assert client.get("/v1/metadata").status_code == 401
        authorized = client.get(
            "/v1/metadata",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        assert authorized.status_code == 200
        # A non-ASCII credential must be rejected, not raise inside the comparison.
        non_ascii = client.get(
            "/v1/metadata",
            headers={b"Authorization": "Bearer gateway-sécret".encode()},
        )
        assert non_ascii.status_code == 401


class _DumpedResult:
    def model_dump(
        self,
        *,
        mode: str,
        by_alias: bool,
        exclude_none: bool,
    ) -> dict[str, Any]:
        assert mode == "json"
        assert by_alias is True
        assert exclude_none is True
        return {
            "resultType": "input_required",
            "inputRequests": {},
            "requestState": "opaque",
        }


class _LowLevelSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None,
        allow_input_required: bool,
        allow_claimed: bool,
    ) -> _DumpedResult:
        self.calls += 1
        assert allow_input_required is True
        assert allow_claimed is True
        return _DumpedResult()


class _RawSdkClient:
    def __init__(self) -> None:
        self.session = _LowLevelSession()


def test_sdk_facade_exposes_input_required_without_automatic_retry() -> None:
    raw = _RawSdkClient()
    facade = SdkConnectedMcpClient(
        raw,
        max_response_bytes=100_000,
        sdk_version="2.1.0-test",
    )
    result = asyncio.run(facade.call_tool("lookup", {"id": "A"}))
    assert raw.session.calls == 1
    with pytest.raises(GatewayError) as caught:
        map_call_result(
            result,
            config=_loaded().value.results,
            output_schema=None,
        )
    assert caught.value.code == "mcp_input_required_unsupported"


def test_cancelled_session_creation_releases_capacity_and_transport() -> None:
    async def run() -> None:
        factory = _Factory(hang_reset=True)
        factory.reset_started = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_started.wait(), timeout=1)
        creation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creation
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert factory.clients[1].closed is True

        factory.hang_reset = False
        created = await service.create_session(context=_context(), fixtures=None)
        await service.delete_session(created["session_id"])
        await service.shutdown()

    asyncio.run(run())


def test_shutdown_cancels_and_awaits_in_progress_session_creation() -> None:
    async def run() -> None:
        factory = _Factory(hang_reset=True)
        factory.reset_started = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_started.wait(), timeout=1)
        await service.shutdown()
        with pytest.raises(GatewayError) as caught:
            await creation
        assert caught.value.code == "mcp_gateway_shutting_down"
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert all(client.closed for client in factory.clients)

    asyncio.run(run())


def test_cancellation_while_registering_a_ready_session_cleans_worker() -> None:
    async def run() -> None:
        factory = _Factory(pause_reset_return=True)
        factory.reset_ready = asyncio.Event()
        factory.release_reset = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        creation = asyncio.create_task(service.create_session(context=_context(), fixtures=None))
        await asyncio.wait_for(factory.reset_ready.wait(), timeout=1)
        await service._registry_lock.acquire()
        factory.release_reset.set()
        await asyncio.sleep(0)
        creation.cancel()
        service._registry_lock.release()
        with pytest.raises(asyncio.CancelledError):
            await creation
        assert service._creating_sessions == 0
        assert not service._starting_workers
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_shutdown_serializes_with_in_progress_startup() -> None:
    async def run() -> None:
        factory = _Factory(hang_discovery=True)
        factory.discovery_started = asyncio.Event()
        factory.release_discovery = asyncio.Event()
        service, _ = _service(factory=factory)
        startup = asyncio.create_task(service.start())
        await asyncio.wait_for(factory.discovery_started.wait(), timeout=1)
        shutdown = asyncio.create_task(service.shutdown())
        factory.release_discovery.set()
        await asyncio.gather(startup, shutdown)
        with pytest.raises(GatewayError) as caught:
            service.metadata()
        assert caught.value.code == "mcp_gateway_not_ready"
        assert all(client.closed for client in factory.clients)

    asyncio.run(run())


def test_expiry_waits_for_an_inflight_call_before_cleanup() -> None:
    async def run() -> None:
        factory = _Factory(block_business_call=True)
        factory.business_started = asyncio.Event()
        factory.release_business = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        created = await service.create_session(context=_context(), fixtures=None)
        session_id = created["session_id"]
        call = asyncio.create_task(
            service.call_tool(
                session_id,
                name="inventory_lookup",
                arguments={"item_id": "A"},
                turn_index=0,
            )
        )
        await asyncio.wait_for(factory.business_started.wait(), timeout=1)
        session = service._sessions[session_id]
        expired_lookup = asyncio.create_task(service.get_state(session_id))
        await asyncio.sleep(0)
        session.created_at -= float(service.config.limits.episode_timeout_s) + 1
        assert factory.clients[1].closed is False

        factory.release_business.set()
        assert (await call)["item"].endswith(":A")
        with pytest.raises(GatewayError) as caught:
            await expired_lookup
        assert caught.value.code == "mcp_session_unknown"
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_cancelled_delete_keeps_teardown_tracked_until_completion() -> None:
    async def run() -> None:
        factory = _Factory(block_business_call=True)
        factory.business_started = asyncio.Event()
        factory.release_business = asyncio.Event()
        service, _ = _service(factory=factory)
        await service.start()
        created = await service.create_session(context=_context(), fixtures=None)
        session_id = created["session_id"]
        call = asyncio.create_task(
            service.call_tool(
                session_id,
                name="inventory_lookup",
                arguments={"item_id": "A"},
                turn_index=0,
            )
        )
        await asyncio.wait_for(factory.business_started.wait(), timeout=1)
        deletion = asyncio.create_task(service.delete_session(session_id))
        await asyncio.sleep(0)
        deletion.cancel()
        with pytest.raises(asyncio.CancelledError):
            await deletion
        assert service._teardown_tasks

        factory.release_business.set()
        await call
        await asyncio.gather(*list(service._teardown_tasks))
        assert factory.clients[1].closed is True
        await service.shutdown()

    asyncio.run(run())


def test_http_adapter_streams_request_limit_and_rejects_nonfinite_json() -> None:
    service, _ = _service()
    app = create_gateway_app(service, max_request_bytes=32)
    with TestClient(app) as client:
        oversized = client.post(
            "/v1/sessions",
            content=(chunk for chunk in (b'{"context":"', b"x" * 64, b'"}')),
            headers={"Content-Type": "application/json"},
        )
        assert oversized.status_code == 413
        nonfinite = client.post(
            "/v1/sessions",
            content=b'{"context":NaN,"fixtures":null}',
            headers={"Content-Type": "application/json"},
        )
        assert nonfinite.status_code == 400
        assert nonfinite.json()["error"]["code"] == "mcp_request_invalid"


@pytest.mark.parametrize(
    "artifacts",
    [
        GatewayArtifacts(GATEWAY_DIGEST, shim_artifact_digest=ZERO_DIGEST),
        GatewayArtifacts(GATEWAY_DIGEST, snapshot_digest=ZERO_DIGEST),
    ],
)
def test_mode_a_rejects_irrelevant_artifact_digests(
    artifacts: GatewayArtifacts,
) -> None:
    async def run() -> None:
        factory = _Factory()
        service = GatewayService(
            _loaded(),
            artifacts=artifacts,
            connection_factory=factory,
        )
        with pytest.raises(GatewayError) as caught:
            await service.start()
        assert caught.value.code == "mcp_gateway_identity_invalid"

    asyncio.run(run())
