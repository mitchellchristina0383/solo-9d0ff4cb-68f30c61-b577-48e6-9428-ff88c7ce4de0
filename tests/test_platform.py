"""端到端自测：基于 ASGI transport 同时跑调度器与两台节点模拟器。

运行：
  PYTHONPATH=/tmp/pylibs PLATFORM_DB=/tmp/platform-test/test.db \
  python3 -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

import httpx
import pytest

# 必须在导入 app 之前配置环境
TEST_DIR = f"/tmp/platform-test-{uuid.uuid4().hex[:8]}"
os.environ["PLATFORM_DB"] = os.path.join(TEST_DIR, "test.db")
os.environ["PLATFORM_NODES"] = (
    "node-a|模拟节点A|http://testnode-a:9001|8192,"
    "node-b|模拟节点B|http://testnode-b:9002|8192"
)
os.environ["NODE_CALL_TIMEOUT"] = "1.0"
os.environ["LOAD_TIMEOUT"] = "4"
os.environ["POLL_INTERVAL"] = "0.2"
os.environ["HEALTH_INTERVAL"] = "3600"  # 测试中手动驱动健康监测

from app.main import app as platform_app          # noqa: E402
from app import config, service                    # noqa: E402
from node import simulator                         # noqa: E402


class RoutingTransport(httpx.AsyncBaseTransport):
    """按 host 把请求路由到对应 ASGI 应用，无需真实端口。"""

    def __init__(self, sim_a, sim_b):
        self._a = httpx.ASGITransport(app=sim_a)
        self._b = httpx.ASGITransport(app=sim_b)
        self._p = httpx.ASGITransport(app=platform_app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "testnode-a":
            return await self._a.handle_async_request(request)
        if host == "testnode-b":
            return await self._b.handle_async_request(request)
        return await self._p.handle_async_request(request)


@pytest.fixture
async def client():
    # 每台模拟器独立实例、独立内存态；每用例全新 SQLite 库
    os.makedirs(TEST_DIR, exist_ok=True)
    if os.path.exists(os.environ["PLATFORM_DB"]):
        os.remove(os.environ["PLATFORM_DB"])
    from app import db as db_mod
    db_mod._conn = None
    # 节点在线内存态同样在用例间重置，避免失联边沿/自动对账串扰
    for rt in service.NODE_RUNTIME.values():
        rt.update(online=None, last_check=None, detail="尚未探测", mode="")
    # ASGI 测试传输下用 0 表示直接抛连接异常；真实 uvicorn 下用长挂起复现超时
    sim_a = simulator.create_app("node-a", "模拟节点A", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    sim_b = simulator.create_app("node-b", "模拟节点B", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    # 每个用例使用全新节点实例，杜绝故障模式/副本/栅栏在测试间残留
    sim_a.state.replicas.clear()
    sim_a.state.fences.clear()
    sim_b.state.replicas.clear()
    sim_b.state.fences.clear()
    transport = RoutingTransport(sim_a, sim_b)
    async with httpx.AsyncClient(base_url="http://platform", transport=transport,
                                 timeout=10) as c:
        # 手动初始化（不走 lifespan），并替换服务里的全局 httpx client
        service.init_db()
        await service.node_client.close()
        service.node_client._client = httpx.AsyncClient(
            base_url="http://platform", transport=transport, timeout=1.0
        )
        # 启动与生产相同的后台轮询（含健康监测，驱动失联检测与恢复自动对账）
        async def poll_loop():
            while True:
                await service.poll_loading()
                await service.poll_releasing()
                await asyncio.sleep(0.2)

        async def health_loop():
            while True:
                await service.health_monitor_once()
                await asyncio.sleep(0.5)

        poller = asyncio.create_task(poll_loop())
        health = asyncio.create_task(health_loop())
        yield c
        poller.cancel()
        health.cancel()
    await service.node_client.close()


async def wait_for(client, dep_id, statuses, timeout=12.0):
    deadline = asyncio.get_event_loop().time() + timeout
    last = None
    while asyncio.get_event_loop().time() < deadline:
        d = await client.get(f"/api/deployments/{dep_id}")
        last = d.json()
        if last["status"] in statuses:
            return last
        await asyncio.sleep(0.25)
    raise AssertionError(f"部署 {dep_id} 未进入 {statuses}，最后状态={last['status']} / {last['reason']}")


def node_free(book, nid):
    return next(n["free_mb"] for n in book if n["node_id"] == nid)


@pytest.mark.asyncio
async def test_01_full_deploy_flow_and_available(client):
    # 两台节点初始各 8192
    book = (await client.get("/api/nodes")).json()
    assert {n["node_id"] for n in book} == {"node-a", "node-b"}
    assert all(n["free_mb"] == 8192 for n in book)

    r = await client.post("/api/deployments", json={
        "model_name": "llama3-8b", "mem_per_replica_mb": 2048, "replica_count": 2})
    assert r.status_code == 201, r.text
    dep_id = r.json()["id"]

    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    # 两个副本必须分处不同节点
    nodes = {rep["node_id"] for rep in dep["replicas"]}
    assert nodes == {"node-a", "node-b"}
    assert all(rep["status"] == "LOADED" for rep in dep["replicas"])

    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 2048 and n["free_mb"] == 6144 for n in book)


@pytest.mark.asyncio
async def test_02_concurrent_reservation_never_overcommits(client):
    # 每台 8192：每副本 3000，双副本部署每台占 3000；并发 3 个 -> 只有 2 个能成功
    async def submit(i):
        return await client.post("/api/deployments", json={
            "model_name": f"m{i}", "mem_per_replica_mb": 3000, "replica_count": 2})

    results = await asyncio.gather(*(submit(i) for i in range(3)))
    codes = [r.status_code for r in results]
    assert codes.count(201) == 2, codes
    assert codes.count(409) == 1, codes

    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] <= n["capacity_mb"]
        assert n["used_mb"] == 6000, n  # 2 个部署 × 3000
        assert n["free_mb"] == 2192

    # 显存不足时再提交应明确拒绝
    r = await client.post("/api/deployments", json={
        "model_name": "big", "mem_per_replica_mb": 4000, "replica_count": 1})
    assert r.status_code == 409
    assert "显存不足" in r.json()["detail"]


@pytest.mark.asyncio
async def test_03_load_fail_releases_reservation_and_retry(client):
    # 让节点 A 加载失败；单副本部署落到剩余显存最多的节点（A/B 相同，排序后取 A）
    await client.post("/api/nodes/node-a/control", json={"load_fail": True})
    r = await client.post("/api/deployments", json={
        "model_name": "fail-model", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"FAILED"})
    rep = dep["replicas"][0]
    assert rep["status"] == "LOAD_FAILED"
    # 失败不占显存
    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 0 for n in book)

    # 清除故障后重试，应成功
    await client.post("/api/nodes/node-a/control",
                      json={"load_fail": False, "mode": "normal"})
    r2 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r2.status_code == 200
    dep2 = await wait_for(client, dep_id, {"AVAILABLE"})
    assert dep2["epoch"] == 2 and dep2["attempts"] == 2
    # 旧 epoch 副本保持终态，不复活
    old = [x for x in dep2["replicas"] if x["epoch"] == 1]
    new = [x for x in dep2["replicas"] if x["epoch"] == 2]
    assert old[0]["status"] == "LOAD_FAILED" and new[0]["status"] == "LOADED"


@pytest.mark.asyncio
async def test_04_unreachable_timeout_keeps_hold_then_reconcile_releases(client):
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "ghost", "mem_per_replica_mb": 1500, "replica_count": 2})
    dep_id = r.json()["id"]

    # A 正常加载，B 失联 -> 部署 UNVERIFIED，B 上显存保留占用
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    await service.health_monitor_once()  # 驱动一次探测周期，确认失联被健康监测捕获
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] == "TIMEOUT"
    # A 可能已加载或仍在加载，但绝不能被标记为失败/释放
    assert by_node["node-a"]["status"] in {"LOADING", "LOADED"}

    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 1500  # 待核期间仍占用，不可被新部署用掉

    # 该显存不能被重新分配：节点 B 剩余不足时提交双副本必被拒
    r2 = await client.post("/api/deployments", json={
        "model_name": "blocked", "mem_per_replica_mb": 7000, "replica_count": 2})
    assert r2.status_code == 409

    # 节点恢复（失联期间请求从未到达模拟器，B 上实际没有该副本）
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 触发恢复自动对账
    # 新语义：恢复后对账查无副本，但加载请求理论上仍可能晚到生效，
    # 必须保持占用、转待核并提示，而不是判失败释放
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    by_node = {x["node_id"]: x for x in rec["replicas"]}
    assert by_node["node-b"]["status"] == "UNVERIFIED"
    assert any("待核" in note or "在途" in note for note in rec["reconcile_notes"])
    book = (await client.get("/api/nodes")).json()
    assert next(n for n in book if n["node_id"] == "node-b")["used_mb"] == 1500

    # 值班员确认放弃 -> 显式撤销：节点栅栏建立并确认无占用后才释放
    rev = (await client.post(f"/api/deployments/{dep_id}/revoke")).json()
    assert "撤销" in "".join(rev["reconcile_notes"])
    dep = await wait_for(client, dep_id, {"FAILED"}, timeout=12)
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] in {"RELEASED", "LOAD_FAILED"}
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 0  # 确认释放后显存可重新分配


@pytest.mark.asyncio
async def test_05_late_ack_must_not_revive_old_replica(client):
    # 场景：节点 A 在加载请求在途时失联，调度器保留占用转待核；
    # 之后节点其实完成了加载（迟到回执/延迟状态变化），恢复后补确认；
    # 再撤销旧加载、重试 epoch=2，旧 epoch 的迟到状态绝不能复活旧副本。

    # 1) 失联期间发起部署 -> TIMEOUT 待核，占用保留
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    await service.health_monitor_once()  # 确认健康监测捕获失联（online=False）
    r = await client.post("/api/deployments", json={
        "model_name": "late", "mem_per_replica_mb": 512, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["status"] == "TIMEOUT"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 512

    # 2) 节点恢复，但它在“失联期间”其实把副本加载好了（模拟迟到回执落地）。
    #    先写入迟到状态再恢复，避免健康循环的自动对账抢先判定为无残留。
    sim_state = client._transport._a.app.state  # type: ignore[attr-defined]
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 探测恢复边沿，触发自动对账
    # 对账补确认 -> AVAILABLE（同一 epoch 的合法补确认，不是复活）
    (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=8)
    assert dep["replicas"][0]["status"] == "LOADED"

    # 3) 再次失联并人工撤销旧加载；失联期间撤销无法确认，恢复后须先建立栅栏再释放
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    deadline = asyncio.get_event_loop().time() + 15
    dep = None
    while asyncio.get_event_loop().time() < deadline:
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"]):
            break
        await asyncio.sleep(0.25)
    assert dep is not None and all(
        x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"])

    # 4) 模拟旧 epoch 的“迟到回执”：节点侧再次出现 epoch=1 loaded
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    # 重试 -> epoch=2；调度器只认当前 epoch，旧副本终态不变
    r3 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r3.status_code == 200
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    assert dep["epoch"] == 2
    old = [x for x in dep["replicas"] if x["epoch"] == 1]
    new = [x for x in dep["replicas"] if x["epoch"] == 2]
    assert all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in old)  # 不复活
    assert new[0]["status"] == "LOADED"
    # 新副本使用不同 rid，节点侧旧记录与新记录互不干扰
    assert new[0]["id"] != rid


@pytest.mark.asyncio
async def test_06_reconcile_failure_stays_unverified_with_reason(client):
    # B 失联 -> 待核；保持失联时手动对账，必须仍是待核并说明原因
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "nv", "mem_per_replica_mb": 900, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    assert any("失联" in note or "不可达" in note for note in rec["reconcile_notes"])
    # 显存仍保留
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 900


@pytest.mark.asyncio
async def test_07_recovery_revokes_stale_load_then_confirms_release(client):
    # 正常部署成功后，节点 B 失联，调度器侧强制转待核并撤销；
    # 恢复后对账必须撤销节点上的旧加载，确认释放后才可重新分配
    r = await client.post("/api/deployments", json={
        "model_name": "svc", "mem_per_replica_mb": 2500, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    b_rep = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    rid = b_rep["id"]

    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    # 直接通过服务层把 B 副本置为待核并请求释放（模拟失联期间人工处置）
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")

    # 失联中对账不能释放，且撤销请求未被节点确认 -> 栅栏未建立、显存继续保留
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192 - 2500
    import app.db as db_mod0
    row0 = db_mod0.query_one("SELECT * FROM replicas WHERE id=?", (rid,))
    assert row0["revoke_fenced_at"] is None  # 未确认的撤销不得释放显存
    # 恢复：节点上旧加载仍在（loaded），自动/手动对账应撤销并最终确认释放。
    # 新语义：必须先等节点受理 unload（建立栅栏），再查询确认无占用才允许 RELEASED。
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()
    deadline = asyncio.get_event_loop().time() + 15
    while asyncio.get_event_loop().time() < deadline:
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
        if b["status"] in {"RELEASED", "LOAD_FAILED"}:
            break
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        await asyncio.sleep(0.25)
    else:
        raise AssertionError("恢复后副本未在时限内确认释放")
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    assert b["status"] == "RELEASED", dep  # 经“栅栏+查询确认”路径释放
    assert b["released_at"] is not None
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192  # 关键：确认释放后显存全额可用

    # 节点侧旧记录应为 unloaded，且重新部署能拿到显存
    r2 = await client.post("/api/deployments", json={
        "model_name": "new-tenant", "mem_per_replica_mb": 7000, "replica_count": 1})
    assert r2.status_code == 201, r2.text


@pytest.mark.asyncio
async def test_08_manual_reconcile_while_load_inflight_keeps_hold(client):
    # 值班场景复现：提交部署后立刻手动对账，加载请求尚未到节点 -> 查无副本。
    # 正确行为：保持显存占用、提示待核，绝不判失败释放；晚到加载仍可补确认。
    await client.post("/api/nodes/node-a/control", json={"gate_load": True})
    r = await client.post("/api/deployments", json={
        "model_name": "inflight", "mem_per_replica_mb": 3000, "replica_count": 1})
    dep_id = r.json()["id"]

    # 等待调度器把加载请求挂在节点入口（节点上尚无副本记录）
    sim_state = client._transport._a.app.state  # type: ignore[attr-defined]
    deadline = asyncio.get_event_loop().time() + 5
    while asyncio.get_event_loop().time() < deadline:
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if dep["replicas"][0]["status"] == "LOADING":
            break
        await asyncio.sleep(0.05)
    assert dep["replicas"][0]["status"] == "LOADING"
    assert dep["replicas"][0]["node_id"] == "node-a"
    # 节点侧确实没有副本记录（请求在途）
    rid = dep["replicas"][0]["id"]
    assert rid not in sim_state.replicas

    # 立即手动对账：查无副本，但必须保持占用、转待核
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED", rec
    rep = rec["replicas"][0]
    assert rep["status"] == "UNVERIFIED"
    assert any("待核" in n or "在途" in n for n in rec["reconcile_notes"])
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 3000  # 显存没有被释放

    # 占用期间并发部署不得超额：双副本要求两台节点都放得下，
    # node-a 仅剩 5192，申请 6000×2 必被拒（node-b 虽空闲也无法完成分散分配）
    blocked = await client.post("/api/deployments", json={
        "model_name": "blocker", "mem_per_replica_mb": 6000, "replica_count": 2})
    assert blocked.status_code == 409
    # 单节点可容纳的申请仍可成功，且只能落在空闲的 node-b，绝不与待核副本争用 node-a
    ok = await client.post("/api/deployments", json={
        "model_name": "elsewhere", "mem_per_replica_mb": 4000, "replica_count": 1})
    assert ok.status_code == 201, ok.text
    dep_ok = await wait_for(client, ok.json()["id"], {"AVAILABLE"}, timeout=8)
    assert dep_ok["replicas"][0]["node_id"] == "node-b"

    # 放行加载请求：晚到加载生效，再次对账应合法补确认 -> AVAILABLE
    await client.post("/api/nodes/node-a/control", json={"gate_load": False})
    deadline = asyncio.get_event_loop().time() + 10
    while asyncio.get_event_loop().time() < deadline:
        rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
        if rec["status"] == "AVAILABLE":
            break
        await asyncio.sleep(0.2)
    assert rec["status"] == "AVAILABLE", rec
    assert rec["replicas"][0]["status"] == "LOADED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 3000


@pytest.mark.asyncio
async def test_09_revoke_fences_late_load_then_releases_only_after_confirmation(client):
    # 加载在途 + 人工撤销：必须先阻止旧请求随后生效（节点栅栏），
    # 节点确认无占用后才释放显存；迟到的加载请求被明确拒绝。
    # 先用单副本小部署占据 node-a，使目标部署确定性落到 node-b
    seat = (await client.post("/api/deployments", json={
        "model_name": "seat", "mem_per_replica_mb": 100, "replica_count": 1})).json()
    await wait_for(client, seat["id"], {"AVAILABLE"}, timeout=8)
    await client.post("/api/nodes/node-b/control", json={"gate_load": True})
    r = await client.post("/api/deployments", json={
        "model_name": "revoke-me", "mem_per_replica_mb": 2200, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"PENDING", "UNVERIFIED"}, timeout=6)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["node_id"] == "node-b"
    sim_b = client._transport._b.app.state  # type: ignore[attr-defined]

    # 先对账转待核（请求仍挂起），保持占用
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["replicas"][0]["status"] == "UNVERIFIED"

    # 显式撤销：unload 先在节点建立栅栏
    rev = (await client.post(f"/api/deployments/{dep_id}/revoke")).json()
    assert rev["replicas"][0]["status"] == "RELEASING"
    import app.db as db_mod
    row = db_mod.query_one("SELECT * FROM replicas WHERE id=?", (rid,))
    assert row["revoke_fenced_at"] is not None  # 节点已受理撤销
    assert sim_b.fences.get(rid) == row["epoch"]
    # 撤销受理但尚未完成查询确认前，显存仍保留占用
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192 - 2200

    # 放行挂起的加载请求：必须被栅栏拒绝（409 fenced），节点不产生占用
    await client.post("/api/nodes/node-b/control", json={"gate_load": False})
    deadline = asyncio.get_event_loop().time() + 10
    while asyncio.get_event_loop().time() < deadline:
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if dep["replicas"][0]["status"] in {"RELEASED", "LOAD_FAILED"}:
            break
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        await asyncio.sleep(0.2)
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["replicas"][0]["status"] == "RELEASED", dep
    # 节点侧没有同 epoch 的加载记录，栅栏仍在
    assert rid not in sim_b.replicas or sim_b.replicas[rid]["status"] != "loaded"
    assert sim_b.fences.get(rid) == dep["replicas"][0]["epoch"]
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192  # 确认无占用后才释放

    # 显存释放后新部署可以使用同一空间
    r2 = await client.post("/api/deployments", json={
        "model_name": "after-release", "mem_per_replica_mb": 7000, "replica_count": 1})
    assert r2.status_code == 201, r2.text


@pytest.mark.asyncio
async def test_10_revoke_while_node_lost_stays_held_until_node_confirms(client):
    # 撤销尝试时节点失联：撤销结果未确认 -> 绝不释放；
    # 恢复后补发撤销、确认无占用才释放（含加载超时语义）。
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "ghost-revoke", "mem_per_replica_mb": 1800, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["status"] in {"TIMEOUT", "UNVERIFIED"}
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 1800

    # 失联下撤销：调用无法确认，栅栏不得建立、显存继续保留
    await client.post(f"/api/deployments/{dep_id}/revoke")
    import app.db as db_mod
    await asyncio.sleep(0.3)
    row = db_mod.query_one("SELECT * FROM replicas WHERE id=?", (rid,))
    assert row["status"] == "RELEASING" and row["revoke_fenced_at"] is None
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 1800  # 未确认绝不释放

    # 失联期间反复对账也不能释放
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    assert any("失联" in n or "不可达" in n for n in rec["reconcile_notes"])

    # 恢复（加载从未到达，节点无记录）：补发撤销建立栅栏 -> 查询确认 -> 释放
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    await service.health_monitor_once()
    deadline = asyncio.get_event_loop().time() + 12
    while asyncio.get_event_loop().time() < deadline:
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if dep["replicas"][0]["status"] in {"RELEASED", "LOAD_FAILED"}:
            break
        await asyncio.sleep(0.2)
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["replicas"][0]["status"] == "RELEASED", dep
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192


@pytest.mark.asyncio
async def test_11_concurrent_deploys_never_overcommit_while_replicas_pending(client):
    # 在“查无副本、保持待核”的窗口内并发提交部署，账本仍不得超额
    await client.post("/api/nodes/node-a/control", json={"gate_load": True})
    await client.post("/api/nodes/node-b/control", json={"gate_load": True})

    async def submit(i):
        return await client.post("/api/deployments", json={
            "model_name": f"race{i}", "mem_per_replica_mb": 4000, "replica_count": 2})

    results = await asyncio.gather(*(submit(i) for i in range(3)))
    codes = sorted(r.status_code for r in results)
    assert codes == [201, 201, 409], codes

    # 全部副本仍卡在在途/待核，但预留已记账：每节点恰好 2×4000
    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] == 8000 and n["free_mb"] == 192, n
    # 待核窗口内再提交仍被拒（剩余 192MB，3000MB 副本无处安放）
    again = await client.post("/api/deployments", json={
        "model_name": "one-more", "mem_per_replica_mb": 3000, "replica_count": 1})
    assert again.status_code == 409

    # 放行后全部正常加载，占用不变、无重复占用
    await client.post("/api/nodes/node-a/control", json={"gate_load": False})
    await client.post("/api/nodes/node-b/control", json={"gate_load": False})
    ok_ids = [r.json()["id"] for r in results if r.status_code == 201]
    for dep_id in ok_ids:
        dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=12)
        assert all(x["status"] == "LOADED" for x in dep["replicas"])
    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] == 8000 and n["free_mb"] == 192

