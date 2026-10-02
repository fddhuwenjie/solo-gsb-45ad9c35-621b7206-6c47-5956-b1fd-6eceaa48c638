"""旋转舞台走位复核台 —— FastAPI + SQLite 后端。

数据模型（schema v2）
---------------------
- settings：舞台半径；
- entities：演员 / 道具及其安全半径；mounted 标明运动方式：
  * ``rotating``（随台）：站在转台上，随转台一起旋转；
  * ``fixed``（固定）：固定在地面（全局坐标），不随转台旋转；
- keyframes：每个实体按时间记录的关键帧位置 (x, y)。
  随台实体记录的是**转台局部坐标**，固定实体记录的是**全局坐标**；
- turntable_keyframes：整张转台**共享**的角度关键帧 (t_ms, angle)，
  同一时刻所有随台实体共用同一个转台姿态，不再可能各转各的；
- formation_groups / formation_links（v2 新增，队形约束）：
  有向跟随关系 主实体 -> 从属实体，每条关系带相对距离 distance 与
  相对角度 angle（以**主实体坐标架**为准：随台主实体用转台局部架，
  固定主实体用全局架）。编辑主实体关键帧、共享转台姿态或增改跟随关系时，
  沿约束图拓扑传播、为从属实体生成/覆盖关键帧。约束图成环、从属实体被多个
  主实体驱动、传播后关键帧越界，都会**整单拒绝**（HTTP 409）并带冲突链。

计算规则
--------
1. 实体位置在相邻关键帧之间线性插值；
2. 转台角度沿「最短弧」插值（例如 350° -> 10° 走 +20°），
   所有随台实体共用这一个角度；固定道具角度恒为 0、不旋转；
3. 随台实体的局部坐标按转台角度换算为全局坐标：g = R(angle) * p；
   固定实体的关键帧坐标本身就是全局坐标；
4. 每 50ms 采样一次，检查：
   - 任意两个安全圆相交（圆心距 < r1 + r2）；
   - 安全圆越出舞台（|g| + r > 舞台半径）；
5. 同类型、同实体且在时间上相邻的连续命中合并为一个风险区间。

任何编辑请求都会立即重算风险、写入 SQLite，并返回完整状态。
旧版（v0，angle 挂在实体关键帧上）的数据库会在启动时自动迁移。
"""

import json
import math
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "app.db"))
STATIC_DIR = os.path.join(BASE_DIR, "static")

SAMPLE_MS = 50          # 风险采样间隔（毫秒）
EPS = 1e-9
MIN_DURATION_MS = 10_000
KINDS = ("actor", "prop")
MOUNTS = ("rotating", "fixed")
PALETTE = [
    "#4da3ff", "#ffb84d", "#7dd87d", "#ff7d9c",
    "#b388ff", "#4dd0e1", "#f06292", "#aeea00",
]

# v1 起 angle 从实体关键帧剥离，成为全转台共享的 turntable_keyframes；
# entities 增加 mounted（随台 / 固定）。
# v2 起增加 formation_groups / formation_links：有向队形跟随约束。
SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    stage_radius REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    kind    TEXT NOT NULL,
    radius  REAL NOT NULL,
    color   TEXT NOT NULL,
    mounted TEXT NOT NULL DEFAULT 'rotating'
);
CREATE TABLE IF NOT EXISTS keyframes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    t_ms      INTEGER NOT NULL,
    x         REAL NOT NULL,
    y         REAL NOT NULL,
    UNIQUE(entity_id, t_ms)
);
CREATE TABLE IF NOT EXISTS turntable_keyframes (
    t_ms  INTEGER PRIMARY KEY,
    angle REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS risks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT NOT NULL,            -- collision | bounds
    entity_ids TEXT NOT NULL,            -- JSON 数组
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS formation_groups (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL,
    color TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS formation_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER NOT NULL REFERENCES formation_groups(id) ON DELETE CASCADE,
    leader_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    follower_id INTEGER NOT NULL UNIQUE REFERENCES entities(id) ON DELETE CASCADE,
    distance    REAL NOT NULL,
    angle       REAL NOT NULL,
    UNIQUE(group_id, leader_id, follower_id)
);
"""


# --------------------------------------------------------------------------
# 数据库
# --------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    ).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({name})")}


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """把旧库（每个实体关键帧各带一个 angle）迁移到共享转台模型。

    - 实体全部标记为随台（旧模型没有「固定」概念）；
    - 每个时刻的共享转台角度取**最早录入实体**（keyframes.id 最小）
      当时记录的 angle——同一时刻冲突的其它 angle 被归一化覆盖，
      迁移后所有随台实体保证共用同一姿态。
    """
    cols = _table_columns(conn, "entities")
    if "mounted" not in cols:
        conn.execute(
            "ALTER TABLE entities ADD COLUMN mounted TEXT NOT NULL DEFAULT 'rotating'"
        )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turntable_keyframes ("
        "t_ms INTEGER PRIMARY KEY, angle REAL NOT NULL)"
    )
    # 旧库里每个时刻可能有多个互相矛盾的 angle；取最早一条关键帧为准
    conn.execute(
        """
        INSERT OR IGNORE INTO turntable_keyframes (t_ms, angle)
        SELECT k.t_ms, k.angle
        FROM keyframes k
        WHERE k.id = (
            SELECT MIN(k2.id) FROM keyframes k2 WHERE k2.t_ms = k.t_ms
        )
        """
    )
    # 重建 keyframes：去掉 angle 列（x/y 保留；旧实体均随台，故仍是局部坐标）
    conn.executescript(
        """
        CREATE TABLE keyframes_v1 (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            t_ms      INTEGER NOT NULL,
            x         REAL NOT NULL,
            y         REAL NOT NULL,
            UNIQUE(entity_id, t_ms)
        );
        INSERT INTO keyframes_v1 (id, entity_id, t_ms, x, y)
            SELECT id, entity_id, t_ms, x, y FROM keyframes;
        DROP TABLE keyframes;
        ALTER TABLE keyframes_v1 RENAME TO keyframes;
        """
    )
    conn.execute("PRAGMA user_version = 1")


def init_db() -> None:
    with db() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            if _has_table(conn, "entities"):
                _migrate_v0_to_v1(conn)
            else:
                conn.executescript(SCHEMA)
                conn.execute("PRAGMA user_version = 1")
        else:
            conn.executescript(SCHEMA)

        # v2：队形约束表（全新数据库由 SCHEMA 一并建好，这里只做旧库补建）
        if version < 2:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS formation_groups (
                    id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    name  TEXT NOT NULL,
                    color TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS formation_links (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id    INTEGER NOT NULL REFERENCES formation_groups(id) ON DELETE CASCADE,
                    leader_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    follower_id INTEGER NOT NULL UNIQUE REFERENCES entities(id) ON DELETE CASCADE,
                    distance    REAL NOT NULL,
                    angle       REAL NOT NULL,
                    UNIQUE(group_id, leader_id, follower_id)
                );
                """
            )
            conn.execute("PRAGMA user_version = 2")

        if conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0:
            conn.execute("INSERT INTO settings (id, stage_radius) VALUES (1, 5.0)")
        if conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0 \
                and conn.execute("SELECT COUNT(*) FROM turntable_keyframes").fetchone()[0] == 0:
            seed_demo(conn)
        refresh_risks(conn)


def seed_demo(conn: sqlite3.Connection) -> None:
    """首次启动时写入一段有代表性的走位。

    - 两名演员固定在地面（全局坐标）直行，在舞台中心相碰；
    - 转台在 0~8s 沿最短弧转到 180°，桌上的两个随台道具共用这一姿态，
      其中道具桌会扫到演员路线（旋转致碰）；
    - 演员乙末段走出舞台边界（越界）。
    """
    # (name, kind, radius, color, mounted, keyframes[(t, x, y), ...])
    demo = [
        ("演员甲", "actor", 0.6, "#4da3ff", "fixed",
         [(0, -3.0, 0.0), (6000, 3.0, 0.0)]),
        ("演员乙", "actor", 0.6, "#ffb84d", "fixed",
         [(0, 0.0, -3.0), (6000, 0.0, 3.0), (9000, 0.0, 5.4)]),
        # 随台道具只记录局部坐标；角度由共享转台关键帧统一驱动
        ("道具桌", "prop", 1.0, "#b388ff", "rotating",
         [(0, 0.0, 2.6), (8000, 0.0, 2.6)]),
        ("道具灯", "prop", 0.5, "#4dd0e1", "rotating",
         [(0, 2.6, 2.6), (8000, 2.6, 2.6)]),
    ]
    for name, kind, radius, color, mounted, kfs in demo:
        eid = conn.execute(
            "INSERT INTO entities (name, kind, radius, color, mounted) "
            "VALUES (?,?,?,?,?)",
            (name, kind, radius, color, mounted),
        ).lastrowid
        conn.executemany(
            "INSERT INTO keyframes (entity_id, t_ms, x, y) VALUES (?,?,?,?)",
            [(eid, t, x, y) for t, x, y in kfs],
        )
    # 角度用 180°（而非 360°）：最短弧规则下 0° 与 360° 重合，不会转动；
    # 局部 (0, 2.6) 旋转途中经过 (2.6, 0) 附近，会扫过演员甲的行进路线
    conn.executemany(
        "INSERT INTO turntable_keyframes (t_ms, angle) VALUES (?,?)",
        [(0, 0.0), (8000, 180.0)],
    )


# --------------------------------------------------------------------------
# 插值与风险计算（前端 app.js 中有完全对应的实现）
# --------------------------------------------------------------------------

def shortest_arc_delta(a0: float, a1: float) -> float:
    """从角度 a0 到 a1 的最短弧增量，结果落在 (-180, 180]。"""
    return (a1 - a0 + 180.0) % 360.0 - 180.0


def sample_turntable(angles: list[dict], t: int) -> float:
    """t 时刻全转台共享的角度。

    没有任何转台关键帧时恒为 0°；超出范围保持首/末帧；
    中间沿最短弧线性插值。
    """
    if not angles:
        return 0.0
    first, last = angles[0], angles[-1]
    if t <= first["t_ms"]:
        return first["angle"]
    if t >= last["t_ms"]:
        return last["angle"]

    lo, hi = 0, len(angles) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if angles[mid]["t_ms"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = angles[lo], angles[hi]
    f = (t - a["t_ms"]) / (b["t_ms"] - a["t_ms"])
    return a["angle"] + shortest_arc_delta(a["angle"], b["angle"]) * f


def sample_entity(kfs: list[dict], t: int):
    """返回 t 时刻实体关键帧插值后的 (x, y)；无关键帧返回 None。

    随台实体得到的是局部坐标（调用方再按转台角度换算），
    固定实体得到的就是全局坐标。超出范围保持首/末帧。
    """
    if not kfs:
        return None
    first, last = kfs[0], kfs[-1]
    if t <= first["t_ms"]:
        return first["x"], first["y"]
    if t >= last["t_ms"]:
        return last["x"], last["y"]

    lo, hi = 0, len(kfs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if kfs[mid]["t_ms"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = kfs[lo], kfs[hi]
    f = (t - a["t_ms"]) / (b["t_ms"] - a["t_ms"])
    return (
        a["x"] + (b["x"] - a["x"]) * f,
        a["y"] + (b["y"] - a["y"]) * f,
    )


def to_global(x: float, y: float, angle_deg: float):
    """局部坐标 -> 全局坐标（绕舞台中心旋转 angle_deg 度）。"""
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return x * c - y * s, x * s + y * c


def compute_risks(entities: list[dict], turntable: list[dict],
                  stage_radius: float,
                  formations: list[dict] | None = None) -> list[dict]:
    """按 SAMPLE_MS 采样并把连续命中合并为风险区间。

    若给定 formations，位置优先按队形约束图拓扑解析（与舞台预览一致）；
    没有编队关系的实体退化为各自关键帧插值。
    """
    active = [e for e in entities if e["keyframes"]]
    t_max = 0
    for e in active:
        t_max = max(t_max, e["keyframes"][-1]["t_ms"])
    if turntable:
        t_max = max(t_max, turntable[-1]["t_ms"])

    parent, children = build_link_map(formations or [])

    hits: dict[tuple, list[int]] = {}
    t = 0
    while t <= t_max:
        # 转台姿态在同一时刻只有一个，所有随台实体共用
        angle = sample_turntable(turntable, t)
        if formations:
            fpos, _, _, _ = scenario_positions(
                entities, turntable, parent, children, t)
        pos = {}
        for e in active:
            if formations and e["id"] in fpos:
                pos[e["id"]] = fpos[e["id"]]              # 队形拓扑位置
            else:
                x, y = sample_entity(e["keyframes"], t)
                if e["mounted"] == "fixed":
                    pos[e["id"]] = (x, y)                 # 固定道具：不旋转
                else:
                    pos[e["id"]] = to_global(x, y, angle) # 随台实体：共用姿态

        # 越界：安全圆超出舞台边界
        for e in active:
            if e["id"] not in pos:
                continue
            gx, gy = pos[e["id"]]
            if math.hypot(gx, gy) + e["radius"] > stage_radius + EPS:
                hits.setdefault(("bounds", (e["id"],)), []).append(t)

        # 相碰：两个安全圆相交（严格相交，相切不算）
        present = {e["id"]: e for e in active if e["id"] in pos}
        ids = list(present)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                aid, bid = ids[i], ids[j]
                a, b = present[aid], present[bid]
                ax, ay = pos[aid]
                bx, by = pos[bid]
                if math.hypot(ax - bx, ay - by) < a["radius"] + b["radius"] - EPS:
                    hits.setdefault(("collision", (aid, bid)), []).append(t)
        t += SAMPLE_MS

    risks = []
    for (rtype, ids), times in hits.items():
        start = prev = times[0]
        for tt in times[1:]:
            if tt - prev > SAMPLE_MS:          # 采样不连续 -> 新区间
                risks.append(_make_risk(rtype, ids, start, prev))
                start = tt
            prev = tt
        risks.append(_make_risk(rtype, ids, start, prev))

    risks.sort(key=lambda r: (r["start_ms"], r["end_ms"], r["type"]))
    return risks


def _make_risk(rtype: str, ids: tuple, first_hit: int, last_hit: int) -> dict:
    # 命中样本代表该 50ms 窗口内有风险，故区间右端点延长一个采样步长
    return {
        "type": rtype,
        "entity_ids": list(ids),
        "start_ms": first_hit,
        "end_ms": last_hit + SAMPLE_MS,
    }


def refresh_risks(conn: sqlite3.Connection) -> None:
    entities = load_entities(conn)
    turntable = load_turntable(conn)
    formations = load_formations(conn)
    stage_radius = load_settings(conn)["stage_radius"]
    risks = compute_risks(entities, turntable, stage_radius, formations)
    conn.execute("DELETE FROM risks")
    conn.executemany(
        "INSERT INTO risks (type, entity_ids, start_ms, end_ms) VALUES (?,?,?,?)",
        [(r["type"], json.dumps(r["entity_ids"]), r["start_ms"], r["end_ms"])
         for r in risks],
    )


def load_settings(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT stage_radius FROM settings WHERE id = 1").fetchone()
    return {"stage_radius": row["stage_radius"]}


def load_turntable(conn: sqlite3.Connection) -> list[dict]:
    return [
        {"t_ms": r["t_ms"], "angle": r["angle"]}
        for r in conn.execute("SELECT t_ms, angle FROM turntable_keyframes ORDER BY t_ms")
    ]


def load_entities(conn: sqlite3.Connection) -> list[dict]:
    entities = []
    for row in conn.execute("SELECT * FROM entities ORDER BY id"):
        kfs = [
            {"t_ms": k["t_ms"], "x": k["x"], "y": k["y"]}
            for k in conn.execute(
                "SELECT t_ms, x, y FROM keyframes "
                "WHERE entity_id = ? ORDER BY t_ms",
                (row["id"],),
            )
        ]
        entities.append({
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "radius": row["radius"],
            "color": row["color"],
            "mounted": row["mounted"],
            "keyframes": kfs,
        })
    return entities


def load_formations(conn: sqlite3.Connection) -> list[dict]:
    groups = [
        {"id": r["id"], "name": r["name"], "color": r["color"], "links": []}
        for r in conn.execute("SELECT * FROM formation_groups ORDER BY id")
    ]
    gmap = {g["id"]: g for g in groups}
    for r in conn.execute("SELECT * FROM formation_links ORDER BY id"):
        g = gmap.get(r["group_id"])
        if g is None:
            continue
        g["links"].append({
            "id": r["id"],
            "leader_id": r["leader_id"],
            "follower_id": r["follower_id"],
            "distance": r["distance"],
            "angle": r["angle"],
        })
    return groups


def build_state(conn: sqlite3.Connection) -> dict:
    entities = load_entities(conn)
    turntable = load_turntable(conn)
    risks = [
        {
            "id": r["id"],
            "type": r["type"],
            "entity_ids": json.loads(r["entity_ids"]),
            "start_ms": r["start_ms"],
            "end_ms": r["end_ms"],
        }
        for r in conn.execute("SELECT * FROM risks ORDER BY start_ms, id")
    ]
    t_max = 0
    for e in entities:
        if e["keyframes"]:
            t_max = max(t_max, e["keyframes"][-1]["t_ms"])
    if turntable:
        t_max = max(t_max, turntable[-1]["t_ms"])
    duration = max(MIN_DURATION_MS, int(math.ceil((t_max + 1) / 1000.0)) * 1000)
    return {
        "settings": load_settings(conn),
        "entities": entities,
        "turntable_keyframes": turntable,
        "formations": load_formations(conn),
        "risks": risks,
        "duration_ms": duration,
    }


# --------------------------------------------------------------------------
# 队形约束（v2）：有向跟随图、拓扑传播与冲突拒绝
# --------------------------------------------------------------------------

FORM_COLORS = [
    "#ffd54f", "#4dd0e1", "#ff8a65", "#81c784",
    "#ba68c8", "#e57373", "#4db6ac", "#a1887f",
]


def build_link_map(formations: list[dict]):
    """把编队里的关系折叠为约束图：
    parent[follower] = 关系（每个从属实体至多一条入边）；
    children[leader]  = [follower, ...]。
    """
    parent: dict[int, dict] = {}
    children: dict[int, list[int]] = {}
    for g in formations:
        for lk in g["links"]:
            parent[lk["follower_id"]] = lk
            children.setdefault(lk["leader_id"], []).append(lk["follower_id"])
    return parent, children


def candidate_formations(formations: list[dict], group_id: int,
                         link: dict) -> list[dict]:
    """把候选关系并入编队结构，供「投影场景」预算使用（不改动原数据）：
    先移除各编队中同 id 的旧关系，再把候选关系挂到目标编队。"""
    out = []
    for g in formations:
        links = [lk for lk in g["links"] if lk["id"] != link["id"]]
        if g["id"] == group_id:
            links = links + [link]
        out.append({**g, "links": links})
    return out


def detect_cycle(children: dict[int, list[int]]):
    """迭代 DFS 检测有向环，返回环上节点链（首尾同 id）；无环返回 None。"""
    nodes = set(children)
    for fs in children.values():
        nodes.update(fs)
    color = {n: 0 for n in nodes}          # 0 未访问 / 1 在当前路径 / 2 已完成
    for start in list(nodes):
        if color[start] != 0:
            continue
        color[start] = 1
        stack = [(start, 0)]
        path = [start]
        while stack:
            node, idx = stack[-1]
            kids = children.get(node, [])
            if idx < len(kids):
                kid = kids[idx]
                stack[-1] = (node, idx + 1)
                if color.get(kid, 0) == 1:
                    ci = path.index(kid)
                    return path[ci:] + [kid]
                if color.get(kid, 0) == 0:
                    color[kid] = 1
                    stack.append((kid, 0))
                    path.append(kid)
            else:
                color[node] = 2
                stack.pop()
                path.pop()
    return None


def bfs_chain(children: dict[int, list[int]], src: int, dst: int):
    """约束图中 src -> ... -> dst 的一条节点链，找不到返回 None。"""
    if src == dst:
        return [src]
    prev = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for kid in children.get(cur, []):
            if kid in prev:
                continue
            prev[kid] = cur
            if kid == dst:
                chain = [dst]
                while prev[chain[-1]] is not None:
                    chain.append(prev[chain[-1]])
                chain.reverse()
                return chain
            queue.append(kid)
    return None


def root_chain(parent: dict[int, dict], fid: int):
    """从 fid 沿主实体边向上走到根，返回 [root, ..., fid]。"""
    chain = [fid]
    while chain[0] in parent:
        chain.insert(0, parent[chain[0]]["leader_id"])
    return chain


def chain_names(by_id: dict, chain: list[int]) -> list[str]:
    return [by_id[i]["name"] if i in by_id else f"#{i}" for i in chain]


def conflict_exception(code: str, message: str, **extra) -> HTTPException:
    detail = {"code": code, "message": message}
    detail.update(extra)
    return HTTPException(status_code=409, detail=detail)


def validate_candidate_link(by_id: dict, formations: list[dict],
                            leader_id: int, follower_id: int,
                            ignore_link_id: int | None = None) -> None:
    """校验一条（可能尚未落库的）主->从关系：实体存在、不指自己、
    从属实体没有别的主实体（多主拒绝）、加入后约束图不成环。
    违例一律抛 409 并带冲突链。
    """
    if leader_id not in by_id or follower_id not in by_id:
        raise HTTPException(status_code=404, detail="主实体或从属实体不存在")
    if leader_id == follower_id:
        raise conflict_exception(
            "self_link", "主实体与从属实体不能是同一个实体",
            chain=[leader_id], chain_names=chain_names(by_id, [leader_id]),
        )
    groups = [
        {**g, "links": [lk for lk in g["links"] if lk["id"] != ignore_link_id]}
        for g in formations
    ]
    parent, children = build_link_map(groups)

    # 多主：从属实体已经有另一个主实体
    old = parent.get(follower_id)
    if old is not None and old["leader_id"] != leader_id:
        c1 = bfs_chain(children, old["leader_id"], follower_id) \
            or [old["leader_id"], follower_id]
        c2 = [leader_id, follower_id]
        raise conflict_exception(
            "multi_driver",
            f"「{by_id[follower_id]['name']}」已被「{by_id[old['leader_id']]['name']}」"
            f"驱动，不能再由「{by_id[leader_id]['name']}」驱动",
            follower_id=follower_id,
            chains=[c1, c2],
            chains_names=[chain_names(by_id, c1), chain_names(by_id, c2)],
        )

    # 成环检测：同一条边（仅编辑距离/角度）已在图中，不必重复挂载；
    # 新增边临时挂上再检测（整单拒绝时尚未落库）
    if not (old is not None and old["leader_id"] == leader_id):
        children.setdefault(leader_id, []).append(follower_id)
    cyc = detect_cycle(children)
    if cyc is not None:
        raise conflict_exception(
            "cycle",
            "约束图成环：" + " → ".join(chain_names(by_id, cyc)),
            chain=cyc, chain_names=chain_names(by_id, cyc),
        )


def scenario_positions(entities: list[dict], turntable: list[dict],
                       parent: dict[int, dict], children: dict[int, list[int]],
                       t: int, angle_override: float | None = None):
    """在「投影场景」中按约束图拓扑计算 t 时刻各实体全局坐标。

    根实体（无主实体）由自身关键帧插值；从属实体由主实体位置按
    相对距离/角度推出——相对位移以**主实体坐标架**为基准
    （随台主实体：转台局部架；固定主实体：全局架）。

    返回 (pos, grot, gdep, angle)：
    - pos[eid]  = (gx, gy)；
    - grot[eid] = 实体自身坐标架相对全局的角度（随台=转台角，固定=0）；
    - gdep[eid] = 全局位置是否依赖转台角度（沿主实体边继承）；
    关键帧缺失导致整条链解析不出的实体不在结果里。
    """
    angle = angle_override if angle_override is not None \
        else sample_turntable(turntable, t)
    pos: dict[int, tuple] = {}
    grot: dict[int, float] = {}
    gdep: dict[int, bool] = {}
    remaining = {e["id"]: e for e in entities}
    changed = True
    while remaining and changed:
        changed = False
        for eid in list(remaining):
            par = parent.get(eid)
            if par is not None and par["leader_id"] not in pos:
                continue                       # 主实体还没解析，下一轮再来
            e = remaining.pop(eid)
            if par is None:
                sm = sample_entity(e["keyframes"], t)
                if sm is None:
                    continue                   # 根实体无关键帧，链路断开
                x, y = sm
                if e["mounted"] == "fixed":
                    pos[eid], grot[eid], gdep[eid] = (x, y), 0.0, False
                else:
                    pos[eid], grot[eid], gdep[eid] = to_global(x, y, angle), angle, True
            else:
                lid = par["leader_id"]
                gx, gy = pos[lid]
                a = math.radians(grot[lid] + par["angle"])
                pos[eid] = (
                    gx + par["distance"] * math.cos(a),
                    gy + par["distance"] * math.sin(a),
                )
                grot[eid] = angle if e["mounted"] == "rotating" else 0.0
                gdep[eid] = gdep[lid]
            changed = True
    return pos, grot, gdep, angle


def _override_entity_keyframe(entities: list[dict], eid: int,
                              t_ms: int, x: float, y: float) -> None:
    e = next((e for e in entities if e["id"] == eid), None)
    if e is None:
        return
    kfs = e["keyframes"]
    for k in kfs:
        if k["t_ms"] == t_ms:
            k["x"], k["y"] = x, y
            return
    kfs.append({"t_ms": t_ms, "x": x, "y": y})
    kfs.sort(key=lambda k: k["t_ms"])


def stored_coords(e: dict, gx: float, gy: float, angle: float):
    """全局坐标换算为落库坐标：随台实体存转台局部坐标（全局位置按当前
    转台角做逆旋转 R(-angle)），固定实体直接存全局坐标。"""
    if e["mounted"] == "rotating":
        r = math.radians(angle)
        c, s = math.cos(r), math.sin(r)
        return gx * c + gy * s, -gx * s + gy * c
    return gx, gy


def upsert_keyframe_rows(conn: sqlite3.Connection, frames: list[tuple]) -> None:
    """把 (entity_id, t_ms, x, y) 关键帧批量落库，同实体同时刻覆盖。"""
    conn.executemany(
        "INSERT INTO keyframes (entity_id, t_ms, x, y) VALUES (?,?,?,?) "
        "ON CONFLICT(entity_id, t_ms) DO UPDATE SET "
        "x = excluded.x, y = excluded.y",
        frames,
    )


def propagate_edit(entities: list[dict], turntable: list[dict],
                   formations: list[dict], stage_radius: float,
                   t_ms: int, angle_override: float | None,
                   source: str, edited_id: int | None):
    """编辑主实体关键帧（source='leader'）或共享转台姿态（source='turntable'）
    后，计算需要为从属实体生成/覆盖的关键帧。

    越界冲突抛 409（带冲突链）；调用方拿到结果后再统一写库，保证原子提交。
    返回 [(follower_id, x, y), ...]，坐标按从属实体自身语义
    （随台=局部坐标，固定=全局坐标）。
    """
    by_id = {e["id"]: e for e in entities}
    parent, children = build_link_map(formations)
    pos, grot, gdep, angle = scenario_positions(
        entities, turntable, parent, children, t_ms, angle_override)

    if source == "leader":
        # 主实体关键帧：它的全部下游从属实体都要落帧
        targets = []
        stack = list(children.get(edited_id, []))
        seen = set()
        while stack:
            fid = stack.pop()
            if fid in seen:
                continue
            seen.add(fid)
            targets.append(fid)
            stack.extend(children.get(fid, []))
    else:
        # 转台姿态：只需为「全局位置随转台变化」或「自身随台（局部坐标需反算）」
        # 的从属实体落帧；固定主->固定从这种不受转台影响的链不动。
        targets = [
            fid for fid in parent
            if gdep.get(fid) or by_id[fid]["mounted"] == "rotating"
        ]

    updates = []
    for fid in targets:
        if fid not in pos:
            continue                       # 上游缺关键帧，该时刻无法解析
        gx, gy = pos[fid]
        e = by_id[fid]
        if math.hypot(gx, gy) + e["radius"] > stage_radius + EPS:
            if source == "leader":
                chain = bfs_chain(children, edited_id, fid) or [edited_id, fid]
                msg_head = "编辑主实体关键帧"
            else:
                chain = root_chain(parent, fid)
                msg_head = "编辑共享转台姿态"
            raise conflict_exception(
                "out_of_bounds",
                f"{msg_head}后，从属实体「{e['name']}」的传播位置"
                f"({gx:.2f}, {gy:.2f}) 超出舞台（半径 {stage_radius:g}m），"
                f"冲突链：{' → '.join(chain_names(by_id, chain))}",
                chain=chain, chain_names=chain_names(by_id, chain),
                t_ms=t_ms, x=round(gx, 3), y=round(gy, 3),
            )
        lx, ly = stored_coords(e, gx, gy, angle)
        updates.append((fid, round(lx, 3), round(ly, 3)))
    return updates


def propagate_link_edit(entities: list[dict], turntable: list[dict],
                        formations: list[dict], stage_radius: float,
                        leader_id: int, follower_id: int):
    """候选跟随关系（formations 为已并入候选关系的投影）沿链预算从属关键帧。

    时刻集合 = 主实体链（根 -> 直接主实体）已有关键帧 ∪ 相关共享转台关键帧
    （根到从属实体的链上有随台实体时，转台角度才影响传播结果）。逐时刻按
    随台/固定语义计算全部下游从属实体的落帧；任一传播位置越界抛 409
    （带冲突链），调用方据此整单拒绝。返回 [(follower_id, t_ms, x, y), ...]，
    坐标为落库语义（随台=局部坐标，固定=全局坐标）。
    """
    by_id = {e["id"]: e for e in entities}
    parent, children = build_link_map(formations)

    # 受影响从属实体：候选关系的从属实体及其全部下游
    targets = []
    stack = [follower_id]
    seen = set()
    while stack:
        fid = stack.pop()
        if fid in seen:
            continue
        seen.add(fid)
        targets.append(fid)
        stack.extend(children.get(fid, []))

    # 时刻并集：主实体链（根 -> 直接主实体）已有关键帧的时刻
    times = set()
    for eid in root_chain(parent, leader_id):
        e = by_id.get(eid)
        if e:
            times.update(k["t_ms"] for k in e["keyframes"])

    # 相关共享转台关键帧：根到任一受影响从属实体的链上有随台实体时，
    # 转台角度才会改变传播结果或落库坐标
    rotating_involved = any(
        by_id[eid]["mounted"] == "rotating"
        for fid in targets
        for eid in root_chain(parent, fid)
        if eid in by_id
    )
    if rotating_involved:
        times.update(k["t_ms"] for k in turntable)

    updates = []
    for t in sorted(times):
        pos, _grot, _gdep, angle = scenario_positions(
            entities, turntable, parent, children, t)
        for fid in targets:
            if fid not in pos:
                continue                   # 上游缺关键帧，该时刻无法解析
            gx, gy = pos[fid]
            e = by_id[fid]
            if math.hypot(gx, gy) + e["radius"] > stage_radius + EPS:
                chain = root_chain(parent, fid)
                raise conflict_exception(
                    "out_of_bounds",
                    f"编辑跟随关系后，从属实体「{e['name']}」在 {t}ms 的传播位置"
                    f"({gx:.2f}, {gy:.2f}) 超出舞台（半径 {stage_radius:g}m），"
                    f"冲突链：{' → '.join(chain_names(by_id, chain))}",
                    chain=chain, chain_names=chain_names(by_id, chain),
                    t_ms=t, x=round(gx, 3), y=round(gy, 3),
                )
            lx, ly = stored_coords(e, gx, gy, angle)
            updates.append((fid, t, round(lx, 3), round(ly, 3)))
    return updates


# --------------------------------------------------------------------------
# 请求模型
# --------------------------------------------------------------------------

class SettingsIn(BaseModel):
    stage_radius: float = Field(gt=0, le=1000)


class EntityIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    kind: str = "actor"
    radius: float = Field(default=0.5, ge=0.05, le=100)
    color: str | None = None
    mounted: str = "rotating"


class EntityUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=40)
    kind: str | None = None
    radius: float | None = Field(default=None, ge=0.05, le=100)
    color: str | None = None
    mounted: str | None = None


class KeyframeIn(BaseModel):
    t_ms: int = Field(ge=0)
    x: float
    y: float
    # 兼容旧客户端：若带 angle，则同时写入共享转台关键帧
    angle: float | None = None


class TurntableKeyframeIn(BaseModel):
    t_ms: int = Field(ge=0)
    angle: float


class FormationGroupIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    color: str | None = None


class FormationLinkIn(BaseModel):
    leader_id: int
    follower_id: int
    distance: float = Field(gt=0, le=1000)
    angle: float = Field(ge=-360, le=720)


class FormationLinkUpdate(BaseModel):
    distance: float | None = Field(default=None, gt=0, le=1000)
    angle: float | None = Field(default=None, ge=-360, le=720)
    group_id: int | None = None


# --------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="旋转舞台走位复核台", lifespan=lifespan)


@app.get("/api/state")
def get_state():
    with db() as conn:
        return build_state(conn)


@app.put("/api/settings")
def update_settings(payload: SettingsIn):
    with db() as conn:
        conn.execute("UPDATE settings SET stage_radius = ? WHERE id = 1",
                     (payload.stage_radius,))
        refresh_risks(conn)
        return build_state(conn)


@app.post("/api/entities", status_code=201)
def create_entity(payload: EntityIn):
    if payload.kind not in KINDS:
        raise HTTPException(status_code=400, detail="kind 必须是 actor 或 prop")
    if payload.mounted not in MOUNTS:
        raise HTTPException(status_code=400,
                            detail="mounted 必须是 rotating（随台）或 fixed（固定）")
    with db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        color = payload.color or PALETTE[count % len(PALETTE)]
        eid = conn.execute(
            "INSERT INTO entities (name, kind, radius, color, mounted) "
            "VALUES (?,?,?,?,?)",
            (payload.name, payload.kind, payload.radius, color, payload.mounted),
        ).lastrowid
        # 新实体在 t=0 给一个初始关键帧，沿舞台中线圆周错位摆放，便于直接拖动。
        # 随台实体当作局部坐标、固定实体当作全局坐标，数值相同、语义在插值时区分。
        stage_radius = load_settings(conn)["stage_radius"]
        slot = count * 2.39996                 # 黄金角，避免新实体叠在一起
        x = round(stage_radius * 0.5 * math.cos(slot), 2)
        y = round(stage_radius * 0.5 * math.sin(slot), 2)
        conn.execute(
            "INSERT INTO keyframes (entity_id, t_ms, x, y) VALUES (?,?,?,?)",
            (eid, 0, x, y),
        )
        refresh_risks(conn)
        return build_state(conn)


@app.put("/api/entities/{eid}")
def update_entity(eid: int, payload: EntityUpdate):
    fields = payload.model_dump(exclude_none=True)
    if "kind" in fields and fields["kind"] not in KINDS:
        raise HTTPException(status_code=400, detail="kind 必须是 actor 或 prop")
    if "mounted" in fields and fields["mounted"] not in MOUNTS:
        raise HTTPException(status_code=400,
                            detail="mounted 必须是 rotating（随台）或 fixed（固定）")
    with db() as conn:
        row = conn.execute("SELECT id FROM entities WHERE id = ?", (eid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="实体不存在")
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE entities SET {sets} WHERE id = ?",
                         (*fields.values(), eid))
        refresh_risks(conn)
        return build_state(conn)


@app.delete("/api/entities/{eid}")
def delete_entity(eid: int):
    with db() as conn:
        row = conn.execute("SELECT id FROM entities WHERE id = ?", (eid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="实体不存在")
        conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
        refresh_risks(conn)
        return build_state(conn)


@app.put("/api/entities/{eid}/keyframes")
def upsert_keyframe(eid: int, payload: KeyframeIn):
    """新增或更新（实体, 时刻）对应的位置关键帧。

    随台实体为局部坐标、固定实体为全局坐标。若请求携带 angle（旧客户端），
    顺带把该角度写入共享转台关键帧，使旧接口行为保持一致。

    若该实体是队形中的主实体，按随台/固定语义把位置沿约束图传播给全部
    从属实体并生成/覆盖其关键帧；成环/多主（图结构异常）或传播后越界，
    整次编辑（含主实体这一帧）一并拒绝，不写库。
    """
    with db() as conn:
        row = conn.execute("SELECT id FROM entities WHERE id = ?", (eid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="实体不存在")

        # 先在投影场景中把传播结果算完（可能抛 409），通过后再写库：原子提交
        entities = load_entities(conn)
        turntable = load_turntable(conn)
        formations = load_formations(conn)
        stage_radius = load_settings(conn)["stage_radius"]
        _override_entity_keyframe(entities, eid, payload.t_ms, payload.x, payload.y)
        angle_override = payload.angle
        propagated = propagate_edit(
            entities, turntable, formations, stage_radius,
            payload.t_ms, angle_override, "leader", eid,
        )

        conn.execute(
            "INSERT INTO keyframes (entity_id, t_ms, x, y) "
            "VALUES (?,?,?,?) "
            "ON CONFLICT(entity_id, t_ms) DO UPDATE SET "
            "x = excluded.x, y = excluded.y",
            (eid, payload.t_ms, payload.x, payload.y),
        )
        if payload.angle is not None:
            conn.execute(
                "INSERT INTO turntable_keyframes (t_ms, angle) VALUES (?,?) "
                "ON CONFLICT(t_ms) DO UPDATE SET angle = excluded.angle",
                (payload.t_ms, payload.angle),
            )
        upsert_keyframe_rows(
            conn, [(fid, payload.t_ms, x, y) for fid, x, y in propagated])
        refresh_risks(conn)
        return build_state(conn)


@app.delete("/api/entities/{eid}/keyframes/{t_ms}")
def delete_keyframe(eid: int, t_ms: int):
    with db() as conn:
        conn.execute(
            "DELETE FROM keyframes WHERE entity_id = ? AND t_ms = ?",
            (eid, t_ms),
        )
        refresh_risks(conn)
        return build_state(conn)


@app.put("/api/turntable/keyframes")
def upsert_turntable_keyframe(payload: TurntableKeyframeIn):
    """新增/覆盖某个时刻的共享转台角度，全部随台实体同时生效。

    队形约束的从属实体按随台/固定语义被重新定位并生成/覆盖关键帧；
    传播后任何从属实体越界，整次编辑（含转台帧）一并拒绝，不写库。
    """
    with db() as conn:
        # 先在投影场景（转台角替换为新值）中算传播，通过后再写库：原子提交
        entities = load_entities(conn)
        turntable = load_turntable(conn)
        formations = load_formations(conn)
        stage_radius = load_settings(conn)["stage_radius"]
        propagated = propagate_edit(
            entities, turntable, formations, stage_radius,
            payload.t_ms, payload.angle, "turntable", None,
        )

        conn.execute(
            "INSERT INTO turntable_keyframes (t_ms, angle) VALUES (?,?) "
            "ON CONFLICT(t_ms) DO UPDATE SET angle = excluded.angle",
            (payload.t_ms, payload.angle),
        )
        upsert_keyframe_rows(
            conn, [(fid, payload.t_ms, x, y) for fid, x, y in propagated])
        refresh_risks(conn)
        return build_state(conn)


@app.delete("/api/turntable/keyframes/{t_ms}")
def delete_turntable_keyframe(t_ms: int):
    with db() as conn:
        conn.execute("DELETE FROM turntable_keyframes WHERE t_ms = ?", (t_ms,))
        refresh_risks(conn)
        return build_state(conn)


# --------------------------------------------------------------------------
# 队形约束：编队与跟随关系管理
# --------------------------------------------------------------------------

def _color_re_ok(color: str) -> bool:
    return len(color) == 7 and color[0] == "#" \
        and all(c in "0123456789abcdefABCDEF" for c in color[1:])


@app.post("/api/formations", status_code=201)
def create_formation(payload: FormationGroupIn):
    with db() as conn:
        color = payload.color
        if not color or not _color_re_ok(color):
            count = conn.execute("SELECT COUNT(*) FROM formation_groups").fetchone()[0]
            color = FORM_COLORS[count % len(FORM_COLORS)]
        gid = conn.execute(
            "INSERT INTO formation_groups (name, color) VALUES (?,?)",
            (payload.name, color),
        ).lastrowid
        return build_state(conn)


@app.put("/api/formations/{gid}")
def update_formation(gid: int, payload: FormationGroupIn):
    with db() as conn:
        if conn.execute("SELECT id FROM formation_groups WHERE id = ?", (gid,)) \
                .fetchone() is None:
            raise HTTPException(status_code=404, detail="编队不存在")
        color = payload.color or ""
        if not _color_re_ok(color):
            raise HTTPException(status_code=400, detail="颜色需为 #rrggbb 形式")
        conn.execute("UPDATE formation_groups SET name = ?, color = ? WHERE id = ?",
                     (payload.name, color, gid))
        return build_state(conn)


@app.delete("/api/formations/{gid}")
def delete_formation(gid: int):
    with db() as conn:
        if conn.execute("SELECT id FROM formation_groups WHERE id = ?", (gid,)) \
                .fetchone() is None:
            raise HTTPException(status_code=404, detail="编队不存在")
        conn.execute("DELETE FROM formation_groups WHERE id = ?", (gid,))
        refresh_risks(conn)
        return build_state(conn)


@app.post("/api/formations/{gid}/links", status_code=201)
def create_link(gid: int, payload: FormationLinkIn):
    """新增跟随关系：先把候选关系并入约束图，收集主实体链关键帧与相关共享
    转台关键帧的时刻并集，沿链预算全部从属关键帧；成环/多主/自指/传播越界
    整单拒绝（409，关系与关键帧均不变），通过后在同一事务中写入关系、
    落从属帧并重算风险。"""
    with db() as conn:
        if conn.execute("SELECT id FROM formation_groups WHERE id = ?", (gid,)) \
                .fetchone() is None:
            raise HTTPException(status_code=404, detail="编队不存在")
        entities = load_entities(conn)
        by_id = {e["id"]: e for e in entities}
        turntable = load_turntable(conn)
        formations = load_formations(conn)
        stage_radius = load_settings(conn)["stage_radius"]
        # 成环 / 多主 / 自指：整单拒绝并指出冲突链（此时尚未写库）
        validate_candidate_link(
            by_id, formations, payload.leader_id, payload.follower_id)

        # 候选关系并入约束图做投影预算；任一传播位置越界抛 409，
        # 此时尚未写库，关系与关键帧都保持原样
        candidate = candidate_formations(formations, gid, {
            "id": None,
            "leader_id": payload.leader_id,
            "follower_id": payload.follower_id,
            "distance": payload.distance,
            "angle": payload.angle,
        })
        propagated = propagate_link_edit(
            entities, turntable, candidate, stage_radius,
            payload.leader_id, payload.follower_id)

        try:
            conn.execute(
                "INSERT INTO formation_links "
                "(group_id, leader_id, follower_id, distance, angle) "
                "VALUES (?,?,?,?,?)",
                (gid, payload.leader_id, payload.follower_id,
                 payload.distance, payload.angle),
            )
        except sqlite3.IntegrityError:
            # UNIQUE(follower_id)：并发或绕过校验时的最后一道防线
            raise conflict_exception(
                "multi_driver",
                f"从属实体 #{payload.follower_id} 已被其它主实体驱动",
                follower_id=payload.follower_id,
            )
        upsert_keyframe_rows(conn, propagated)
        refresh_risks(conn)
        return build_state(conn)


@app.put("/api/formations/links/{lid}")
def update_link(lid: int, payload: FormationLinkUpdate):
    """修改跟随关系（距离/角度/所属编队）：与新增同样先做投影预算，
    传播越界整单拒绝（409，关系与关键帧均不变），通过后在同一事务中
    更新关系、落从属帧并重算风险。"""
    with db() as conn:
        row = conn.execute("SELECT * FROM formation_links WHERE id = ?", (lid,)) \
            .fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="跟随关系不存在")
        new_group = payload.group_id if payload.group_id is not None else row["group_id"]
        if conn.execute("SELECT id FROM formation_groups WHERE id = ?", (new_group,)) \
                .fetchone() is None:
            raise HTTPException(status_code=404, detail="目标编队不存在")

        if payload.distance is not None or payload.angle is not None \
                or payload.group_id is not None:
            entities = load_entities(conn)
            by_id = {e["id"]: e for e in entities}
            turntable = load_turntable(conn)
            formations = load_formations(conn)
            stage_radius = load_settings(conn)["stage_radius"]
            # 主从不变时同样要重新做一次成环/多主校验（组间转移后图不变，
            # 但保留校验以杜绝脏数据）
            validate_candidate_link(
                by_id, formations, row["leader_id"], row["follower_id"],
                ignore_link_id=lid,
            )
            new_distance = payload.distance if payload.distance is not None \
                else row["distance"]
            new_angle = payload.angle if payload.angle is not None else row["angle"]
            # 候选关系（新距离/角度/编队）并入约束图做投影预算；
            # 任一传播位置越界抛 409，此时尚未写库
            candidate = candidate_formations(formations, new_group, {
                "id": lid,
                "leader_id": row["leader_id"],
                "follower_id": row["follower_id"],
                "distance": new_distance,
                "angle": new_angle,
            })
            propagated = propagate_link_edit(
                entities, turntable, candidate, stage_radius,
                row["leader_id"], row["follower_id"])
            conn.execute(
                "UPDATE formation_links SET group_id = ?, distance = ?, angle = ? "
                "WHERE id = ?",
                (new_group, new_distance, new_angle, lid),
            )
            upsert_keyframe_rows(conn, propagated)
            refresh_risks(conn)
        return build_state(conn)


@app.delete("/api/formations/links/{lid}")
def delete_link(lid: int):
    with db() as conn:
        conn.execute("DELETE FROM formation_links WHERE id = ?", (lid,))
        refresh_risks(conn)
        return build_state(conn)


# 静态前端必须最后挂载，否则会拦截 /api/* 路由
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
