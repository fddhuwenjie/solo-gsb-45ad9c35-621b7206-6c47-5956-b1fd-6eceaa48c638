'use strict';
/* 旋转舞台走位复核台 —— 原生 JS 前端。
 * 姿态模型（与 backend/main.py 完全对应）：
 * - 转台角度是按时间排列的【共享】关键帧 state.turntable_keyframes；
 * - 随台实体（mounted=rotating）记录局部坐标，换算时共用同一角度；
 * - 固定实体（mounted=fixed）记录全局坐标，不随转台旋转。 */

const $ = (s) => document.querySelector(s);

const SAMPLE_MS = 50;

let state = null;
const ui = {
  time: 0,
  playing: false,
  selectedId: null,
  dragId: null,        // 舞台上正在拖动的实体
  dragKf: null,        // 拖动中最近一次关键帧数据
  scrubbing: false,    // 时间轴拖动中
  lastTs: null,
  selectedFormationId: null,
  lastConflict: null,  // 最近一次整单拒绝（409）的结构化信息
};

/* ================= 数学：与后端一致 ================= */

function shortestArcDelta(a0, a1) {
  // 结果落在 (-180, 180]
  return ((((a1 - a0) % 360) + 540) % 360) - 180;
}

function sampleTurntable(angles, t) {
  // 全转台共享的当前角度；无关键帧恒为 0；范围外保持首/末帧；中间走最短弧
  if (!angles.length) return 0;
  if (t <= angles[0].t_ms) return angles[0].angle;
  const last = angles[angles.length - 1];
  if (t >= last.t_ms) return last.angle;

  let lo = 0, hi = angles.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (angles[mid].t_ms <= t) lo = mid; else hi = mid;
  }
  const a = angles[lo], b = angles[hi];
  const f = (t - a.t_ms) / (b.t_ms - a.t_ms);
  return a.angle + shortestArcDelta(a.angle, b.angle) * f;
}

function sampleEntity(ent, t) {
  const kfs = ent.keyframes;
  if (!kfs.length) return null;
  if (t <= kfs[0].t_ms) return { x: kfs[0].x, y: kfs[0].y };
  const last = kfs[kfs.length - 1];
  if (t >= last.t_ms) return { x: last.x, y: last.y };

  let lo = 0, hi = kfs.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (kfs[mid].t_ms <= t) lo = mid; else hi = mid;
  }
  const a = kfs[lo], b = kfs[hi];
  const f = (t - a.t_ms) / (b.t_ms - a.t_ms);
  return {
    x: a.x + (b.x - a.x) * f,
    y: a.y + (b.y - a.y) * f,
  };
}

function toGlobal(x, y, angleDeg) {
  const r = (angleDeg * Math.PI) / 180;
  const c = Math.cos(r), s = Math.sin(r);
  return [x * c - y * s, x * s + y * c];
}

function toLocal(gx, gy, angleDeg) {
  // 全局 -> 转台局部坐标（反向旋转）
  const r = (-angleDeg * Math.PI) / 180;
  const c = Math.cos(r), s = Math.sin(r);
  return [gx * c - gy * s, gx * s + gy * c];
}

function positionsAt(t) {
  // 同一时刻只有一个转台角度，所有随台实体共用
  const angle = sampleTurntable(state.turntable_keyframes, t);
  const out = {};
  for (const e of state.entities) {
    const sm = sampleEntity(e, t);
    if (!sm) continue;
    if (e.mounted === 'fixed') {
      out[e.id] = { gx: sm.x, gy: sm.y, angle: 0 };          // 固定：不旋转
    } else {
      const [gx, gy] = toGlobal(sm.x, sm.y, angle);
      out[e.id] = { gx, gy, angle };                         // 随台：共用姿态
    }
  }
  return out;
}

/* ---- 队形约束：与后端 propagate_edit 同一套语义 ---- */

function formationMaps() {
  // parent[follower] = link；children[leader] = [follower,...]
  const parent = new Map();
  const children = new Map();
  for (const g of state.formations || []) {
    for (const lk of g.links) {
      parent.set(lk.follower_id, lk);
      if (!children.has(lk.leader_id)) children.set(lk.leader_id, []);
      children.get(lk.leader_id).push(lk.follower_id);
    }
  }
  return { parent, children };
}

function descendantIds(leaderId, children) {
  const out = [];
  const seen = new Set([leaderId]);
  const stack = [...(children.get(leaderId) || [])];
  while (stack.length) {
    const fid = stack.pop();
    if (seen.has(fid)) continue;
    seen.add(fid);
    out.push(fid);
    stack.push(...(children.get(fid) || []));
  }
  return out;
}

/* 在「投影场景」里按约束图拓扑计算全局坐标（与后端 scenario_positions 对应）。
 * override: {entityId: {gx, gy}} 用拖动中的全局位置替换该实体；
 * 返回 {pos: Map(id -> {gx,gy,angle}), gdep: Map(id -> bool)}。 */
function formationPositionsAt(t, override = null) {
  const angle = sampleTurntable(state.turntable_keyframes, t);
  const { parent } = formationMaps();
  const pos = new Map();
  const gdep = new Map();
  const remaining = new Map(state.entities.map((e) => [e.id, e]));
  let changed = true;
  while (remaining.size && changed) {
    changed = false;
    for (const eid of [...remaining.keys()]) {
      const par = parent.get(eid);
      if (par && !pos.has(par.leader_id)) continue;
      const e = remaining.get(eid);
      remaining.delete(eid);
      let gx, gy, ea, dep;
      if (override && override[eid]) {
        gx = override[eid].gx; gy = override[eid].gy;
        ea = e.mounted === 'fixed' ? 0 : angle;
        dep = e.mounted === 'rotating';
      } else if (!par) {
        const sm = sampleEntity(e, t);
        if (!sm) continue;                     // 根实体无关键帧，链路断开
        if (e.mounted === 'fixed') { gx = sm.x; gy = sm.y; ea = 0; dep = false; }
        else { [gx, gy] = toGlobal(sm.x, sm.y, angle); ea = angle; dep = true; }
      } else {
        const lp = pos.get(par.leader_id);
        const rad = ((lp.angle ?? 0) + par.angle) * Math.PI / 180;
        gx = lp.gx + par.distance * Math.cos(rad);
        gy = lp.gy + par.distance * Math.sin(rad);
        ea = e.mounted === 'fixed' ? 0 : angle;
        dep = gdep.get(par.leader_id);
      }
      pos.set(eid, { gx, gy, angle: ea });
      gdep.set(eid, dep);
      changed = true;
    }
  }
  return { pos, gdep, angle };
}

/* ================= 工具 ================= */

const round2 = (v) => Math.round(v * 100) / 100;
const snap50 = (t) => Math.max(0, Math.round(t / SAMPLE_MS) * SAMPLE_MS);

function fmt(t) {
  const m = Math.floor(t / 60000);
  const s = (t - m * 60000) / 1000;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

const entityById = (id) => state.entities.find((e) => e.id === id);
const mountedText = (m) => (m === 'fixed' ? '固定' : '随台');

function activeRisks() {
  return state.risks.filter((r) => r.start_ms <= ui.time && ui.time <= r.end_ms);
}

function riskNames(r) {
  return r.entity_ids
    .map((id) => { const e = entityById(id); return e ? e.name : `#${id}`; })
    .join(' × ');
}

/* ================= API ================= */

async function api(path, method = 'GET', body = null) {
  const opt = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== null) opt.body = JSON.stringify(body);
  const res = await fetch('/api' + path, opt);
  if (!res.ok) {
    let detail = '请求失败';
    let structured = null;
    try {
      const j = await res.json();
      if (j && typeof j.detail === 'object' && j.detail.message) {
        structured = j.detail;
        detail = j.detail.message;
      } else {
        detail = (j && j.detail) || detail;
      }
    } catch (_) { /* ignore */ }
    showConflict(structured || detail);
    const err = new Error(detail);
    err.status = res.status;
    err.detail = structured;
    throw err;
  }
  clearConflict();
  return res.json();
}

/* ---- 冲突横幅（整单拒绝时展示冲突链） ---- */

function conflictChainText(detail) {
  if (!detail || typeof detail !== 'object') return '';
  if (detail.code === 'multi_driver' && detail.chains_names) {
    return detail.chains_names.map((c) => c.join(' → ')).join('　｜　');
  }
  if (detail.chain_names) return detail.chain_names.join(' → ');
  return '';
}

function showConflict(detail) {
  let banner = $('#conflictBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'conflictBanner';
    banner.className = 'conflict-banner';
    banner.innerHTML = '<span class="cb-close">✕</span><div class="cb-text"></div>';
    banner.addEventListener('click', clearConflict);
    $('#stageWrap').appendChild(banner);
  }
  const text = typeof detail === 'string' ? detail : detail.message || '请求被拒绝';
  const chain = typeof detail === 'string' ? '' : conflictChainText(detail);
  const codeMap = {
    cycle: '约束图成环',
    multi_driver: '从属实体被多个主实体驱动',
    out_of_bounds: '传播后关键帧超出舞台',
    self_link: '主从不能为同一实体',
  };
  const tag = typeof detail === 'object' && codeMap[detail.code]
    ? `【整单拒绝 · ${codeMap[detail.code]}】` : '【请求被拒绝】';
  banner.querySelector('.cb-text').innerHTML =
    `${esc(tag)} ${esc(text)}${chain ? `<br>冲突链：<span class="chain">${esc(chain)}</span>` : ''}`;
  banner.classList.remove('hidden');
}

function clearConflict() {
  const banner = $('#conflictBanner');
  if (banner) banner.classList.add('hidden');
}

function afterState() {
  if (ui.selectedId != null && !state.entities.some((e) => e.id === ui.selectedId)) {
    ui.selectedId = null;
  }
  if (ui.selectedId == null && state.entities.length) {
    ui.selectedId = state.entities[0].id;
  }
  if (ui.selectedFormationId != null &&
      !(state.formations || []).some((g) => g.id === ui.selectedFormationId)) {
    ui.selectedFormationId = null;
  }
  if (ui.selectedFormationId == null && (state.formations || []).length) {
    ui.selectedFormationId = state.formations[0].id;
  }
  // 编辑可能缩短总时长（删除末关键帧），钳制播放头避免越出时间轴
  if (ui.time > state.duration_ms) ui.time = state.duration_ms;
  renderSidebar();
  renderDynamic();
}

async function refresh() {
  state = await api('/state');
  afterState();
}

function upsertLocal(ent, kf) {
  const i = ent.keyframes.findIndex((k) => k.t_ms === kf.t_ms);
  if (i >= 0) ent.keyframes[i] = kf;
  else {
    ent.keyframes.push(kf);
    ent.keyframes.sort((a, b) => a.t_ms - b.t_ms);
  }
}

/* ================= 播放控制 ================= */

function play() {
  if (ui.playing) return;
  if (ui.time >= state.duration_ms) ui.time = 0;
  ui.playing = true;
  ui.lastTs = null;
  $('#btnPlay').textContent = '⏸ 暂停';
  requestAnimationFrame(tick);
}

function pause() {
  ui.playing = false;
  $('#btnPlay').textContent = '▶ 播放';
}

function tick(ts) {
  if (!ui.playing) return;
  if (ui.lastTs == null) ui.lastTs = ts;
  ui.time += ts - ui.lastTs;
  ui.lastTs = ts;
  if (ui.time >= state.duration_ms) {
    ui.time = state.duration_ms;
    pause();
  }
  renderDynamic();
  if (ui.playing) requestAnimationFrame(tick);
}

function seek(t) {
  ui.time = Math.min(Math.max(0, t), state.duration_ms);
  renderDynamic();
}

function jumpRisk(dir) {
  if (!state.risks.length) return;
  const t = ui.time;
  let target = null;
  if (dir > 0) {
    target = state.risks.find((r) => r.start_ms > t + 1) || state.risks[0];
  } else {
    const before = state.risks.filter((r) => r.start_ms < t - 1);
    target = before.length ? before[before.length - 1] : state.risks[state.risks.length - 1];
  }
  seek(target.start_ms);
}

/* ================= 侧栏渲染 ================= */

function renderSidebar() {
  $('#stageRadius').value = state.settings.stage_radius;

  const { parent, children } = formationMaps();
  const ul = $('#entityList');
  if (state.entities.length) {
    ul.innerHTML = state.entities.map((e) => {
      const roles = [];
      if (children.has(e.id)) roles.push('<span class="badge role leader">主</span>');
      if (parent.has(e.id)) roles.push('<span class="badge role follow">从</span>');
      return `
      <li data-id="${e.id}" class="${e.id === ui.selectedId ? 'selected' : ''}">
        <span class="dot" style="background:${e.color}"></span>
        <span class="r-list-name">${esc(e.name)}</span>
        ${roles.join('')}
        <span class="badge">${e.kind === 'actor' ? '演员' : '道具'}</span>
        <span class="badge mount ${e.mounted === 'fixed' ? 'fixed' : 'rot'}">${mountedText(e.mounted)}</span>
        <span class="muted">r=${e.radius}</span>
      </li>`;
    }).join('');
  } else {
    ul.innerHTML = '<li class="muted">暂无实体，点击上方按钮添加</li>';
  }

  const ent = entityById(ui.selectedId);
  $('#entityEditor').classList.toggle('hidden', !ent);
  if (ent) {
    $('#edName').value = ent.name;
    $('#edKind').value = ent.kind;
    $('#edMounted').value = ent.mounted;
    $('#edRadius').value = ent.radius;
    $('#edColor').value = ent.color;
  }
  renderTurntablePanel();
  renderKfPanel();
  renderFormationPanel();
  renderRiskList();
}

function renderTurntablePanel() {
  const t = snap50(ui.time);
  const kf = state.turntable_keyframes.find((k) => k.t_ms === t);
  $('#btnTfDelete').classList.toggle('hidden', !kf);
  $('#tfChips').innerHTML = state.turntable_keyframes
    .map((k) => `<span class="chip tf-chip" data-t="${k.t_ms}" title="转台角度 ${k.angle}°">
        ${fmt(k.t_ms)} ${k.angle.toFixed(0)}°<b class="x" data-tf-del="${k.t_ms}">×</b></span>`)
    .join('');
  // 用户正在输入时不覆盖
  if (document.activeElement.id !== 'tfAngle') {
    const ang = kf
      ? kf.angle
      : sampleTurntable(state.turntable_keyframes, ui.time);
    $('#tfAngle').value = ang.toFixed(1);
  }
}

function renderKfPanel() {
  const ent = entityById(ui.selectedId);
  $('#kfNone').classList.toggle('hidden', !!ent);
  $('#kfForm').classList.toggle('hidden', !ent);
  if (!ent) return;
  const fixed = ent.mounted === 'fixed';
  $('#kfLabelX').textContent = fixed ? '全局 X (m)' : '局部 X (m)';
  $('#kfLabelY').textContent = fixed ? '全局 Y (m)' : '局部 Y (m)';
  const exists = ent.keyframes.some((k) => k.t_ms === snap50(ui.time));
  $('#btnKfDelete').classList.toggle('hidden', !exists);
  $('#kfChips').innerHTML = ent.keyframes
    .map((k) => `<span class="chip" data-t="${k.t_ms}" title="x=${k.x} y=${k.y}">
        ${fmt(k.t_ms)}<b class="x" data-del="${k.t_ms}">×</b></span>`)
    .join('');
  updateKfReadout();
}

function updateKfReadout() {
  const ent = entityById(ui.selectedId);
  if (!ent) return;
  const sm = sampleEntity(ent, ui.time);
  const ang = sampleTurntable(state.turntable_keyframes, ui.time);
  const axis = ent.mounted === 'fixed' ? '全局' : '局部';
  if (sm) {
    const angleTxt = ent.mounted === 'fixed'
      ? '固定于地面（不旋转）'
      : `转台角度=${ang.toFixed(1)}°`;
    $('#kfSampled').textContent =
      `当前插值 @${fmt(ui.time)}：${axis} x=${sm.x.toFixed(2)}  y=${sm.y.toFixed(2)}  ${angleTxt}`;
  } else {
    $('#kfSampled').textContent = '当前插值：该实体还没有关键帧';
  }
  const exists = ent.keyframes.some((k) => k.t_ms === snap50(ui.time));
  $('#btnKfDelete').classList.toggle('hidden', !exists);
  // 自动填入该时刻已有关键帧（否则填插值）；用户正在输入时不覆盖
  const editing = ['kfX', 'kfY'].includes(document.activeElement.id);
  if (!editing) {
    const src = ent.keyframes.find((k) => k.t_ms === snap50(ui.time)) || sm;
    if (src) {
      $('#kfX').value = src.x.toFixed(2);
      $('#kfY').value = src.y.toFixed(2);
    }
  }
}

function renderRiskList() {
  const ul = $('#riskList');
  if (!state.risks.length) {
    ul.innerHTML = '<li class="muted">暂无风险 🎉</li>';
    return;
  }
  ul.innerHTML = state.risks.map((r) => `
    <li data-risk="${r.id}" class="risk-${r.type}">
      <span class="tag">${r.type === 'collision' ? '相碰' : '越界'}</span>
      <span class="r-list-name">${esc(riskNames(r))}</span>
      <span class="time-range">${fmt(r.start_ms)} – ${fmt(r.end_ms)}</span>
    </li>`).join('');
}

/* ================= 编队面板 ================= */

function renderFormationPanel() {
  const groups = state.formations || [];
  const { parent } = formationMaps();

  const gul = $('#formationList');
  if (!groups.length) {
    gul.innerHTML = '<li class="muted">暂无编队，点击「＋ 新建编队」创建</li>';
  } else {
    gul.innerHTML = groups.map((g) => `
      <li data-fid="${g.id}" class="${g.id === ui.selectedFormationId ? 'selected' : ''}">
        <span class="form-bar">
          <span class="swatch" style="background:${g.color}"></span>
          <span class="r-list-name">${esc(g.name)}</span>
          <span class="muted">${g.links.length} 条关系</span>
        </span>
      </li>`).join('');
  }

  const g = groups.find((x) => x.id === ui.selectedFormationId);
  $('#formationEditor').classList.toggle('hidden', !g);
  if (!g) return;

  $('#fmTitle').textContent = g.name;
  if (document.activeElement.id !== 'fmName') $('#fmName').value = g.name;
  if (document.activeElement.id !== 'fmColor') $('#fmColor').value = g.color;

  // 新增关系的主/从实体下拉；已被驱动的实体在「从属」侧标注当前主实体
  const opt = (e, note = '') =>
    `<option value="${e.id}">${esc(e.name)}（${mountedText(e.mounted)}）${note}</option>`;
  $('#lnkLeader').innerHTML = state.entities.map((e) => opt(e)).join('');
  $('#lnkFollower').innerHTML = state.entities.map((e) => {
    const par = parent.get(e.id);
    const note = par
      ? ` — 已跟随「${esc(entityById(par.leader_id)?.name || '?')}」`
      : '';
    return opt(e, note);
  }).join('');

  $('#linkList').innerHTML = g.links.length
    ? g.links.map((lk) => {
        const le = entityById(lk.leader_id);
        const fe = entityById(lk.follower_id);
        return `
        <li data-lid="${lk.id}">
          <span class="r-list-name">
            ${esc(le ? le.name : `#${lk.leader_id}`)}
            <span class="link-arrow">→</span>
            ${esc(fe ? fe.name : `#${lk.follower_id}`)}
          </span>
          <span class="link-nums">
            <input type="number" step="0.1" data-lfield="distance" value="${lk.distance}" title="相对距离(m)">m
            <input type="number" step="1" data-lfield="angle" value="${lk.angle}" title="相对角度(°)">°
          </span>
          <button class="link-del" data-link-del="${lk.id}" title="删除关系">✕</button>
        </li>`;
      }).join('')
    : '<li class="muted">该编队还没有跟随关系</li>';
}

/* ================= 舞台渲染 ================= */

function renderStage() {
  const svg = $('#stage');
  const R = state.settings.stage_radius;
  const vb = R * 1.18;
  svg.setAttribute('viewBox', `${-vb} ${-vb} ${2 * vb} ${2 * vb}`);

  const pos = formationPositionsAt(ui.time).pos;   // 含队形拓扑的全局位置
  const riskIds = new Set();
  for (const r of activeRisks()) r.entity_ids.forEach((id) => riskIds.add(id));

  let s = '';

  // 舞台边界与参考网格
  s += `<circle cx="0" cy="0" r="${R}" class="stage-edge"/>`;
  for (const f of [0.25, 0.5, 0.75]) {
    s += `<circle cx="0" cy="0" r="${R * f}" class="grid-line"/>`;
  }
  s += `<line x1="${-R}" y1="0" x2="${R}" y2="0" class="grid-line"/>`;
  s += `<line x1="0" y1="${-R}" x2="0" y2="${R}" class="grid-line"/>`;

  // 队形约束连线预览：主 -> 从，箭头 + 距离/角度标注
  const groups = state.formations || [];
  s += `<defs>${groups.map((g) => `
    <marker id="arrow-${g.id}" markerWidth="9" markerHeight="9" refX="7" refY="3"
            orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L7,3 L0,6 Z" fill="${g.color}"/>
    </marker>`).join('')}</defs>`;
  for (const g of groups) {
    for (const lk of g.links) {
      const p1 = pos.get(lk.leader_id);
      const p2 = pos.get(lk.follower_id);
      if (!p1 || !p2) continue;
      const fe = entityById(lk.follower_id);
      const le = entityById(lk.leader_id);
      // 起止点收进安全圆，避免箭头埋进圆里
      const ang = Math.atan2(p2.gy - p1.gy, p2.gx - p1.gx);
      const r1 = le ? le.radius : 0;
      const r2 = fe ? fe.radius : 0;
      const x1 = p1.gx + r1 * Math.cos(ang);
      const y1 = p1.gy + r1 * Math.sin(ang);
      const x2 = p2.gx - (r2 + 0.15) * Math.cos(ang);
      const y2 = p2.gy - (r2 + 0.15) * Math.sin(ang);
      const mx = (p1.gx + p2.gx) / 2;
      const my = (p1.gy + p2.gy) / 2;
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
              class="formation-link" stroke="${g.color}" marker-end="url(#arrow-${g.id})">
              <title>${esc(le?.name || '')} → ${esc(fe?.name || '')}
距离 ${lk.distance}m · 相对角 ${lk.angle}°</title></line>
            <text x="${mx}" y="${my - R * 0.02}" text-anchor="middle"
              class="formation-link-label" fill="${g.color}"
              >${lk.distance.toFixed(1)}m / ${lk.angle.toFixed(0)}°</text>`;
    }
  }

  // 全程运动轨迹（每 100ms 采样一个点，随台/固定语义与当前一致，含队形拓扑）
  for (const e of state.entities) {
    if (!e.keyframes.length) continue;
    const pts = [];
    for (let t = 0; t <= state.duration_ms; t += 100) {
      const p = formationPositionsAt(t).pos.get(e.id);
      if (p) pts.push(`${p.gx.toFixed(3)},${p.gy.toFixed(3)}`);
    }
    s += `<polyline points="${pts.join(' ')}" class="trail" style="stroke:${e.color}"/>`;
  }

  // 当前正在发生的相碰：红色虚线连接
  for (const r of activeRisks()) {
    if (r.type !== 'collision') continue;
    const [a, b] = r.entity_ids;
    if (pos.has(a) && pos.has(b)) {
      const pa = pos.get(a), pb = pos.get(b);
      s += `<line x1="${pa.gx}" y1="${pa.gy}" x2="${pb.gx}" y2="${pb.gy}" class="risk-link"/>`;
    }
  }

  // 实体：外圈=安全半径。随台实体画短线表示转台朝向（共用同一角度）；
  // 固定道具画方形锚点，表示钉在地面、不旋转。
  const fontSize = R * 0.09;
  const hitR = R * 0.07;
  for (const e of state.entities) {
    const p = pos.get(e.id);
    if (!p) continue;
    const sel = e.id === ui.selectedId;
    const risk = riskIds.has(e.id);
    const stroke = risk ? '#ff5252' : e.color;
    let marker = '';
    if (e.mounted === 'fixed') {
      const a = R * 0.022;
      marker = `<rect x="${p.gx - a}" y="${p.gy - a}" width="${2 * a}" height="${2 * a}"
                class="anchor" style="fill:${e.color};stroke:${stroke}"/>`;
    } else {
      const rad = (p.angle * Math.PI) / 180;
      const hx = p.gx + e.radius * Math.cos(rad);
      const hy = p.gy + e.radius * Math.sin(rad);
      marker += `<line x1="${p.gx}" y1="${p.gy}" x2="${hx}" y2="${hy}"
                  class="heading" style="stroke:${stroke}"/>`;
      marker += `<circle cx="${p.gx}" cy="${p.gy}" r="${R * 0.014}" style="fill:${e.color}"/>`;
    }
    s += `<g class="entity" data-id="${e.id}" style="cursor:grab">
      <circle cx="${p.gx}" cy="${p.gy}" r="${Math.max(e.radius, hitR)}" fill="rgba(0,0,0,0)"/>
      <circle cx="${p.gx}" cy="${p.gy}" r="${e.radius}" class="safety ${sel ? 'sel' : ''}"
              style="fill:${e.color};stroke:${stroke}"/>
      ${marker}
      <text x="${p.gx}" y="${p.gy + e.radius + fontSize * 1.15}" font-size="${fontSize}"
            text-anchor="middle" class="entity-label">${esc(e.name)}</text>
    </g>`;
  }

  svg.innerHTML = s;
}

/* ================= 时间轴渲染 ================= */

const TL = { padL: 12, padR: 12 };

function tlGeom() {
  const svg = $('#timeline');
  const W = svg.clientWidth || 900;
  const H = svg.clientHeight || 104;
  return {
    W, H,
    x: (t) => TL.padL + (t / state.duration_ms) * (W - TL.padL - TL.padR),
  };
}

function niceStep(duration, W) {
  const steps = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000, 120000];
  const pxPerMs = (W - TL.padL - TL.padR) / duration;
  for (const st of steps) if (st * pxPerMs >= 70) return st;
  return steps[steps.length - 1];
}

function renderTimeline() {
  const svg = $('#timeline');
  const { W, H, x } = tlGeom();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let s = '';

  // 刻度
  const step = niceStep(state.duration_ms, W);
  for (let t = 0; t <= state.duration_ms + 1; t += step) {
    const xx = x(t);
    s += `<line x1="${xx}" y1="${H - 18}" x2="${xx}" y2="${H - 12}" class="tick"/>
          <text x="${xx}" y="${H - 4}" class="tick-label" text-anchor="middle">
            ${(t / 1000).toFixed(step < 1000 ? 1 : 0)}s</text>`;
  }

  // 风险区间：红=相碰，橙=越界
  for (const r of state.risks) {
    const x0 = x(r.start_ms);
    const x1 = Math.max(x(r.end_ms), x0 + 2);
    s += `<rect x="${x0}" y="6" width="${x1 - x0}" height="16" rx="3"
              class="riskbar ${r.type}" data-risk="${r.id}">
            <title>${r.type === 'collision' ? '相碰' : '越界'}：${esc(riskNames(r))}
${fmt(r.start_ms)} – ${fmt(r.end_ms)}</title>
          </rect>`;
  }

  // 共享转台角度关键帧：蓝色小三角（作用于全部随台实体）
  for (const k of state.turntable_keyframes) {
    const cx = x(k.t_ms);
    s += `<path d="M ${cx} 30 l 5 8 l -10 0 Z" class="tkf" data-tf="${k.t_ms}">
            <title>转台角度 ${fmt(k.t_ms)} = ${k.angle}°（全部随台实体共用）</title>
          </path>`;
  }

  // 实体位置关键帧：选中实体为大菱形，其余实体为细线
  for (const e of state.entities) {
    const sel = e.id === ui.selectedId;
    for (const k of e.keyframes) {
      const cx = x(k.t_ms);
      if (sel) {
        s += `<path d="M ${cx} 52 l 5 7 l -5 7 l -5 -7 Z" class="kf-sel"
                style="fill:${e.color}"/>`;
      } else {
        s += `<line x1="${cx}" y1="55" x2="${cx}" y2="64" class="kf-mini"
                style="stroke:${e.color}"/>`;
      }
    }
  }

  // 播放头
  const px = x(ui.time);
  s += `<line x1="${px}" y1="0" x2="${px}" y2="${H - 14}" class="playhead"/>
        <path d="M ${px - 6} 0 L ${px + 6} 0 L ${px} 8 Z" class="playhead-handle"/>`;

  svg.innerHTML = s;
}

function renderDynamic() {
  $('#timeLabel').textContent = `${fmt(ui.time)} / ${fmt(state.duration_ms)}`;
  renderStage();
  renderTimeline();
  renderTurntablePanel();
  updateKfReadout();
  const active = new Set(activeRisks().map((r) => r.id));
  document.querySelectorAll('#riskList li[data-risk]').forEach((li) => {
    li.classList.toggle('active', active.has(+li.dataset.risk));
  });
}

/* ================= 事件绑定 ================= */

function bindEvents() {
  $('#btnPlay').addEventListener('click', () => (ui.playing ? pause() : play()));
  $('#btnNextRisk').addEventListener('click', () => jumpRisk(1));
  $('#btnPrevRisk').addEventListener('click', () => jumpRisk(-1));

  // 舞台半径
  $('#stageRadius').addEventListener('change', async (e) => {
    const v = parseFloat(e.target.value);
    if (v > 0) {
      state = await api('/settings', 'PUT', { stage_radius: v });
      afterState();
    }
  });

  // 新增实体
  $('#btnAddActor').addEventListener('click', () => addEntity('actor'));
  $('#btnAddProp').addEventListener('click', () => addEntity('prop'));

  // 实体列表选中
  $('#entityList').addEventListener('click', (ev) => {
    const li = ev.target.closest('li[data-id]');
    if (!li) return;
    ui.selectedId = +li.dataset.id;
    renderSidebar();
    renderDynamic();
  });

  // 实体编辑
  $('#edName').addEventListener('change', async (e) => {
    const name = e.target.value.trim();
    if (name) await updateEntity({ name });
  });
  $('#edKind').addEventListener('change', (e) => updateEntity({ kind: e.target.value }));
  $('#edMounted').addEventListener('change', (e) => updateEntity({ mounted: e.target.value }));
  $('#edRadius').addEventListener('change', async (e) => {
    const radius = parseFloat(e.target.value);
    if (radius >= 0.05) await updateEntity({ radius });
  });
  $('#edColor').addEventListener('change', (e) => updateEntity({ color: e.target.value }));
  $('#btnDelEntity').addEventListener('click', async () => {
    const ent = entityById(ui.selectedId);
    if (!ent) return;
    if (!confirm(`删除「${ent.name}」及其全部关键帧？`)) return;
    state = await api(`/entities/${ent.id}`, 'DELETE');
    ui.selectedId = null;
    afterState();
  });

  // 共享转台角度关键帧
  $('#btnTfSave').addEventListener('click', async () => {
    let angle = parseFloat($('#tfAngle').value);
    if (!isFinite(angle)) angle = sampleTurntable(state.turntable_keyframes, ui.time);
    state = await api('/turntable/keyframes', 'PUT', {
      t_ms: snap50(ui.time), angle: round2(angle),
    });
    afterState();
  });
  $('#btnTfDelete').addEventListener('click', async () => {
    state = await api(`/turntable/keyframes/${snap50(ui.time)}`, 'DELETE');
    afterState();
  });
  $('#tfChips').addEventListener('click', async (ev) => {
    const del = ev.target.closest('[data-tf-del]');
    if (del) {
      ev.stopPropagation();
      state = await api(`/turntable/keyframes/${del.dataset.tfDel}`, 'DELETE');
      afterState();
      return;
    }
    const chip = ev.target.closest('.tf-chip');
    if (chip) seek(+chip.dataset.t);
  });

  // 实体位置关键帧面板
  $('#btnKfRead').addEventListener('click', () => {
    const sm = sampleEntity(entityById(ui.selectedId), ui.time);
    if (!sm) return;
    $('#kfX').value = sm.x.toFixed(2);
    $('#kfY').value = sm.y.toFixed(2);
  });
  $('#btnKfSave').addEventListener('click', saveKeyframeFromForm);
  $('#btnKfDelete').addEventListener('click', async () => {
    const ent = entityById(ui.selectedId);
    if (!ent) return;
    state = await api(`/entities/${ent.id}/keyframes/${snap50(ui.time)}`, 'DELETE');
    afterState();
  });
  $('#kfChips').addEventListener('click', async (ev) => {
    const del = ev.target.closest('[data-del]');
    if (del) {
      ev.stopPropagation();
      state = await api(`/entities/${ui.selectedId}/keyframes/${del.dataset.del}`, 'DELETE');
      afterState();
      return;
    }
    const chip = ev.target.closest('.chip');
    if (chip) seek(+chip.dataset.t);
  });

  // 风险列表跳转
  $('#riskList').addEventListener('click', (ev) => {
    const li = ev.target.closest('li[data-risk]');
    if (!li) return;
    const r = state.risks.find((x) => x.id === +li.dataset.risk);
    if (r) seek(r.start_ms);
  });

  bindFormationEvents();

  // 舞台：拖动实体 -> 在当前时刻写位置关键帧
  $('#stage').addEventListener('pointerdown', (ev) => {
    const g = ev.target.closest('g.entity');
    if (!g) return;
    pause();
    ui.dragId = +g.dataset.id;
    ui.dragKf = null;
    ui.selectedId = ui.dragId;
    renderSidebar();
    ev.preventDefault();
  });

  window.addEventListener('pointermove', (ev) => {
    if (ui.dragId == null) return;
    const svg = $('#stage');
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());

    const ent = entityById(ui.dragId);
    let lx, ly;
    if (ent.mounted === 'fixed') {
      // 固定道具：指针的全局坐标就是关键帧坐标，角度保持 0
      lx = p.x; ly = p.y;
    } else {
      // 随台实体：用当前共享转台角度把全局位置反算成局部坐标
      const angle = sampleTurntable(state.turntable_keyframes, ui.time);
      [lx, ly] = toLocal(p.x, p.y, angle);
    }
    ui.dragKf = { t_ms: snap50(ui.time), x: round2(lx), y: round2(ly) };
    upsertLocal(ent, ui.dragKf);

    // 队形乐观预览：若拖动的是主实体，按随台/固定语义把从属实体一并摆好
    // （后端提交时会重新传播并做越界/成环校验，被拒即整体回滚）
    const { children } = formationMaps();
    const desc = descendantIds(ui.dragId, children);
    if (desc.length) {
      const { pos, angle: ang } = formationPositionsAt(
        snap50(ui.time), { [ui.dragId]: { gx: p.x, gy: p.y } });
      for (const fid of desc) {
        const fp = pos.get(fid);
        const fe = entityById(fid);
        if (!fp || !fe) continue;
        const [fx, fy] = fe.mounted === 'fixed'
          ? [fp.gx, fp.gy]
          : toLocal(fp.gx, fp.gy, ang);
        upsertLocal(fe, { t_ms: snap50(ui.time), x: round2(fx), y: round2(fy) });
      }
    }
    renderDynamic();
  });

  window.addEventListener('pointerup', async () => {
    if (ui.dragId == null) return;
    const { dragId, dragKf } = ui;
    ui.dragId = null;
    ui.dragKf = null;
    if (dragKf) {
      // 原子提交：后端传播从属实体并校验越界/成环；整单失败则回滚到服务器状态
      try {
        state = await api(`/entities/${dragId}/keyframes`, 'PUT', dragKf);
        afterState();
      } catch (_) {
        await refresh();
      }
    }
  });

  // 时间轴：点击 / 拖动定位；点击风险条/转台帧跳转
  $('#timeline').addEventListener('pointerdown', (ev) => {
    const bar = ev.target.closest('[data-risk]');
    if (bar) {
      const r = state.risks.find((x) => x.id === +bar.dataset.risk);
      if (r) seek(r.start_ms);
      return;
    }
    const tf = ev.target.closest('[data-tf]');
    if (tf) { seek(+tf.dataset.tf); return; }
    ui.scrubbing = true;
    pause();
    timelineSeek(ev);
  });
  window.addEventListener('pointermove', (ev) => {
    if (ui.scrubbing) timelineSeek(ev);
  });
  window.addEventListener('pointerup', () => { ui.scrubbing = false; });

  // 快捷键
  window.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, select, textarea')) return;
    if (ev.code === 'Space') {
      ev.preventDefault();
      ui.playing ? pause() : play();
    } else if (ev.code === 'ArrowRight') {
      seek(ui.time + (ev.shiftKey ? 1000 : SAMPLE_MS));
    } else if (ev.code === 'ArrowLeft') {
      seek(ui.time - (ev.shiftKey ? 1000 : SAMPLE_MS));
    }
  });

  window.addEventListener('resize', () => { if (state) renderTimeline(); });
}

function timelineSeek(ev) {
  const svg = $('#timeline');
  const rect = svg.getBoundingClientRect();
  const { W } = tlGeom();
  const frac = (ev.clientX - rect.left - TL.padL) / (W - TL.padL - TL.padR);
  seek(frac * state.duration_ms);
}

function bindFormationEvents() {
  // 新建编队
  $('#btnAddFormation').addEventListener('click', async () => {
    const n = (state.formations || []).length + 1;
    state = await api('/formations', 'POST', { name: `编队 ${n}` });
    ui.selectedFormationId = Math.max(...state.formations.map((g) => g.id));
    afterState();
  });

  // 选中编队
  $('#formationList').addEventListener('click', (ev) => {
    const li = ev.target.closest('li[data-fid]');
    if (!li) return;
    ui.selectedFormationId = +li.dataset.fid;
    renderSidebar();
  });

  // 编队改名 / 换色
  $('#btnFmSave').addEventListener('click', async () => {
    const g = (state.formations || []).find((x) => x.id === ui.selectedFormationId);
    if (!g) return;
    const name = $('#fmName').value.trim() || g.name;
    const color = $('#fmColor').value;
    state = await api(`/formations/${g.id}`, 'PUT', { name, color });
    afterState();
  });

  // 删除编队（级联删除其关系；不影响实体关键帧）
  $('#btnFmDel').addEventListener('click', async () => {
    const g = (state.formations || []).find((x) => x.id === ui.selectedFormationId);
    if (!g) return;
    if (!confirm(`删除编队「${g.name}」及其 ${g.links.length} 条跟随关系？`)) return;
    state = await api(`/formations/${g.id}`, 'DELETE');
    ui.selectedFormationId = null;
    afterState();
  });

  // 新增跟随关系（原子校验：成环/多主由后端整单拒绝）
  $('#btnLinkAdd').addEventListener('click', async () => {
    const gid = ui.selectedFormationId;
    if (gid == null) return;
    const leader_id = +$('#lnkLeader').value;
    const follower_id = +$('#lnkFollower').value;
    const distance = parseFloat($('#lnkDistance').value);
    const angle = parseFloat($('#lnkAngle').value);
    if (!isFinite(distance) || distance <= 0) { alert('相对距离需为正数'); return; }
    if (!isFinite(angle)) { alert('相对角度需为数字'); return; }
    state = await api(`/formations/${gid}/links`, 'POST',
      { leader_id, follower_id, distance: round2(distance), angle: round2(angle) });
    afterState();
  });

  // 关系列表：就地改距离/角度、删除
  $('#linkList').addEventListener('change', async (ev) => {
    const input = ev.target.closest('input[data-lfield]');
    if (!input) return;
    const li = ev.target.closest('li[data-lid]');
    const lid = +li.dataset.lid;
    const field = input.dataset.lfield;
    const v = parseFloat(input.value);
    if (!isFinite(v)) return;
    try {
      state = await api(`/formations/links/${lid}`, 'PUT', { [field]: round2(v) });
      afterState();
    } catch (_) {
      renderFormationPanel();    // 被拒则还原输入框
    }
  });
  $('#linkList').addEventListener('click', async (ev) => {
    const del = ev.target.closest('[data-link-del]');
    if (!del) return;
    state = await api(`/formations/links/${del.dataset.linkDel}`, 'DELETE');
    afterState();
  });
}

async function addEntity(kind) {  const n = state.entities.filter((e) => e.kind === kind).length + 1;
  const name = kind === 'actor' ? `演员${n}` : `道具${n}`;
  state = await api('/entities', 'POST', {
    name,
    kind,
    radius: kind === 'actor' ? 0.5 : 0.8,
  });
  ui.selectedId = Math.max(...state.entities.map((e) => e.id));
  afterState();
}

async function updateEntity(patch) {
  if (ui.selectedId == null) return;
  state = await api(`/entities/${ui.selectedId}`, 'PUT', patch);
  afterState();
}

async function saveKeyframeFromForm() {
  const ent = entityById(ui.selectedId);
  if (!ent) return;
  const sm = sampleEntity(ent, ui.time);
  let x = parseFloat($('#kfX').value);
  let y = parseFloat($('#kfY').value);
  if (!isFinite(x)) x = sm ? sm.x : 0;
  if (!isFinite(y)) y = sm ? sm.y : 0;
  const kf = { t_ms: snap50(ui.time), x: round2(x), y: round2(y) };
  state = await api(`/entities/${ent.id}/keyframes`, 'PUT', kf);
  afterState();
}

/* ================= 启动 ================= */

bindEvents();
refresh().catch((err) => {
  document.body.insertAdjacentHTML(
    'afterbegin',
    `<div style="padding:20px;color:#ff8a8a">加载状态失败：${esc(err.message)}</div>`);
});
