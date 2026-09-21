/* jev-doom-fly replay dashboard (v1, no build step).
   Reconstructs a recorded experiment from /api/recordings/<id> + frames.u8.
   Pure replay: no live simulation, no Jev API calls. */

const state = {
  rec: null,        // parsed recording JSON
  frames: null,     // Uint8Array of all frames
  frameMeta: null,  // {count, shape}
  step: 0,
  playing: false,
  speed: 1,
  lastTick: null,
};

const $ = (id) => document.getElementById(id);

async function init() {
  const runs = await fetchJSON('/api/recordings');
  const sel = $('run-select');
  if (!runs.length) {
    sel.innerHTML = '<option>(no recordings — run: make run-demo)</option>';
    return;
  }
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    opt.textContent = `${r.run_id}  [${r.environment_backend} / ${r.connectome_source} / ${r.jev?.client ?? '?'}]`;
    sel.appendChild(opt);
  }
  sel.onchange = () => load(sel.value);
  await load(runs[0].run_id);
  requestAnimationFrame(tick);
}

async function load(runId) {
  state.rec = await fetchJSON(`/api/recordings/${runId}`);
  const fm = state.rec.frames;
  if (fm && fm.url) {
    const buf = await (await fetch(fm.url)).arrayBuffer();
    state.frames = new Uint8Array(buf);
    state.frameMeta = fm;
  } else {
    state.frames = null; state.frameMeta = null;
  }
  state.step = 0; state.playing = false;
  $('play-btn').innerHTML = '&#9654; Play';
  $('scrubber').max = state.rec.steps.length - 1;
  $('scrubber').value = 0;

  const h = state.rec.header;
  $('run-meta').textContent =
    `env=${h.environment_backend} · connectome=${h.connectome.provenance.source}` +
    `${h.connectome.provenance.reduced ? ' (REDUCED)' : ''} · jev=${h.jev.client}`;
  $('warnings').innerHTML = (h.warnings || [])
    .map(w => `<span class="warn">⚠ ${w}</span>`).join('');
  $('brain-mode').textContent =
    `(${h.connectome.provenance.source}${h.connectome.provenance.reduced ? ', reduced subset' : ''})`;
  buildBars();
  render();
}

function buildBars() {
  const s0 = state.rec.steps[0];
  $('jev-bars').innerHTML = '';
  for (const q of Object.keys(s0.jev ? s0.jev.probabilities : {ATTACK:0,RETREAT:0,EXPLORE:0,THREAT_LEVEL:0}))
    $('jev-bars').appendChild(barRow(q, ''));
  $('pop-bars').innerHTML = '';
  for (const p of Object.keys(s0.populations))
    $('pop-bars').appendChild(barRow(p, 'pop'));
  $('action-bars').innerHTML = '';
  for (const a of Object.keys(s0.motor.scores))
    $('action-bars').appendChild(barRow(a, 'motor'));
}

function barRow(label, cls) {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<span>${label}</span>
    <div class="bar-track"><div class="bar-fill ${cls}" data-key="${label}"></div></div>
    <span class="bar-val" data-val="${label}">–</span>`;
  return row;
}

function setBar(containerId, key, frac, text) {
  const fill = document.querySelector(`#${containerId} [data-key="${key}"]`);
  const val = document.querySelector(`#${containerId} [data-val="${key}"]`);
  if (fill) fill.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  if (val) val.textContent = text;
}

function render() {
  const s = state.rec.steps[state.step];
  if (!s) return;
  // DOOM viewport
  if (state.frames && state.frameMeta) {
    const [h, w] = state.frameMeta.shape;
    const off = state.step * h * w;
    const img = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = state.frames[off + i];
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    const c = $('viewport');
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(img, 0, 0);
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, c.width, c.height);
  }
  $('doom-state').innerHTML =
    `health <b>${s.state.health.toFixed(0)}</b> · ammo <b>${s.state.ammo}</b> · ` +
    `kills <b>${s.state.kills}</b> · tic <b>${s.episode_tic}</b> · ` +
    `threat <b>${s.state.threat_level.toFixed(2)}</b> · ` +
    `enemy ${s.state.enemy_visible ? `dist ${s.state.enemy_distance.toFixed(2)}, ang ${s.state.enemy_angle.toFixed(2)}` : 'not visible'}`;

  // Jev
  if (s.jev) {
    for (const [q, p] of Object.entries(s.jev.probabilities))
      setBar('jev-bars', q, p, p.toFixed(2));
    $('jev-model').textContent = s.jev.is_mock ? '(deterministic mock)' : `(${s.jev.model})`;
    $('jev-meta').innerHTML =
      `request <b>${s.jev.request_id}</b> · latency <b>${s.jev.latency_ms.toFixed(1)} ms</b>`;
  } else {
    $('jev-meta').innerHTML = '<b>no decision yet</b> (previous-decision-stays-active)';
  }

  // populations
  let maxRate = 1;
  for (const p of Object.values(s.populations)) maxRate = Math.max(maxRate, p.mean_rate_hz);
  for (const [name, p] of Object.entries(s.populations))
    setBar('pop-bars', name, p.mean_rate_hz / maxRate, `${p.mean_rate_hz.toFixed(1)} Hz`);
  drawHeatmap(s);

  // motor
  for (const [a, sc] of Object.entries(s.motor.scores)) {
    setBar('action-bars', a, sc, sc.toFixed(2));
    const fill = document.querySelector(`#action-bars [data-key="${a}"]`);
    if (fill) fill.classList.toggle('selected', a === s.motor.selected);
  }
  $('motor-meta').innerHTML =
    `selected <b>${s.motor.selected}</b> · confidence <b>${s.motor.confidence.toFixed(2)}</b>` +
    ` · controller ${s.controller_latency_ms.toFixed(1)} ms`;

  $('clock').textContent = `${(s.t_ms / 1000).toFixed(2)}s`;
  $('scrubber').value = state.step;
  drawTimeline();
}

function drawHeatmap(s) {
  const c = $('heatmap'), ctx = c.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
  const pops = Object.entries(s.populations).filter(([, p]) => p.sampled?.length);
  const rowH = c.height / Math.max(1, pops.length);
  pops.forEach(([name, p], row) => {
    const cellW = c.width / p.sampled.length;
    let maxR = 1;
    for (const n of p.sampled) maxR = Math.max(maxR, n.rate_hz);
    p.sampled.forEach((n, i) => {
      const v = Math.min(1, n.rate_hz / maxR);
      ctx.fillStyle = `rgb(${Math.round(v * 255)}, ${Math.round(v * 120)}, 60)`;
      ctx.fillRect(i * cellW + 1, row * rowH + 1, cellW - 2, rowH - 2);
    });
    ctx.fillStyle = '#8a94a3'; ctx.font = '9px sans-serif';
    ctx.fillText(name, 3, row * rowH + 10);
  });
}

function drawTimeline() {
  const c = $('timeline'), ctx = c.getContext('2d');
  const steps = state.rec.steps;
  const t0 = steps[0].t_ms, t1 = steps[steps.length - 1].t_ms;
  const span = Math.max(1, t1 - t0);
  ctx.fillStyle = '#191d24'; ctx.fillRect(0, 0, c.width, c.height);
  const x = (t) => 8 + (t - t0) / span * (c.width - 16);
  // threat curve
  ctx.strokeStyle = '#ffb454'; ctx.beginPath();
  steps.forEach((s, i) => {
    const y = c.height - 8 - s.state.threat_level * (c.height - 30);
    i ? ctx.lineTo(x(s.t_ms), y) : ctx.moveTo(x(s.t_ms), y);
  });
  ctx.stroke();
  // action ticks
  ctx.fillStyle = '#8dff6e';
  for (const s of steps)
    if (s.motor.selected !== 'noop') ctx.fillRect(x(s.t_ms), c.height - 12, 2, 6);
  // jev decision markers
  ctx.fillStyle = '#6cc3ff';
  let lastReq = null;
  for (const s of steps)
    if (s.jev && s.jev.request_id !== lastReq) {
      ctx.fillRect(x(s.t_ms), 4, 2, 10);
      lastReq = s.jev.request_id;
    }
  // playhead
  ctx.fillStyle = '#fff';
  ctx.fillRect(x(steps[state.step].t_ms) - 1, 0, 2, c.height);
  ctx.fillStyle = '#8a94a3'; ctx.font = '10px sans-serif';
  ctx.fillText('— threat   | jev decisions   | actions', 10, c.height - 2);
}

// transport
$('play-btn').onclick = () => {
  state.playing = !state.playing;
  $('play-btn').innerHTML = state.playing ? '&#10073;&#10073; Pause' : '&#9654; Play';
  state.lastTick = null;
};
$('speed').onchange = (e) => { state.speed = parseFloat(e.target.value); state.lastTick = null; };
$('scrubber').oninput = (e) => { state.step = parseInt(e.target.value); render(); };
$('timeline').onclick = (e) => {
  const r = e.target.getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  state.step = Math.round(frac * (state.rec.steps.length - 1));
  render();
};

function tick(ts) {
  if (state.playing && state.rec) {
    if (state.lastTick == null) state.lastTick = ts;
    const steps = state.rec.steps;
    const curT = steps[state.step].t_ms;
    const elapsed = (ts - state.lastTick) * state.speed;
    let next = state.step;
    while (next + 1 < steps.length && steps[next + 1].t_ms <= curT + elapsed) next++;
    if (next !== state.step) { state.step = next; state.lastTick = ts; render(); }
    if (state.step >= steps.length - 1) {
      state.playing = false;
      $('play-btn').innerHTML = '&#9654; Play';
    }
  }
  requestAnimationFrame(tick);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

init().catch(e => { document.body.innerHTML = `<pre>${e.stack}</pre>`; });
