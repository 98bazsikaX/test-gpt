/* microgpt — főoldal (név-kiegészítés + újratanítás + élő vizualizációk) */

/* A plotok a "napfényes papír" témát követik */
const chartTheme = {
  paper_bgcolor: '#f6f0e4',
  plot_bgcolor: '#fffdf8',
  font: { family: 'system-ui, sans-serif', color: '#1c2433', size: 13 },
  xaxis: { gridcolor: '#e8dfd0', zerolinecolor: '#e8dfd0' },
  yaxis: { gridcolor: '#e8dfd0', zerolinecolor: '#e8dfd0' },
};

const lossLayout = {
  title: 'Élő loss',
  xaxis: { title: 'lépés' },
  yaxis: { title: 'loss' },
  height: 320,
  showlegend: true,
  legend: { orientation: 'h', y: -0.15 },
  ...chartTheme,
};
const pcaLayout = {
  title: 'Karakter-embeddingek PCA-ja (élő)',
  xaxis: { title: '1. főkomponens' },
  yaxis: { title: '2. főkomponens' },
  height: 320,
  ...chartTheme,
};

async function api(url, method, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  return res.json();
}

async function doComplete() {
  const out = document.getElementById('out'); out.innerHTML = '';
  const err = document.getElementById('err'); err.textContent = '';
  const seed = document.getElementById('seed').value.trim();
  if (!seed) return;
  const d = await api('/complete', 'POST', { seed });
  if (!d.results || !d.results.length) { err.textContent = 'Nincs érvényes betű (a-z).'; return; }
  for (const n of d.results) { const li = document.createElement('li'); li.textContent = n; out.appendChild(li); }
}

async function setTemp() {
  const t = parseFloat(document.getElementById('temp').value);
  document.getElementById('tempval').textContent = t.toFixed(2);
  await api('/temperature', 'POST', { value: t });
}

async function startTrain() {
  const cfg = {
    n_layer: +document.getElementById('f_layers').value,
    n_embd: +document.getElementById('f_embd').value,
    n_head: +document.getElementById('f_heads').value,
    block_size: +document.getElementById('f_block').value,
    steps: +document.getElementById('f_steps').value,
  };
  const d = await api('/retrain', 'POST', cfg);
  if (d.error) { setStatus('hiba → ' + d.error); return; }
  pollLive();
}

/* A konzol-szignatúra sorának formázása:  step 0042/1000 | loss 2.4181 */
function statusText(d) {
  if (d.error) return 'hiba → ' + d.error;
  if (d.running) {
    const last = d.losses && d.losses.length ? d.losses[d.losses.length - 1].toFixed(4) : '—';
    return 'step ' + String(d.step).padStart(4, '0') + '/' + d.total + ' | loss ' + last;
  }
  if (d.losses && d.losses.length) {
    return 'kész · ' + d.total + ' lépés · utolsó loss ' + d.final_loss.toFixed(4);
  }
  return 'várakozás — indíts tanítást lentebb…';
}

function setStatus(text) {
  document.getElementById('trainstatus').textContent = text;
}

/* Exponenciális mozgóátlag (EMA) — "aluláteresztő szűrő" a zajos loss-görbéhez.
   Az alpha szabályozza a simaságot: kicsi = simább (lassabb reakció). */
function ema(arr, alpha) {
  const out = new Array(arr.length);
  if (!arr.length) return out;
  let e = arr[0];
  out[0] = e;
  for (let i = 1; i < arr.length; i++) {
    e = alpha * arr[i] + (1 - alpha) * e;
    out[i] = e;
  }
  return out;
}

const SMOOTH_MIN_POINTS = 60;  // ennyi lépés után jelenik meg a simított vonal

let polling = false;
async function pollLive() {
  if (polling) return;
  polling = true;
  try {
    const d = await api('/plot/live', 'GET');
    setStatus(statusText(d));
    if (d.losses && d.losses.length) {
      const x = Array.from({ length: d.losses.length }, (_, i) => i + 1);
      const traces = [{
        x, y: d.losses, mode: 'lines', name: 'loss',
        line: { color: '#8a94a6', width: 1 },
      }];
      if (d.losses.length >= SMOOTH_MIN_POINTS) {
        traces.push({
          x, y: ema(d.losses, 0.03), mode: 'lines', name: 'mozgóátlag',
          line: { color: '#2f6f68', width: 2.5 },
        });
      }
      Plotly.react('live_loss', traces, lossLayout);
    }
    if (d.pca && d.pca.length) {
      Plotly.react('live_pca', [{
        x: d.pca.map(p => p[0]), y: d.pca.map(p => p[1]),
        text: d.pca_labels, mode: 'markers+text', textposition: 'top center',
        marker: { color: '#c07a12', size: 9 },
      }], pcaLayout);
    }
    if (d.running) setTimeout(pollLive, 500);
  } finally { polling = false; }
}

async function loadConfig() {
  const d = await api('/config', 'GET');
  document.getElementById('f_layers').value = d.n_layer;
  document.getElementById('f_embd').value = d.n_embd;
  document.getElementById('f_heads').value = d.n_head;
  document.getElementById('f_block').value = d.block_size;
  document.getElementById('f_steps').value = d.steps;
  document.getElementById('temp').value = d.temperature;
  document.getElementById('tempval').textContent = d.temperature.toFixed(2);
  if (d.training) {
    pollLive();
  } else {
    setStatus('[' + d.backend + '] várakozás — indíts tanítást lentebb…');
  }
}

document.getElementById('seed').addEventListener('keydown', e => { if (e.key === 'Enter') doComplete(); });
loadConfig();
pollLive();
