/* microgpt — főoldal (név-kiegészítés + újratanítás + élő vizualizációk) */
const lossLayout = { title: 'Élő loss', xaxis: { title: 'lépés' }, yaxis: { title: 'loss' }, template: 'plotly_white', height: 320 };
const pcaLayout = { title: 'Karakter-embeddingek PCA-ja (élő)', xaxis: { title: '1. főkomponens' }, yaxis: { title: '2. főkomponens' }, template: 'plotly_white', height: 320 };

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
  if (d.error) { document.getElementById('trainstatus').textContent = 'Hiba: ' + d.error; return; }
  document.getElementById('trainstatus').textContent = 'Tanítás indul...';
  pollLive();
}

let polling = false;
async function pollLive() {
  if (polling) return;
  polling = true;
  try {
    const d = await api('/plot/live', 'GET');
    const st = document.getElementById('trainstatus');
    if (d.losses && d.losses.length) {
      const x = Array.from({ length: d.losses.length }, (_, i) => i + 1);
      Plotly.react('live_loss', [{ x, y: d.losses, mode: 'lines', name: 'loss' }], lossLayout);
    }
    if (d.pca && d.pca.length) {
      Plotly.react('live_pca', [{
        x: d.pca.map(p => p[0]), y: d.pca.map(p => p[1]),
        text: d.pca_labels, mode: 'markers+text', textposition: 'top center',
      }], pcaLayout);
    }
    if (d.running) {
      st.textContent = 'Tanulás... ' + d.step + '/' + d.total;
      setTimeout(pollLive, 500);
    } else if (d.error) {
      st.textContent = 'Hiba: ' + d.error;
    } else if (d.losses && d.losses.length) {
      st.textContent = 'Kész! ' + d.total + ' lépés, utolsó loss: ' + d.final_loss.toFixed(4);
    }
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
  if (d.training) pollLive();
}

document.getElementById('seed').addEventListener('keydown', e => { if (e.key === 'Enter') doComplete(); });
loadConfig();
pollLive();
