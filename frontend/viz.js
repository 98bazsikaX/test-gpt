/* microgpt — vizualizáció-galería (/viz) */
async function loadPlot(url, id, tries) {
  tries = tries || 0;
  const res = await fetch(url);
  if (res.status === 409 && tries < 30) {   // tréning fut -> várunk és újrapróbáljuk
    setTimeout(() => loadPlot(url, id, tries + 1), 2000);
    return;
  }
  const d = await res.json();
  const el = document.getElementById(id);
  if (d.error) { el.innerHTML = '<p style="color:#b00">' + d.error + '</p>'; return; }
  Plotly.newPlot(el, d.data, d.layout, { responsive: true });
}

function att() {
  loadPlot('/plot/attention?name=' + encodeURIComponent(document.getElementById('att_name').value), 'patt');
}
function dist() {
  loadPlot('/plot/distribution?prefix=' + encodeURIComponent(document.getElementById('dist_prefix').value), 'pdist');
}

loadPlot('/plot/nlayer', 'pnlayer');
loadPlot('/plot/pca_pos', 'ppca_pos');
att();
dist();
