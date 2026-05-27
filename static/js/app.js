(function(){document.documentElement.setAttribute('data-bs-theme', localStorage.getItem('drakeai_theme') || 'dark');})();async function downloadFile(url, filename) {
  const res = await fetch(url, { credentials: 'same-origin' });
  if (!res.ok) throw new Error('Download failed');
  const blob = await res.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

async function loadPredictionSection() {
  const wrap = document.getElementById('predictionContent');
  if (!wrap) return;
  wrap.innerHTML = '<div class="text-secondary">Loading prediction results...</div>';
  try {
    const [porRes, satRes, litRes, currentRes] = await Promise.all([
      fetch('/predict-porosity', { method: 'POST' }),
      fetch('/predict-saturation', { method: 'POST' }),
      fetch('/predict-lithology', { method: 'POST' }),
      fetch('/current-analysis')
    ]);
    const por = await porRes.json();
    const sat = await satRes.json();
    const lit = await litRes.json();
    const current = await currentRes.json();
    if (!por.success || !sat.success || !lit.success) throw new Error('Prediction data unavailable');
    const logOptions = (current.available_logs || []).map(l => `<option value="${l.mnemonic}">${l.mnemonic} ${l.unit ? `(${l.unit})` : ''}</option>`).join('');
    wrap.innerHTML = `
      <div class="row g-4 mb-4">
        <div class="col-lg-4"><div class="glass-panel p-3 h-100"><div class="d-flex justify-content-between align-items-center mb-3"><h5 class="mb-0">Porosity</h5><button id="downloadPorosity" class="btn btn-outline-info btn-sm">Download CSV</button></div><div class="table-responsive"><table class="table table-dark table-striped mb-0"><thead><tr><th>#</th><th>Depth</th><th>Porosity</th><th>Confidence</th></tr></thead><tbody>${por.data.slice(0,5).map((r,i)=>`<tr><td>${i+1}</td><td>${r.DEPTH}</td><td>${r.POROSITY}</td><td>${r.CONFIDENCE}</td></tr>`).join('')}</tbody></table></div></div></div>
        <div class="col-lg-4"><div class="glass-panel p-3 h-100"><div class="d-flex justify-content-between align-items-center mb-3"><h5 class="mb-0">Water Saturation</h5><button id="downloadSaturation" class="btn btn-outline-info btn-sm">Download CSV</button></div><div class="table-responsive"><table class="table table-dark table-striped mb-0"><thead><tr><th>#</th><th>Depth</th><th>P10</th><th>P50</th><th>P90</th></tr></thead><tbody>${sat.data.slice(0,5).map((r,i)=>`<tr><td>${i+1}</td><td>${r.DEPTH}</td><td>${r.P10}</td><td>${r.P50}</td><td>${r.P90}</td></tr>`).join('')}</tbody></table></div></div></div>
        <div class="col-lg-4"><div class="glass-panel p-3 h-100"><div class="d-flex justify-content-between align-items-center mb-3"><h5 class="mb-0">Lithology</h5><button id="downloadLithology" class="btn btn-outline-info btn-sm">Download CSV</button></div><div class="table-responsive"><table class="table table-dark table-striped mb-0"><thead><tr><th>#</th><th>Depth</th><th>Lithology</th><th>Confidence</th></tr></thead><tbody>${lit.data.slice(0,5).map((r,i)=>`<tr><td>${i+1}</td><td>${r.DEPTH}</td><td>${r.LITHOLOGY}</td><td>${r.CONFIDENCE}</td></tr>`).join('')}</tbody></table></div></div></div>
      </div>
      <div class="row g-4">
        <div class="col-lg-4"><div class="glass-panel p-3 h-100"><h5 class="mb-3">Select graph log</h5><select id="predictionLogSelector" class="form-select bg-dark text-light border-secondary">${logOptions}</select><div id="predictionLogProperties" class="mt-3"></div></div></div>
        <div class="col-lg-8"><div class="glass-panel p-3"><h5 class="mb-3">Selected log graph</h5><div id="predictionSelectedLogChart" style="height:430px;"></div></div></div>
      </div>`;

    document.getElementById('downloadPorosity')?.addEventListener('click', ()=>downloadFile('/export-predictions/porosity', 'drakeai_porosity_predictions.csv'));
    document.getElementById('downloadSaturation')?.addEventListener('click', ()=>downloadFile('/export-predictions/saturation', 'drakeai_water_saturation_predictions.csv'));
    document.getElementById('downloadLithology')?.addEventListener('click', ()=>downloadFile('/export-predictions/lithology', 'drakeai_lithology_predictions.csv'));

    async function renderSelectedPredictionLog() {
      const selector = document.getElementById('predictionLogSelector');
      const selected = selector?.value;
      const res = await fetch(`/logs?selected=${encodeURIComponent(selected)}`);
      const data = await res.json();
      if (!data.success || !data.records?.length) return;
      const stats = data.stats?.[selected] || {};
      document.getElementById('predictionLogProperties').innerHTML = `<div class="border border-secondary rounded p-3"><div class="fw-semibold mb-2">${selected}</div><div class="small text-secondary">Min: ${stats.min ?? '-'}<br>Max: ${stats.max ?? '-'}<br>Mean: ${stats.mean ?? '-'}<br>P10: ${stats.p10 ?? '-'}<br>P90: ${stats.p90 ?? '-'}</div></div>`;
      Plotly.newPlot('predictionSelectedLogChart', [{x:data.records.map(r=>r[selected]), y:data.records.map(r=>r.DEPTH), mode:'lines', name:selected, line:{width:2,color:'#3dd5f3'}}], {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#d7dde8'}, yaxis:{autorange:'reversed', title:'Depth'}, xaxis:{title:selected}, margin:{t:30,l:50,r:20,b:40}}, {responsive:true, displaylogo:false});
    }
    document.getElementById('predictionLogSelector')?.addEventListener('change', renderSelectedPredictionLog);
    await renderSelectedPredictionLog();
  } catch (err) {
    wrap.innerHTML = `<div class="alert alert-danger mb-0">${err.message}</div>`;
  }
}

async function loadUncertaintySection() {
  const wrap = document.getElementById('uncertaintyContent');
  if (!wrap) return;
  wrap.innerHTML = '<div class="text-secondary">Loading uncertainty results...</div>';
  try {
    const satRes = await fetch('/predict-saturation', { method: 'POST' });
    const sat = await satRes.json();
    if (!sat.success || !sat.data?.length) throw new Error('Uncertainty data unavailable');
    const firstRows = sat.data.slice(0,5);
    wrap.innerHTML = `
      <div class="glass-panel p-3 mb-4"><h5 class="mb-2">Uncertainty</h5><p class="text-secondary mb-0">Exact uncertainty visualization for the active LAS file using prediction intervals, reliability percentage, and risk indicators.</p></div>
      <div class="glass-panel p-3 mb-4"><h5 class="mb-3">Water Saturation Uncertainty Curves</h5><div id="uncertaintyChart" style="height:460px;"></div></div>
      <div class="glass-panel p-3"><h5 class="mb-3">First five uncertainty results</h5><div class="table-responsive"><table class="table table-dark table-striped mb-0"><thead><tr><th>#</th><th>Depth</th><th>P10</th><th>P50</th><th>P90</th><th>Reliability %</th><th>Risk</th></tr></thead><tbody>${firstRows.map((r,i)=>`<tr><td>${i+1}</td><td>${r.DEPTH}</td><td>${r.P10}</td><td>${r.P50}</td><td>${r.P90}</td><td>${r.RELIABILITY}</td><td>${r.RISK}</td></tr>`).join('')}</tbody></table></div></div>`;
      Plotly.newPlot('uncertaintyChart', [
        {x:sat.data.map(r=>Number(r.P10)), y:sat.data.map(r=>Number(r.DEPTH)), mode:'lines', name:'P10', line:{color:'#f8c84c', width:2, dash:'dot'}},
        {x:sat.data.map(r=>Number(r.P50)), y:sat.data.map(r=>Number(r.DEPTH)), mode:'lines', name:'P50', line:{color:'#3dd5f3', width:3}},
        {x:sat.data.map(r=>Number(r.P90)), y:sat.data.map(r=>Number(r.DEPTH)), mode:'lines', name:'P90', line:{color:'#ff7a90', width:2, dash:'dash'}}
      ], {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#d7dde8'}, yaxis:{autorange:'reversed', title:'Depth'}, xaxis:{title:'Water Saturation (%)'}, margin:{t:30,l:50,r:20,b:40}, legend:{orientation:'h'}}, {responsive:true, displaylogo:false});
  } catch (err) {
    wrap.innerHTML = `<div class="alert alert-danger mb-0">${err.message}</div>`;
  }
}


