/* cuSHR DAG visualizer front-end. Renders one sentence's segmentation lattice
   with Cytoscape + dagre, left-to-right, and toggles a gold-path highlight. */
cytoscape.use(window.cytoscapeDagre);

let cy = null;
let goldOn = true;

const styles = [
  { selector: 'node', style: {
      'label': 'data(word)', 'font-size': 11, 'text-valign': 'center',
      'color': '#10203a', 'background-color': '#cdd8f3', 'border-width': 1,
      'border-color': '#8aa0d6', 'shape': 'round-rectangle', 'width': 'label',
      'height': 22, 'padding': '6px', 'text-wrap': 'wrap' } },
  { selector: 'edge', style: {
      'width': 1.4, 'line-color': '#b6c0d4', 'target-arrow-color': '#b6c0d4',
      'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'arrow-scale': 0.8 } },
  { selector: 'node.gold', style: {
      'background-color': '#ffd34d', 'border-color': '#e0a200', 'border-width': 3,
      'font-weight': 'bold' } },
  { selector: 'edge.gold', style: {
      'line-color': '#e0a200', 'target-arrow-color': '#e0a200', 'width': 3.5, 'z-index': 99 } },
  { selector: '.dim', style: { 'opacity': 0.35 } },
];

function applyGold(data) {
  const goldNodes = new Set(data.gold_path);
  const goldPairs = new Set();
  for (let i = 0; i + 1 < data.gold_path.length; i++) {
    goldPairs.add(data.gold_path[i] + '->' + data.gold_path[i + 1]);
  }
  cy.batch(() => {
    cy.elements().removeClass('gold dim');
    if (!goldOn || data.gold_path.length === 0) return;
    cy.nodes().forEach(n => { if (goldNodes.has(n.id())) n.addClass('gold'); else n.addClass('dim'); });
    cy.edges().forEach(e => {
      const key = e.data('source') + '->' + e.data('target');
      if (goldPairs.has(key)) e.addClass('gold'); else e.addClass('dim');
    });
  });
}

async function load(idx) {
  const meta = document.getElementById('meta');
  meta.className = '';
  meta.textContent = 'loading…';
  let data;
  try {
    const r = await fetch('/api/sentence/' + idx);
    if (!r.ok) { meta.className = 'err'; meta.textContent = 'sentence ' + idx + ' not found'; return; }
    data = await r.json();
  } catch (e) { meta.className = 'err'; meta.textContent = 'network error'; return; }

  const elements = [];
  data.nodes.forEach(n => elements.push({ data: {
    id: n.id, word: n.word || '∅', lemma: n.lemma, morph: n.morph, cng: n.cng, chunk: n.chunk } }));
  data.edges.forEach(e => elements.push({ data: {
    id: e.source + '_' + e.target, source: e.source, target: e.target } }));

  if (cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements, style: styles,
    layout: { name: 'dagre', rankDir: 'LR', nodeSep: 14, rankSep: 55, edgeSep: 6 },
  });

  const tip = document.getElementById('tip');
  cy.on('mouseover', 'node', evt => {
    const d = evt.target.data();
    tip.innerHTML = `<b>${d.word}</b><br>lemma: ${d.lemma}<br>morph: ${d.morph}<br>cng: ${d.cng} · chunk: ${d.chunk}`;
    tip.style.display = 'block';
  });
  cy.on('mousemove', evt => {
    tip.style.left = (evt.originalEvent.clientX + 12) + 'px';
    tip.style.top = (evt.originalEvent.clientY + 12) + 'px';
  });
  cy.on('mouseout', 'node', () => { tip.style.display = 'none'; });

  applyGold(data);
  window._data = data;
  document.getElementById('idx').value = idx;
  meta.textContent = `sentence ${idx} · ${data.stem}.graphml (sent_id ${data.stem}) · ` +
    `${data.nodes.length} nodes · ${data.edges.length} edges · ` +
    (data.gold_path.length ? `gold path: ${data.gold_path.length} words` : 'no gold path');
}

async function findByStem(stem) {
  const meta = document.getElementById('meta');
  try {
    const r = await fetch('/api/lookup/' + encodeURIComponent(stem));
    if (!r.ok) { meta.className = 'err'; meta.textContent = `sent_id ${stem} not found`; return; }
    const { idx } = await r.json();
    load(idx);
  } catch (e) { meta.className = 'err'; meta.textContent = 'network error'; }
}

document.getElementById('load').onclick = () => load(parseInt(document.getElementById('idx').value || '0', 10));
document.getElementById('idx').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('load').click(); });
document.getElementById('find').onclick = () => { const s = document.getElementById('stem').value.trim(); if (s) findByStem(s); };
document.getElementById('stem').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('find').click(); });
document.getElementById('gold').onchange = e => { goldOn = e.target.checked; if (window._data) applyGold(window._data); };

load(0);
