import { fetchInflectionMeta, parse, search } from './api.js';

let metaData = null;
let currentParses = [];
let currentPage = 1;
const itemsPerPage = 10;

const targetSelect = document.getElementById('parse-target');
const formInput = document.getElementById('parse-form');
const submitBtn = document.getElementById('submit-parse-btn');
const fuzzyToggle = document.getElementById('fuzzy-toggle-btn');
const openEndedToggle = document.getElementById('open-ended-toggle-btn');
const resultsSection = document.getElementById('parse-results');
const resultsTable = document.getElementById('parse-results-table');

const paginationContainer = document.getElementById('parse-pagination');
const prevBtn = document.getElementById('parse-prev-btn');
const nextBtn = document.getElementById('parse-next-btn');
const pageInfo = document.getElementById('parse-page-info');

async function loadMeta() {
  try {
    metaData = await fetchInflectionMeta();
    updateTargets();
  } catch (err) {
    console.error('Failed to load parse meta:', err);
  }
}

function updateTargets() {
  if (!metaData) return;
  targetSelect.innerHTML = '';
  metaData.paradigms.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.name;
    opt.textContent = t.name;
    targetSelect.appendChild(opt);
  });
}

function renderParsesPage() {
  const thead = resultsTable.tHead || resultsTable.createTHead();
  const tbody = resultsTable.tBodies[0] || resultsTable.createTBody();
  thead.innerHTML = '';
  tbody.innerHTML = '';

  if (!currentParses.length) {
    const row = tbody.insertRow();
    const cell = row.insertCell();
    cell.textContent = '(no parses)';
    cell.colSpan = 99;
    paginationContainer.style.display = 'none';
    return;
  }

  const totalPages = Math.ceil(currentParses.length / itemsPerPage);
  if (currentPage < 1) currentPage = 1;
  if (currentPage > totalPages) currentPage = totalPages;

  const fuzzy = fuzzyToggle.checked;
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, currentParses.length);
  const pageItems = currentParses.slice(startIndex, endIndex);

  const featKeys = [...new Set(currentParses.flatMap(p => Object.keys(p.features || {})))];
  const headers = [
    ...(fuzzy ? ['Form'] : []),
    'Root', 'Gloss',
    ...featKeys,
    ...(fuzzy ? ['Edit distance'] : []),
  ];
  const hRow = thead.insertRow();
  headers.forEach(h => { const th = document.createElement('th'); th.textContent = h; hRow.appendChild(th); });

  pageItems.forEach(p => {
    const row = tbody.insertRow();
    const cells = [
      ...(fuzzy ? [p.form ?? ''] : []),
      `√${p.root}`, p.gloss || '',
      ...featKeys.map(k => (p.features || {})[k] ?? ''),
      ...(fuzzy ? [p.edit_distance ?? ''] : []),
    ];
    cells.forEach(val => { const td = row.insertCell(); td.textContent = val; });
  });

  if (totalPages > 1) {
    paginationContainer.style.display = 'flex';
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages;
  } else {
    paginationContainer.style.display = 'none';
  }
}

submitBtn.addEventListener('click', async () => {
  const name = targetSelect.value;
  const form = formInput.value.trim();
  if (!form) return;

  submitBtn.disabled = true;
  resultsSection.setAttribute('hidden', '');
  paginationContainer.style.display = 'none';
  try {
    const isFuzzy = fuzzyToggle.checked;
    const isOpenEnded = openEndedToggle.checked;
    const data = isFuzzy
      ? await search("Paradigm", name, form, isOpenEnded)
      : await parse("Paradigm", name, form, isOpenEnded);
    
    currentParses = data.parses || [];
    currentPage = 1;
    renderParsesPage();
    resultsSection.removeAttribute('hidden');
  } catch (err) {
    alert(err.message);
  } finally {
    submitBtn.disabled = false;
  }
});

prevBtn.addEventListener('click', () => {
  if (currentPage > 1) {
    currentPage--;
    renderParsesPage();
  }
});

nextBtn.addEventListener('click', () => {
  const totalPages = Math.ceil(currentParses.length / itemsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderParsesPage();
  }
});

document.querySelector('[data-tab="parse"]').addEventListener('click', () => {
  if (!metaData) loadMeta();
});
