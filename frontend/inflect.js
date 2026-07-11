import { fetchInflectionMeta, fetchRoots, fetchLexicalFeatures, runInflection } from './api.js';

let metaData = null;

const targetSelect = document.getElementById('inflect-target');
const rootContainer = document.getElementById('root-container');
const featuresContainer = document.getElementById('features-container');
const submitBtn = document.getElementById('submit-inflect-btn');
const resultsSection = document.getElementById('inflection-results');
const formsList = document.getElementById('result-forms-list');
const stagesTableHead = document.querySelector('#stages-table thead');
const stagesTableBody = document.querySelector('#stages-table tbody');
const navInflectBtn = document.getElementById('nav-inflect-btn');

async function loadMeta() {
  try {
    metaData = await fetchInflectionMeta();
    updateTargets();
  } catch (err) {
    console.error('Failed to load inflection meta:', err);
  }
}

function updateTargets() {
  if (!metaData) return;
  targetSelect.innerHTML = '';

  const targets = metaData.paradigms
  targets.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.name;
    opt.textContent = t.name;
    targetSelect.appendChild(opt);
  });

  updateFields();
}

async function updateFields() {
  if (!metaData) return;
  const targetName = targetSelect.value;

  rootContainer.innerHTML = '';
  featuresContainer.innerHTML = '';
  resultsSection.setAttribute('hidden', '');

  if (!targetName) return;

  const p = metaData.paradigms.find(x => x.name === targetName);
  if (!p) return;

  const select = buildRootSelect(`Select a root…`);
  select.addEventListener('change', () => onRootChange("Paradigm", targetName, select.value));
  rootContainer.appendChild(select);

  // const allFeatures = [...new Set([...p.features, ...p.lexical_features])];
  // TODO: when MorphemeSequence is implemented, we'll need to account for lexcical features as input
  // For the Paradigm class, we only need to worry about inflection features
  renderFeatureSelectors(p.features);

  try {
    const roots = await fetchRoots("Paradigm", targetName);
    populateRootSelect(select, roots);
  } catch (err) {
    console.warn('Failed to load roots:', err);
  }

}

function buildRootSelect(placeholder) {
  const select = document.createElement('select');
  select.className = 'root-input root-select';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = placeholder;
  select.appendChild(blank);
  return select;
}

function populateRootSelect(select, roots) {
  roots.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r;
    opt.textContent = r;
    select.appendChild(opt);
  });
}

async function onRootChange(kind, name, root) {
  if (!root) {
    featuresContainer.querySelectorAll('.feature-select').forEach(sel => {
      sel.disabled = false;
    });
    return;
  }
  try {
    const lexFeatures = await fetchLexicalFeatures(kind, name, root);
    applyLexicalFeatures(lexFeatures);
  } catch (err) {
    console.warn('Failed to load lexical features:', err);
  }
}

function applyLexicalFeatures(lexFeatures) {
  featuresContainer.querySelectorAll('.feature-select').forEach(sel => {
    const val = lexFeatures[sel.dataset.feature];
    if (val !== undefined) {
      sel.value = val;
      sel.disabled = true;
    } else {
      sel.disabled = false;
    }
  });
}

function renderFeatureSelectors(featuresList) {
  featuresList.forEach(featName => {
    const values = metaData.features[featName] || [];

    const fieldWrapper = document.createElement('div');
    fieldWrapper.style.display = 'flex';
    fieldWrapper.style.flexDirection = 'column';
    fieldWrapper.style.gap = '0.25rem';

    const label = document.createElement('span');
    label.textContent = featName;
    label.style.fontSize = '0.85rem';
    label.style.fontWeight = 'bold';

    const select = document.createElement('select');
    select.className = 'feature-select';
    select.dataset.feature = featName;

    const defOpt = document.createElement('option');
    defOpt.value = 'unmarked';
    defOpt.textContent = 'unmarked';
    select.appendChild(defOpt);

    values.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    });

    fieldWrapper.appendChild(label);
    fieldWrapper.appendChild(select);
    featuresContainer.appendChild(fieldWrapper);
  });
}

targetSelect.addEventListener('change', updateFields);
navInflectBtn.addEventListener('click', loadMeta);

submitBtn.addEventListener('click', async () => {
  if (!metaData) return;
  const name = targetSelect.value;

  const rootInputs = rootContainer.querySelector('.root-input');
  const root = rootInputs.value.trim();

  if (root.length === 0) {
    alert('Please enter a root / root.');
    return;
  }

  const features = {};
  const featureSelects = featuresContainer.querySelectorAll('.feature-select');
  featureSelects.forEach(sel => {
    features[sel.dataset.feature] = sel.value;
  });

  submitBtn.disabled = true;
  submitBtn.textContent = 'Generating...';

  try {
    const res = await runInflection(name, root, features);
    displayResults(res);
  } catch (err) {
    alert(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Generate Inflected Forms';
  }
});

function displayResults(data) {
  resultsSection.removeAttribute('hidden');

  formsList.innerHTML = '';
  if (data.forms && data.forms.length > 0) {
    data.forms.forEach(f => {
      const div = document.createElement('div');
      div.textContent = f;
      formsList.appendChild(div);
    });
  } else {
    formsList.innerHTML = '<span style="color:#ef4444;">No forms generated.</span>';
  }

  stagesTableHead.innerHTML = '';
  stagesTableBody.innerHTML = '';

  stagesTableHead.innerHTML = `
    <tr>
      <th>Order</th>
      <th>Marker Kind</th>
      <th>Marker Value</th>
      <th>Feature Value</th>
      <th>Form</th>
    </tr>
  `;

  data.stages.forEach(s => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>${s.stage ?? ''}</td>
        <td>${s.marker_kind ?? ''}</td>
        <td>${s.marker_value ?? ''}</td>
        <td>${s.feature_values ?? ''}</td>
        <td>${s.surface_forms ?? ''}</td>
      `;
    stagesTableBody.appendChild(tr);
  });
}

loadMeta();
