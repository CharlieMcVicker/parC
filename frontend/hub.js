import { fetchGrammarStats, recompile } from './api.js';

let lastHealth = null;
let lastStats = null;
let isFetching = false;

const statusDot = document.querySelector('.status-dot');
const statusLabel = document.querySelector('.status-label');
const statsSection = document.getElementById('stats-section');
const statsGrid = document.getElementById('stats-grid');
const recompileBtn = document.getElementById('recompile-btn');


const CARD_GROUPS = {
  Phonology: {
    inventory: {
      title: 'Inventory',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.phones} phones`, `${d.tags} tags`, `${d.classes} classes`]
    },
    patterns: {
      title: 'Patterns',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.total} patterns`]
    },
    rules: {
      title: 'Rules',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.total} rules`]
    },
  },
  Exponence: {
    feature_definitions: {
      title: 'Feat. Definitions',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.total} features`]
    },
    feature_markers: {
      title: 'Feat. Markers',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.total} markers`, `${d.inflection_stages} inflection stages`]
    },
    contingent_markers: {
      title: 'Cont. Markers',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`, `${d.total} markers`]
    },
  },
  Lexicon: {
    part_of_speech: {
      title: 'Part of Sp.',
      format: (d) => [`${d.files} files`, `${d.roots} roots`, `${d.invalid_files} invalid files`]
    },
  },
  Morphotactics: {
    paradigms: {
      title: 'Paradigms',
      format: (d) => [`${d.files} files`, `${d.invalid_files} invalid files`]
    },
  },
};

async function poll() {
  if (isFetching) return;
  isFetching = true;
  try {
    const stats = await fetchGrammarStats();
    lastStats = stats;
    renderStats(lastStats, false);
    statusDot.className = 'status-dot online';
    statusLabel.textContent = 'Online';
  } catch (err) {
    statusDot.className = 'status-dot offline';
    statusLabel.textContent = 'Offline';
    renderStats(lastStats, true);
  } finally {
    isFetching = false;
  }
}

async function handleRecompile() {
  recompileBtn.disabled = true;
  try {
    await recompile();
    await poll();
  } catch (err) {
    console.error("Recompile failed", err);
  } finally {
    recompileBtn.disabled = false;
  }
}

recompileBtn.addEventListener('click', handleRecompile);
setInterval(poll, 5000);
poll();

function renderStats(stats, isStale) {
  if (!stats) {
    statsSection.setAttribute('hidden', '');
    return;
  }
  statsSection.removeAttribute('hidden');

  if (isStale) {
    statsGrid.classList.add('stale');
  } else {
    statsGrid.classList.remove('stale');
  }

  statsGrid.innerHTML = '';
  for (const [groupName, cards] of Object.entries(CARD_GROUPS)) {
    const groupEl = document.createElement('div');
    groupEl.className = 'stats-group';

    const heading = document.createElement('h4');
    heading.className = 'stats-group-title';
    heading.textContent = groupName;
    groupEl.appendChild(heading);

    const cardsEl = document.createElement('div');
    cardsEl.className = 'stats-group-cards';

    for (const [key, meta] of Object.entries(cards)) {
      const data = stats[key];
      if (!data) continue;

      const card = document.createElement('div');
      card.className = 'stat-card';

      const title = document.createElement('h5');
      title.textContent = meta.title;
      card.appendChild(title);

      const list = document.createElement('ul');
      meta.format(data).forEach(line => {
        const li = document.createElement('li');
        li.textContent = line;
        list.appendChild(li);
      });
      card.appendChild(list);
      cardsEl.appendChild(card);
    }

    groupEl.appendChild(cardsEl);
    statsGrid.appendChild(groupEl);
  }
}
