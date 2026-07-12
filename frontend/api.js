export async function fetchGrammarStats() {
  const res = await fetch("/grammar-stats");
  if (res.status === 503) {
    const body = await res.json().catch(() => ({}));
    throw Object.assign(new Error(body.detail ?? "Grammar not loaded"), { status: 503 });
  }
  if (!res.ok) throw new Error(`stats: ${res.status}`);
  return res.json();
}

export async function fetchInflectionMeta() {
  const res = await fetch("/inflection-meta");
  if (!res.ok) throw new Error(`inflection meta: ${res.status}`);
  return res.json();
}

export async function fetchRoots(kind, name) {
  const res = await fetch(`/roots?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`roots: ${res.status}`);
  return res.json();
}

export async function fetchLexicalFeatures(kind, name, root) {
  const res = await fetch(`/lexical-features?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}&root=${encodeURIComponent(root)}`);
  if (!res.ok) throw new Error(`lexical features: ${res.status}`);
  return res.json();
}

export async function fetchPatterns() {
  const res = await fetch("/patterns");
  if (!res.ok) throw new Error(`patterns: ${res.status}`);
  return res.json();
}

export async function fetchRules() {
  const res = await fetch("/rules");
  if (!res.ok) throw new Error(`rules: ${res.status}`);
  return res.json();
}

export async function testPattern(pattern, testIncludes, testExcludes) {
  const res = await fetch("/test-pattern", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pattern, test_includes: testIncludes, test_excludes: testExcludes }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Test failed: ${res.status}`);
  }
  return res.json();
}

export async function testRule(rule, testMappings) {
  const res = await fetch("/test-rule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rule, test_mappings: testMappings }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Test failed: ${res.status}`);
  }
  return res.json();
}

export async function parse(kind, name, form, openEnded = false) {
  const res = await fetch("/parse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, name, form, open_ended: openEnded }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Parse failed: ${res.status}`);
  }
  return res.json();
}

export async function search(kind, name, form, openEnded = false) {
  const res = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, name, form, open_ended: openEnded }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Search failed: ${res.status}`);
  }
  return res.json();
}

export async function runInflection(name, root, features) {
  const res = await fetch("/inflect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, root, features })
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Inflection failed: ${res.status}`);
  }
  return res.json();
}


