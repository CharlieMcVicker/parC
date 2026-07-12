"""
Functional FSA compilation for parC grammar.

Compiles inventory + features into a pynini SymbolTable, builds a token map
for pattern string tokenization, and compiles all patterns into pynini FSAs
via recursive descent parsing.

Caches:
  symbol table  → get_yaml_dir()/.cache/symbol_table.syms  (inv + feat dirs)
  pattern FSAs  → in-memory only                      (inv + feat + pat dirs)
  token map     → in-memory only                      (same three dirs)
  special FSAs  → in-memory only                      (inv + feat dirs)
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from graphlib import TopologicalSorter
from typing import Literal

import pynini
from pynini.lib import rewrite

from parC.yaml_utils.cache import (
    observed_cache,
    compute_cache_key,
    get_cached_symbol_table,
    save_cached_symbol_table,
    get_cached_pattern_fsts,
    save_cached_pattern_fsts,
)
from parC.fst_utils import ReservedSymbolMixin as R
from parC.yaml_utils.models import Feature, Inventory, Pattern, Token
from parC.yaml_utils.yaml_server import (
    get_feature_array,
    get_inventory_items,
    get_patterns,
)
from parC.yaml_utils.yaml_server import kind_dir

"""
## Symbol table
"""


def build_symbol_table(
    inventory: Inventory,
    features: tuple[Feature, ...],
) -> pynini.SymbolTable:
    syms = pynini.SymbolTable()
    syms.add_symbol(R.epsilon_ref)
    for phone in dict.fromkeys(inventory.phones):
        syms.add_symbol(phone)
    for tag in dict.fromkeys(inventory.tags):
        syms.add_symbol(tag)
    for feature in features:
        for value in feature.values:
            syms.add_symbol(f"[{feature.name}={value}]")
            syms.add_symbol(f"<{feature.name}.discharged={value}>")
    for sym in R.boundary_symbols:
        syms.add_symbol(sym)
    for sym in R.edit_tags:
        syms.add_symbol(sym)
    for sym in R.bow_eow_tags:
        syms.add_symbol(sym)

    from parC.grammar.op_tags import get_all_op_tags
    for op_tag in sorted(get_all_op_tags()):
        syms.add_symbol(op_tag)

    return syms


def get_symbol_table_key() -> str:
    config_dirs = [
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
        kind_dir("Paradigm"),
        kind_dir("FeatureMarkers"),
        kind_dir("ContingentFeatureMarkers"),
    ]
    return compute_cache_key("symbol_table", "SymbolTable", config_dirs)


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
        kind_dir("Paradigm"),
        kind_dir("FeatureMarkers"),
        kind_dir("ContingentFeatureMarkers"),
    ]
)
def get_symbol_table() -> pynini.SymbolTable:
    cache_key = get_symbol_table_key()
    loaded = get_cached_symbol_table(cache_key)
    if loaded is not None:
        return loaded
    syms = build_symbol_table(get_inventory_items(), get_feature_array())
    save_cached_symbol_table(cache_key, syms)
    symbol_table = syms
    return symbol_table


"""
## Special FSAs
Sigma, phone, flag, boundary etc. — derived from symbol table; cached in-memory.
"""


def _build_special_fsas(
    syms: pynini.SymbolTable,
    inventory: Inventory,
    features: tuple[Feature, ...],
) -> dict[str, pynini.Fst]:
    phones = list(dict.fromkeys(inventory.phones))
    if not phones:
        raise ValueError(
            "Cannot build sigma FSAs without any phones in inventory.")
    phone_fsa = pynini.union(
        *[pynini.accep(p, token_type=syms) for p in phones]
    ).optimize()

    all_tags = list(dict.fromkeys(inventory.tags))
    for feat in features:
        for val in feat.values:
            all_tags.append(f"[{feat.name}={val}]")
            all_tags.append(f"<{feat.name}.discharged={val}>")
    from parC.grammar.op_tags import get_all_op_tags
    for op_tag in get_all_op_tags():
        all_tags.append(op_tag)
    flag_fsa = (
        pynini.union(*[pynini.accep(t, token_type=syms)
                     for t in all_tags]).optimize()
        if all_tags
        else pynini.accep("", token_type=syms)
    )

    affix_fsa = pynini.accep(R.affix_boundary, token_type=syms)
    clitic_fsa = pynini.accep(R.clitic_boundary, token_type=syms)
    periphrase_fsa = pynini.accep(R.periphrasis_break, token_type=syms)
    boundary_fsa = pynini.union(affix_fsa, clitic_fsa, periphrase_fsa)

    bow_fsa = pynini.accep(R.bow, token_type=syms)
    eow_fsa = pynini.accep(R.eow, token_type=syms)
    word_edge_fsa = pynini.union(bow_fsa, eow_fsa)

    sigma = pynini.union(phone_fsa, flag_fsa, boundary_fsa,
                         word_edge_fsa).optimize()
    sigma_star = sigma.star.optimize()

    return {
        "phone": phone_fsa,
        "flag": flag_fsa,
        "sigma": sigma,
        "sigma_star": sigma_star,
        "bow": bow_fsa,
        "eow": eow_fsa,
        "word_edge": word_edge_fsa,
        "boundary": boundary_fsa,
        "affix_boundary": affix_fsa,
        "clitic_boundary": clitic_fsa,
        "periphrasis_break": periphrase_fsa,
    }


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def get_special_fsas() -> dict[str, pynini.Fst]:
    syms = get_symbol_table()

    inventory = get_inventory_items()
    features = get_feature_array()
    special_fsas = _build_special_fsas(syms, inventory, features)
    return special_fsas


"""
## Token map
Tokens store (value, kind) only — no embedded FSAs.
"""


def _build_token_map(
    syms: pynini.SymbolTable,
    inventory: Inventory,
    features: tuple[Feature, ...],
    patterns: dict[str, Pattern],
) -> dict[str, list[Token]]:
    tokens: dict[str, list[Token]] = defaultdict(list)

    tokens["dot"].append(Token(R.dot, "special_ref"))

    tokens["ref"].extend(
        Token(ref, "special_ref")
        for ref in (R.phone_ref, R.flag_ref, R.sigma_ref, R.epsilon_ref, R.boundary_ref)
    )

    for d in R.left_delimiters:
        tokens["left_delimiter"].append(Token(d, "left_delimiter"))
    for d in R.right_delimiters:
        tokens["right_delimiter"].append(Token(d, "right_delimiter"))
    for op in R.unary_operators:
        tokens["unary_operator"].append(Token(op, "unary_operator"))
    tokens["pipe_operator"].append(Token(R.pipe_operator, "pipe_operator"))
    tokens["caret_operator"].append(Token(R.caret_operator, "caret_operator"))

    tokens["tag"].append(Token(R.bow, "bow_eow"))
    tokens["tag"].append(Token(R.eow, "bow_eow"))

    for tag in R.edit_tags:
        tokens["tag"].append(Token(tag, "edit_flag"))

    for sym in (R.affix_boundary, R.clitic_boundary, R.periphrasis_break):
        tokens["boundary"].append(Token(sym, "boundary"))

    for phone in inventory.phones:
        tokens["phone"].append(Token(phone, "phone"))

    for tag in inventory.tags:
        tokens["tag"].append(Token(tag, "tag"))

    for feat in features:
        for val in feat.values:
            tokens["tag"].append(Token(f"[{feat.name}={val}]", "tag"))
            tokens["tag"].append(Token(f"<{feat.name}.discharged={val}>", "tag"))

    from parC.grammar.op_tags import get_all_op_tags
    for op_tag in get_all_op_tags():
        tokens["tag"].append(Token(op_tag, "tag"))

    # inventory classes — FSAs built separately in _build_class_fsts
    for name in inventory.item_map:
        tokens["ref"].append(Token(name, "class_ref"))

    for ref in patterns:
        tokens["ref"].append(Token(ref, "pattern_ref"))

    return {kind: sorted(lst, key=len, reverse=True) for kind, lst in tokens.items()}


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def get_token_map() -> dict[str, list[Token]]:
    syms = get_symbol_table()
    inventory = get_inventory_items()
    features = get_feature_array()
    patterns = get_patterns()
    token_map = _build_token_map(syms, inventory, features, patterns)
    return token_map


"""
## Inventory class FSAs
Built separately from token map; merged into compiled_patterns before parsing.
"""


def _build_class_fsts(
    syms: pynini.SymbolTable,
    inventory: Inventory,
) -> dict[str, pynini.Fst]:
    phone_set = set(inventory.phones)
    tag_set = set(inventory.tags)
    result: dict[str, pynini.Fst] = {}
    for name, contents in inventory.item_map.items():
        if name in phone_set or name in tag_set:
            continue
        child_fsas = [pynini.accep(p, token_type=syms)
                      for p in contents.phones]
        child_fsas += [pynini.accep(t, token_type=syms) for t in contents.tags]
        if not child_fsas:
            continue
        result[name] = pynini.union(*child_fsas).optimize()
    return result


"""
## Recursive descent parser
"""


def _preprocess_str(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    if s.startswith(R.word_edge):
        s = R.bow + s[1:]
    if s.endswith(R.word_edge):
        s = s[:-1] + R.eow
    return s


def _infer_token_type(s: str, phone_starts: set[str]) -> str:
    c = s[0]
    if c in phone_starts:
        return "phone"
    if c == "[":
        return "tag"
    if c == "<":
        if ".discharged=" in s:
            return "tag"
        return "ref"
    if c in R.unary_operators:
        return "unary_operator"
    if c == R.pipe_operator:
        return "pipe_operator"
    if c == R.caret_operator:
        return "caret_operator"
    if c in R.left_delimiters:
        return "left_delimiter"
    if c in R.right_delimiters:
        return "right_delimiter"
    if c in R.boundary_symbols:
        return "boundary"
    if c == R.dot:
        return "dot"
    return "phone"


def _tokenize_str(
    pattern_str: str,
    token_map: dict[str, list[Token]],
    phone_starts: set[str],
) -> list[Token]:
    s = _preprocess_str(pattern_str)
    result: list[Token] = []
    i = 0
    while i < len(s):
        token_type = _infer_token_type(s[i:], phone_starts)
        match = next(
            (
                tok
                for tok in token_map.get(token_type, [])
                if s.startswith(tok.value, i)
            ),
            None,
        )
        if match is None:
            # breakpoint()
            raise ValueError(
                f"Unrecognized token at position {i} in '{s}' "
                f"(inferred type: '{token_type}')"
            )
        result.append(match)
        i += len(match)
    return result


def _interpret_unary_operator(fst: pynini.Fst, op: str) -> pynini.Fst:
    if op == "?":
        return fst.ques
    if op == "+":
        return fst.plus
    if op == "*":
        return fst.star
    raise ValueError(f"Unknown unary operator: {op!r}")


def _atom_to_fst(
    tok: Token,
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
) -> pynini.Fst:
    if tok.kind in ("phone", "tag", "bow_eow", "edit_flag", "boundary"):
        return pynini.accep(tok.value, token_type=syms)
    if tok.kind in ("class_ref", "pattern_ref"):
        if tok.value not in compiled_patterns:
            raise ValueError(f"Ref '{tok.value}' not compiled yet")
        return compiled_patterns[tok.value]
    if tok.kind in ("special_ref", "dot"):
        if tok.value == R.phone_ref:
            return special_fsas["phone"]
        if tok.value == R.flag_ref:
            return special_fsas["flag"]
        if tok.value in (R.sigma_ref, R.dot):
            return special_fsas["sigma"]
        if tok.value == R.boundary_ref:
            return special_fsas["boundary"]
        if tok.value == R.epsilon_ref:
            return pynini.accep("", token_type=syms)
        raise ValueError(f"Unknown special ref: {tok.value!r}")
    raise ValueError(f"Cannot convert token {tok!r} to FSA")


def _parse_factor_sequence(
    tokens: list[Token],
    i: int,
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
    sigma: pynini.Fst,
) -> tuple[list[pynini.Fst], int]:
    fsas: list[pynini.Fst] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind in ("right_delimiter", "pipe_operator", "unary_operator"):
            break
        if tok.kind == "left_delimiter":
            f, i = _parse_delimited_factor(
                tokens, i, compiled_patterns, syms, special_fsas, sigma
            )
            fsas.append(f)
        else:
            fsas.append(_atom_to_fst(
                tok, compiled_patterns, syms, special_fsas))
            i += 1
    return fsas, i


def _parse_delimited_factor(
    tokens: list[Token],
    i: int,
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
    sigma: pynini.Fst,
) -> tuple[pynini.Fst, int]:
    left = tokens[i].value
    i += 1
    if left == "{":
        negated = i < len(tokens) and tokens[i].kind == "caret_operator"
        if negated:
            i += 1
        factors, i = _parse_factor_sequence(
            tokens, i, compiled_patterns, syms, special_fsas, sigma
        )
        inner = pynini.union(*factors)
        if negated:
            inner = pynini.difference(sigma, inner)
        expected = "}"
    elif left == "(":
        inner, i = _parse_expression(
            tokens, i, compiled_patterns, syms, special_fsas, sigma
        )
        expected = ")"
    else:
        raise ValueError(f"Unexpected left delimiter: {left!r}")
    if tokens[i].value != expected:
        raise ValueError(f"Expected '{expected}' but got '{tokens[i].value}'")
    return inner, i + 1


def _parse_term(
    tokens: list[Token],
    i: int,
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
    sigma: pynini.Fst,
) -> tuple[pynini.Fst, int]:
    if tokens[i].kind in ("right_delimiter", "pipe_operator"):
        raise ValueError(
            f"Unexpected {tokens[i].kind!r} token at start of term")
    fsas: list[pynini.Fst] = []
    while i < len(tokens) and tokens[i].kind not in (
        "right_delimiter",
        "pipe_operator",
    ):
        factor_list, i = _parse_factor_sequence(
            tokens, i, compiled_patterns, syms, special_fsas, sigma
        )
        if i < len(tokens) and tokens[i].kind == "unary_operator":
            if not factor_list:
                raise ValueError("Unary operator with no preceding factor")
            factor_list[-1] = _interpret_unary_operator(
                factor_list[-1], tokens[i].value
            )
            i += 1
        fsas.extend(factor_list)
    if not fsas:
        raise ValueError("Empty term")
    result = fsas[0]
    for f in fsas[1:]:
        result = pynini.concat(result, f)
    return result, i


def _parse_expression(
    tokens: list[Token],
    i: int,
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
    sigma: pynini.Fst,
) -> tuple[pynini.Fst, int]:
    term, i = _parse_term(tokens, i, compiled_patterns,
                          syms, special_fsas, sigma)
    terms = [term]
    while i < len(tokens) and tokens[i].kind == "pipe_operator":
        i += 1
        t, i = _parse_term(tokens, i, compiled_patterns,
                           syms, special_fsas, sigma)
        terms.append(t)
        if i >= len(tokens) or tokens[i].kind == "right_delimiter":
            break
    return pynini.union(*terms), i


def _parse_tokens(
    tokens: list[Token],
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    special_fsas: dict[str, pynini.Fst],
    sigma: pynini.Fst,
) -> pynini.Fst:
    fst, end = _parse_expression(
        tokens, 0, compiled_patterns, syms, special_fsas, sigma
    )
    if end != len(tokens):
        raise ValueError(f"Leftover tokens after parse: {tokens[end:]}")
    return fst


def _parse_pattern(
    pattern_str: str,
    token_map: dict[str, list[Token]],
    phone_starts: set[str],
    compiled_patterns: dict[str, pynini.Fst],
    syms: pynini.SymbolTable,
    sigma: pynini.Fst,
    special_fsas: dict[str, pynini.Fst],
) -> pynini.Fst:
    if not pattern_str:
        return pynini.accep("", token_type=syms)
    toks = _tokenize_str(pattern_str, token_map, phone_starts)
    fst = _parse_tokens(toks, compiled_patterns, syms, special_fsas, sigma)
    fst.optimize()
    return fst


"""
## Pattern compilation
"""


def compile_all_patterns(
    patterns: dict[str, Pattern],
    token_map: dict[str, list[Token]],
    phone_starts: set[str],
    syms: pynini.SymbolTable,
    sigma: pynini.Fst,
    special_fsas: dict[str, pynini.Fst],
    class_fsts: dict[str, pynini.Fst],
) -> dict[str, pynini.Fst]:
    """
    First compute the dependency graph across all pattern strings
    for topological sorting.
    """
    dep_graph: dict[str, set[str]] = {ref: set() for ref in patterns}
    for ref, pat in patterns.items():
        for token in re.findall(r"<([^>]+)>", pat.pattern):
            if token in patterns:
                dep_graph[ref].add(token)
    order = list(TopologicalSorter(dep_graph).static_order())

    compiled: dict[str, pynini.Fst] = dict(class_fsts)
    for ref in order:
        pat = patterns[ref]
        try:
            compiled[ref] = _parse_pattern(
                pat.pattern,
                token_map,
                phone_starts,
                compiled,
                syms,
                sigma,
                special_fsas,
            )
        except Exception as e:
            raise ValueError(f"Error compiling pattern '{ref}': {e}") from e
    return compiled


def get_pattern_fsts_key() -> str:
    config_dirs = [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
    child_keys = {"symbol_table": get_symbol_table_key()}
    return compute_cache_key("patterns", "Patterns", config_dirs, child_keys)


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def get_pattern_fsts() -> dict[str, pynini.Fst]:
    """Returns compiled_patterns (class FSTs + pattern FSTs). Disk-cached via SHA-256."""
    cache_key = get_pattern_fsts_key()
    loaded = get_cached_pattern_fsts(cache_key)
    if loaded is not None:
        return loaded
    syms = get_symbol_table()
    inventory = get_inventory_items()
    features = get_feature_array()
    patterns = get_patterns()
    special_fsas = get_special_fsas()
    class_fsts = _build_class_fsts(syms, inventory)
    phone_starts = {p[0] for p in inventory.phones}
    token_map = _build_token_map(syms, inventory, features, patterns)
    pattern_fsts = compile_all_patterns(
        patterns,
        token_map,
        phone_starts,
        syms,
        special_fsas["sigma"],
        special_fsas,
        class_fsts,
    )
    save_cached_pattern_fsts(cache_key, pattern_fsts)
    return pattern_fsts


"""
## FSA ↔ string utilities
"""


def get_sigma_star() -> pynini.Fst:
    sigma_star = get_special_fsas()["sigma_star"]
    return sigma_star


def _cached_context() -> tuple[
    dict[str, list[Token]],
    set[str],
    dict[str, pynini.Fst],
    pynini.SymbolTable,
    pynini.Fst,
    dict[str, pynini.Fst],
]:
    token_map = get_token_map()
    syms = get_symbol_table()
    compiled = get_pattern_fsts()
    special_fsas = get_special_fsas()
    phone_starts = {p[0] for p in get_inventory_items().phones}
    sigma = special_fsas["sigma"]
    return token_map, phone_starts, compiled, syms, sigma, special_fsas


"""
### String → FSA
"""


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def fsa(pattern_str: str) -> pynini.Fst:
    token_map, phone_starts, compiled, syms, sigma, special_fsas = _cached_context()
    return _parse_pattern(
        pattern_str, token_map, phone_starts, compiled, syms, sigma, special_fsas
    )


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def word_fsa(word_str: str, prefix: str | None = None) -> pynini.Fst:
    import re
    # Extract trailing discharged tags to place them outside [EOW]
    match = re.search(r"((?:<[^>]+\.discharged=[^>]+>)+)$", word_str)
    if match:
        tags = match.group(1)
        base_word = word_str[:-len(tags)]
        tagged = R.bow + base_word + R.eow + tags
    else:
        tagged = R.bow + word_str + R.eow
        
    if prefix:
        tagged = prefix + tagged
    token_map, phone_starts, compiled, syms, sigma, special_fsas = _cached_context()
    return _parse_pattern(
        tagged, token_map, phone_starts, compiled, syms, sigma, special_fsas
    )


@observed_cache(
    [
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def wordlist_fsa(words: list[str]) -> pynini.Fst:
    return pynini.union(*[word_fsa(w) for w in words]).optimize()


"""
### FSA → string
"""


def _decode_labels(
    label_iter,
    syms: pynini.SymbolTable,
    strip_word_edge_symbols: bool = False,
    strip_all_tags: bool = False,
) -> str:
    word = ""
    for label in label_iter:
        if label == 0:
            continue
        symbol = syms.find(label)
        if strip_all_tags and symbol[0] == "[":
            continue
        if strip_word_edge_symbols and symbol in R.bow_eow_tags:
            continue
        word += symbol
    return word


def fsm_strings_and_weights(
    fst: pynini.Fst,
    project: Literal["input", "output"] = "output",
    nshortest: int | None = None,
    strip_word_edge_symbols: bool = False,
    strip_all_tags: bool = False,
) -> list[tuple[str, float]]:
    syms = get_symbol_table()
    projected = pynini.project(fst, project_type=project)
    if nshortest is not None:
        projected = rewrite.lattice_to_nshortest(
            projected, nshortest=nshortest)
    seen: set[str] = set()
    decoded: list[tuple[str, float]] = []
    path_iter = projected.paths()
    while not path_iter.done():
        word = _decode_labels(
            path_iter.olabels(),
            syms,
            strip_word_edge_symbols=strip_word_edge_symbols,
            strip_all_tags=strip_all_tags,
        )
        if word not in seen:
            seen.add(word)
            decoded.append((word, float(path_iter.weight())))
        path_iter.next()
    decoded.sort(key=lambda t: t[1])
    return decoded


def fsm_strings(
    fst: pynini.Fst,
    project: Literal["input", "output"] = "output",
    nshortest: int | None = None,
    strip_word_edge_symbols: bool = False,
    strip_all_tags: bool = False,
) -> list[str]:
    return [
        s
        for s, _ in fsm_strings_and_weights(
            fst, project, nshortest, strip_word_edge_symbols, strip_all_tags
        )
    ]


def fsm_string(
    fst: pynini.Fst,
    project: Literal["input", "output"] = "output",
    strip_word_edge_symbols: bool = False,
    strip_all_tags: bool = False,
) -> str:
    strings = fsm_strings(
        fst, project, 1, strip_word_edge_symbols, strip_all_tags)
    if len(strings) != 1:
        raise ValueError(f"Expected single string, got {strings}")
    return strings[0]


def filter_strings_by_pattern(
    input_fst: pynini.Fst,
    pattern_fst: pynini.Fst,
) -> list[str]:
    return fsm_strings(pynini.intersect(input_fst, pattern_fst).optimize())
