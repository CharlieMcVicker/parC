import os
import pytest
import pynini
from parC.constants import get_yaml_dir
from parC.fst_utils import ReservedSymbolMixin as R
from parC.grammar.acceptor_compilation import fsa, word_fsa, get_sigma_star, get_symbol_table, fsm_strings
from parC.yaml_utils.models import SingleStringMarker
from parC.grammar.transducer_compilation import compile_marker, compile_gated_marker, get_gated_marker_fst

# Set the active test yaml directory
os.environ["YAML_DIR"] = "yaml/spanish-example"

def test_suffix_transformer():
    """Test that a suffix transformer correctly appends the suffix before [EOW]."""
    marker = SingleStringMarker(kind="suffix", value="-suf")
    fst = compile_marker(marker)
    
    # Test on a simple root
    root = fsa("[BOW]habl[EOW]")
    res = pynini.compose(root, fst)
    out = fsm_strings(res, strip_all_tags=True)
    assert "habl-suf" in out

def test_prefix_transformer():
    """Test that a prefix transformer correctly prepends the prefix after [BOW]."""
    marker = SingleStringMarker(kind="prefix", value="pre-")
    fst = compile_marker(marker)
    
    root = fsa("[BOW]habl[EOW]")
    res = pynini.compose(root, fst)
    out = fsm_strings(res, strip_all_tags=True)
    assert "pre-habl" in out

def test_gated_suffix_trigger_matched():
    """Test that gated suffix applies when the trigger tags match."""
    marker = SingleStringMarker(kind="suffix", value="-as")
    gated = compile_gated_marker(marker, ["[person_number=2sg]", "[tense=present]"])
    
    # Input has matching tags
    inp = fsa("[BOW]cant[EOW][person_number=2sg][tense=present]")
    res = pynini.compose(inp, gated)
    out = fsm_strings(res)
    assert any("cant-as" in s for s in out)
    assert any("[person_number=2sg]" in s for s in out)
    assert any("[tense=present]" in s for s in out)

def test_gated_suffix_trigger_mismatched():
    """Test that gated suffix acts as identity when trigger tags do not match."""
    marker = SingleStringMarker(kind="suffix", value="-as")
    gated = compile_gated_marker(marker, ["[person_number=2sg]", "[tense=present]"])
    
    # Input has different tags (1sg instead of 2sg)
    inp = fsa("[BOW]cant[EOW][person_number=1sg][tense=present]")
    res = pynini.compose(inp, gated)
    out = fsm_strings(res)
    
    # Should not have changed the stem to "cant-as"
    assert not any("cant-as" in s for s in out)
    # Stem should remain "cant"
    assert any("cant" in s and not "cant-as" in s for s in out)
    # Tags should be preserved
    assert any("[person_number=1sg]" in s for s in out)

def test_composition_cascade():
    """Test composing a sequence of gated transducers."""
    # Suffix 1 (applied if 2sg present)
    m1 = SingleStringMarker(kind="suffix", value="-a")
    g1 = compile_gated_marker(m1, ["[person_number=2sg]"])
    
    # Suffix 2 (applied if present tense)
    m2 = SingleStringMarker(kind="suffix", value="-s")
    g2 = compile_gated_marker(m2, ["[tense=present]"])
    
    # Cascade: Input -> g1 -> g2
    # Case 1: matches both triggers -> cant -> cant-a -> cant-a-s
    inp1 = fsa("[BOW]cant[EOW][person_number=2sg][tense=present]")
    res1 = pynini.compose(inp1, g1)
    res1 = pynini.compose(res1, g2)
    out1 = fsm_strings(res1)
    assert any("cant-a-s" in s for s in out1)
    
    # Case 2: matches only tense=present -> cant -> cant -> cant-s
    inp2 = fsa("[BOW]cant[EOW][person_number=3sg][tense=present]")
    res2 = pynini.compose(inp2, g1)
    res2 = pynini.compose(res2, g2)
    out2 = fsm_strings(res2)
    assert any("cant-s" in s for s in out2)
    assert not any("cant-a" in s for s in out2)


def test_cached_gated_marker():
    """Test that the cached get_gated_marker_fst compiles and caches correctly."""
    marker = SingleStringMarker(kind="suffix", value="-as")
    tags = ("[person_number=2sg]", "[tense=present]")
    g1 = get_gated_marker_fst(marker, tags)
    g2 = get_gated_marker_fst(marker, tags)
    assert g1 is g2  # Should be the same cached instance
    
    inp = fsa("[BOW]cant[EOW][person_number=2sg][tense=present]")
    res = pynini.compose(inp, g1)
    assert any("cant-as" in s for s in fsm_strings(res))

