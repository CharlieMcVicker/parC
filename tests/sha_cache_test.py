import os
import tempfile
import json
import pytest
from parC.constants import get_yaml_dir
from parC.yaml_utils.cache import (
    get_file_sha256,
    get_dir_sha256,
    compute_cache_key,
    CACHE_DIR,
)
from parC.grammar.transducer_compilation import get_rule_fst_key
from parC.grammar.paradigm_compilation import get_paradigm_cache_key


def test_file_and_dir_hashing():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "test1.yaml")
        with open(file1, "w") as f:
            f.write("key: value")
            
        hash1 = get_file_sha256(file1)
        assert len(hash1) == 64
        
        # Directory hash
        dir_hash1 = get_dir_sha256(tmpdir)
        assert len(dir_hash1) == 64
        
        # Modify file -> hashes should change
        with open(file1, "w") as f:
            f.write("key: new_value")
            
        hash2 = get_file_sha256(file1)
        dir_hash2 = get_dir_sha256(tmpdir)
        
        assert hash1 != hash2
        assert dir_hash1 != dir_hash2


def test_dependency_caching_and_keys():
    # Verify we can compute rule key
    rule_key = get_rule_fst_key("diphthongization")
    assert len(rule_key) == 64
    
    # Check that metadata file exists
    meta_path = os.path.join(CACHE_DIR, f"{rule_key}.meta")
    assert os.path.exists(meta_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["name"] == "diphthongization"
    assert meta["kind"] == "Rule"
    assert "config_dependencies" in meta
    assert "child_fst_dependencies" in meta
    assert "symbol_table" in meta["child_fst_dependencies"]


def test_paradigm_recursive_keys():
    # Verify paradigm cache key resolves child marker and rule dependencies
    paradigm_key = get_paradigm_cache_key("verb_a_stem")
    assert len(paradigm_key) == 64
    
    meta_path = os.path.join(CACHE_DIR, f"{paradigm_key}.meta")
    assert os.path.exists(meta_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["name"] == "verb_a_stem"
    assert meta["kind"] == "Paradigm"
    assert len(meta["child_fst_dependencies"]) > 0
    # The child dependencies of the paradigm should contain markers
    has_marker = any(k.startswith("Marker/") for k in meta["child_fst_dependencies"].keys())
    assert has_marker


def test_open_parse_graph_and_subcomponent_caching():
    from parC.grammar.paradigm_compilation import get_open_parse_graph, get_paradigm_cache_key
    import pynini
    
    paradigm_name = "verb_a_stem"
    cache_key = get_paradigm_cache_key(paradigm_name)
    
    # Clean up any existing cached files for this key to ensure we test building and caching
    suffixes = ["_open_parse", "_open_inflect", "_open_input_acceptor", "_stages"]
    for suff in suffixes:
        for p in [os.path.join(CACHE_DIR, f"{cache_key}{suff}.fst"),
                  os.path.join(CACHE_DIR, f"{cache_key}{suff}.meta")]:
            if os.path.exists(p):
                os.remove(p)
        for i in range(20):
            p = os.path.join(CACHE_DIR, f"{cache_key}_stages_seq_{i}.fst")
            if os.path.exists(p):
                os.remove(p)
                
    # Build and cache
    g = get_open_parse_graph(paradigm_name)
    assert isinstance(g, pynini.Fst)
    
    # Assert cached files exist
    assert os.path.exists(os.path.join(CACHE_DIR, f"{cache_key}_open_parse.fst"))
    assert os.path.exists(os.path.join(CACHE_DIR, f"{cache_key}_open_inflect.fst"))
    assert os.path.exists(os.path.join(CACHE_DIR, f"{cache_key}_open_input_acceptor.fst"))
    assert os.path.exists(os.path.join(CACHE_DIR, f"{cache_key}_stages.meta"))
    assert os.path.exists(os.path.join(CACHE_DIR, f"{cache_key}_stages_seq_0.fst"))
    
    # Build again -> should HIT cache
    g2 = get_open_parse_graph(paradigm_name)
    assert isinstance(g2, pynini.Fst)
    
    # Verify build_inflect_graph_for_root_regex without cache_key hits cache
    from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
    g3 = build_inflect_graph_for_root_regex(paradigm_name, "<Phone>*")
    assert isinstance(g3, pynini.Fst)

    # Verify build_inflect_graph_for_root_regex with infer_lexical_features=True caches and hits
    infer_key = f"{cache_key}_infer"
    for p in [os.path.join(CACHE_DIR, f"{infer_key}_open_inflect.fst"),
              os.path.join(CACHE_DIR, f"{infer_key}_open_input_acceptor.fst")]:
        if os.path.exists(p):
            os.remove(p)
            
    g4 = build_inflect_graph_for_root_regex(paradigm_name, "<Phone>*", infer_lexical_features=True)
    assert isinstance(g4, pynini.Fst)
    assert os.path.exists(os.path.join(CACHE_DIR, f"{infer_key}_open_inflect.fst"))
    assert os.path.exists(os.path.join(CACHE_DIR, f"{infer_key}_open_input_acceptor.fst"))
    
    # Second call hits cache
    g5 = build_inflect_graph_for_root_regex(paradigm_name, "<Phone>*", infer_lexical_features=True)
    assert isinstance(g5, pynini.Fst)


def test_incremental_composition_caching():
    from parC.grammar.paradigm_compilation import get_open_parse_graph, get_paradigm_cache_key, clear_all_caches
    clear_all_caches()
    import glob
    
    paradigm_name = "verb_a_stem"
    cache_key = get_paradigm_cache_key(paradigm_name)
    
    # Clean up top-level caches to force compilation
    suffixes = ["_open_parse", "_open_inflect", "_open_input_acceptor", "_stages"]
    for suff in suffixes:
        for p in [os.path.join(CACHE_DIR, f"{cache_key}{suff}.fst"),
                  os.path.join(CACHE_DIR, f"{cache_key}{suff}.meta")]:
            if os.path.exists(p):
                os.remove(p)
    
    # Clear any composition cache files first
    for p in glob.glob(os.path.join(CACHE_DIR, "composition_*")):
        os.remove(p)
        
    # This compiles the stages and compositions
    get_open_parse_graph(paradigm_name)
    
    # Check that composition cache files were created
    comp_files = glob.glob(os.path.join(CACHE_DIR, "composition_*.fst"))
    assert len(comp_files) > 0
