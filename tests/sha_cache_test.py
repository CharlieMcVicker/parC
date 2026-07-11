import os
import tempfile
import json
from parC.yaml_utils.cache import (
    get_file_sha256,
    get_dir_sha256,
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
