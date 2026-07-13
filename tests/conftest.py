import os
import sys
import pytest
from parC import pynini_graph
sys.modules["pynini"] = pynini_graph

@pytest.fixture(autouse=True)
def preserve_yaml_dir():
    # Save current env state
    original_env = dict(os.environ)
    
    # We can also explicitly default to "yaml/spanish-example" if YAML_DIR is not present
    if "YAML_DIR" not in os.environ:
        os.environ["YAML_DIR"] = "yaml/spanish-example"
        
    try:
        from parC.grammar.marker_resolution import _markers_for_paradigm_cache
        _markers_for_paradigm_cache.clear()
    except Exception:
        pass

    # Clear disk cache
    try:
        from parC.constants import get_yaml_dir
        import shutil
        cache_dir = os.path.join(get_yaml_dir(), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    yield
    
    # Restore env state completely
    for key in list(os.environ.keys()):
        if key not in original_env:
            del os.environ[key]
    for key, val in original_env.items():
        os.environ[key] = val

    try:
        from parC.grammar.marker_resolution import _markers_for_paradigm_cache
        _markers_for_paradigm_cache.clear()
    except Exception:
        pass

    try:
        from parC.constants import get_yaml_dir
        import shutil
        cache_dir = os.path.join(get_yaml_dir(), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
