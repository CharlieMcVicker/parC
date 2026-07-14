import os
import shutil
import pytest
import pynini
from parC import pynini_graph
from parC.constants import get_yaml_dir


def test_pynini_graph_basic_compilation(tmp_path):
    # Set a temporary yaml dir to isolate the cache
    os.environ["YAML_DIR"] = str(tmp_path)

    # 1. Create nodes
    node_a = pynini_graph.accep("a")
    node_b = pynini_graph.accep("b")

    # 2. Binary-tree union optimization
    union_node = pynini_graph.union(node_a, node_b)

    # Check that union compiled correctly
    fst = union_node.compile()
    assert isinstance(fst, pynini.Fst)

    # Test that is_acceptor compiles on demand
    from parC.fst_utils import is_acceptor

    assert bool(is_acceptor(union_node)) is True


def test_pynini_graph_binary_tree_folding(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)

    nodes = [pynini_graph.accep(char) for char in ["a", "b", "c", "d"]]
    union_node = pynini_graph.union(nodes)

    # Root should be a pairwise union
    assert union_node.op == "union"
    assert len(union_node.children) == 2

    left_child = union_node.children[0]
    right_child = union_node.children[1]

    assert left_child.op == "union"
    assert right_child.op == "union"

    fst = union_node.compile()
    assert isinstance(fst, pynini.Fst)


def test_pynini_graph_caching(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)

    node_a = pynini_graph.accep("a", config_dep={"test_config": "v1"}).set_name(
        "node_a"
    )
    fst1 = node_a.compile()

    # Ensure cache file exists
    cache_dir = os.path.join(str(tmp_path), ".cache")
    cache_file = os.path.join(cache_dir, f"{node_a.cache_key}.fst")
    assert os.path.exists(cache_file)

    # Load again (should be a cache hit)
    node_a_again = pynini_graph.accep("a", config_dep={"test_config": "v1"}).set_name(
        "node_a"
    )
    fst2 = node_a_again.compile()
    assert fst2 is not None

    # Change config dependency (should be a cache miss)
    node_a_changed = pynini_graph.accep("a", config_dep={"test_config": "v2"}).set_name(
        "node_a"
    )
    assert node_a_changed.cache_key != node_a.cache_key


def test_pynini_graph_visualization(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)

    node_a = pynini_graph.accep("a")
    node_b = pynini_graph.accep("b")
    composed = pynini_graph.compose(node_a, node_b)

    viz_path = os.path.join(str(tmp_path), "test_graph.html")
    pynini_graph.export_visualization(composed, output_path=viz_path)

    assert os.path.exists(viz_path)
    with open(viz_path, "r") as f:
        content = f.read()
        assert "graph TD" in content
        assert "CONSTANT_VAL" in content or "ACCEP" in content


def test_parallel_graph_compilation(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)

    # Build a moderately complex graph
    node_a = pynini_graph.accep("a")
    node_b = pynini_graph.accep("b")
    node_c = pynini_graph.accep("c")
    node_d = pynini_graph.accep("d")

    u1 = pynini_graph.union(node_a, node_b)
    u2 = pynini_graph.union(node_c, node_d)
    root = pynini_graph.concat(u1, u2)

    # Compile via parallel dynamic DAG scheduler
    parallel_fst = root.compile(parallel=True, max_workers=4)
    assert parallel_fst is not None

    # Compile via sequential compile (should hit cache/already compiled)
    seq_fst = root.compile()
    assert str(parallel_fst) == str(seq_fst)


def test_concurrent_compilation_lock_behavior(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)

    import threading
    import time

    node = pynini_graph.accep("concurrent_test")

    results = []

    def worker():
        fst = node.compile()
        results.append(fst)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should have successfully retrieved the compiled FST
    assert len(results) == 5
    for r in results:
        assert r is not None
        assert str(r) == str(results[0])


def test_pynini_graph_constant_address_caching():
    import pynini

    # Create an FST
    fst = pynini.accep("test_constant_cache").compile()

    # Verify it is not in the cache yet
    fst_id = id(fst)
    if fst_id in pynini_graph._FST_OBJECT_HASH_CACHE:
        del pynini_graph._FST_OBJECT_HASH_CACHE[fst_id]

    # Call constant to populate the cache
    node1 = pynini_graph.GraphNode.constant(fst)
    assert fst_id in pynini_graph._FST_OBJECT_HASH_CACHE
    cached_sha = pynini_graph._FST_OBJECT_HASH_CACHE[fst_id]

    # Overwrite the cache value to verify it retrieves from cache without re-serializing
    pynini_graph._FST_OBJECT_HASH_CACHE[fst_id] = "dummy_sha_value"

    node2 = pynini_graph.GraphNode.constant(fst)
    assert node2.params["fst_sha"] == "dummy_sha_value"

    # Restore actual cache
    pynini_graph._FST_OBJECT_HASH_CACHE[fst_id] = cached_sha
