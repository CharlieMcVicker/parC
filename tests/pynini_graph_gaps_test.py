import os
import sys
import pytest
from parC import pynini_graph
pynini = pynini_graph._real_pynini

_old = sys.modules.get("pynini")
sys.modules["pynini"] = pynini
try:
    sys.modules.pop("pynini.lib.pynutil", None)
    from pynini.lib import pynutil as real_pynutil
finally:
    sys.modules["pynini"] = _old

def test_graph_node_optimize(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    node = pynini_graph.accep("hello")
    opt_node = node.optimize()
    
    assert opt_node.op == "optimize"
    assert opt_node.children[0] is node
    
    compiled = opt_node.compile()
    expected = pynini.optimize(pynini.accep("hello"))
    assert str(compiled) == str(expected)

def test_graph_node_invert(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    # Create a transducer mapping "a" to "b"
    node = pynini_graph.cross("a", "b")
    inv_node = node.invert()
    
    assert inv_node.op == "invert"
    assert inv_node.children[0] is node
    
    compiled = inv_node.compile()
    expected = pynini.invert(pynini.cross("a", "b"))
    assert str(compiled) == str(expected)

def test_graph_node_project(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    # Create a transducer mapping "a" to "b"
    node = pynini_graph.cross("a", "b")
    proj_input = node.project("input")
    proj_output = node.project("output")
    
    assert proj_input.op == "project"
    assert proj_input.params["project_type"] == "input"
    assert proj_output.op == "project"
    assert proj_output.params["project_type"] == "output"
    
    compiled_in = proj_input.compile()
    compiled_out = proj_output.compile()
    
    expected_in = pynini.project(pynini.cross("a", "b"), project_type="input")
    expected_out = pynini.project(pynini.cross("a", "b"), project_type="output")
    
    assert str(compiled_in) == str(expected_in)
    assert str(compiled_out) == str(expected_out)

def test_graph_node_copy():
    node = pynini_graph.accep("a")
    assert node.copy() is node

def test_graph_node_star(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    node = pynini_graph.accep("a")
    star_node = node.star
    
    assert star_node.op == "star"
    assert star_node.children[0] is node
    
    compiled = star_node.compile()
    expected = pynini.closure(pynini.accep("a"))
    assert str(compiled) == str(expected)

def test_pynini_graph_fst_class(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    fst_node = pynini_graph.Fst()
    assert isinstance(fst_node, pynini_graph.GraphNode)
    assert fst_node.op == "empty_fst"
    
    compiled = fst_node.compile()
    assert isinstance(compiled, pynini.Fst)
    assert str(compiled) == str(pynini.Fst())

def test_pynutil_delayed(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    # Test delete
    del_node = pynini_graph.pynutil.delete("a")
    assert del_node.op == "pynutil_delete"
    
    compiled_del = del_node.compile()
    expected_del = real_pynutil.delete(pynini.accep("a"))
    assert str(compiled_del) == str(expected_del)
    
    # Test insert
    ins_node = pynini_graph.pynutil.insert("b")
    assert ins_node.op == "pynutil_insert"
    
    compiled_ins = ins_node.compile()
    expected_ins = real_pynutil.insert(pynini.accep("b"))
    assert str(compiled_ins) == str(expected_ins)


def test_graph_node_names_and_state_tracking(tmp_path):
    os.environ["YAML_DIR"] = str(tmp_path)
    
    # 1. Test constructor with name
    node1 = pynini_graph.GraphNode(op="accep", children=[], params={"string": "abc"}, name="MyNode")
    assert node1.name == "MyNode"
    assert "name=MyNode" in str(node1)
    assert "name=MyNode" in repr(node1)
    
    # 2. Test builder pattern set_name
    node2 = pynini_graph.GraphNode(op="accep", children=[], params={"string": "def"})
    assert node2.name is None
    assert "name=" not in str(node2)
    
    node2_chained = node2.set_name("ChainedNode")
    assert node2_chained is node2
    assert node2.name == "ChainedNode"
    assert "name=ChainedNode" in str(node2)
    
    # 3. Test tracking of num_states upon compilation
    node3 = pynini_graph.accep("hello")
    assert node3._compiled_fst is None
    
    node3.compile()
    assert node3._compiled_fst is not None
    assert isinstance(node3.num_states(), int)
    assert node3.num_states() > 0
    
    # 4. Test visualization code contains the custom name and states
    mermaid_code = pynini_graph._generate_mermaid_code(node1)
    # node1 has not been compiled yet, so it should say "pending compile"
    assert "MyNode" in mermaid_code
    assert "pending compile" in mermaid_code
    
    node1.compile()
    mermaid_code_compiled = pynini_graph._generate_mermaid_code(node1)
    assert f"states: {node1.num_states()}" in mermaid_code_compiled

