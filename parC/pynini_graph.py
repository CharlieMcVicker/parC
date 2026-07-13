import hashlib
import os
import json
import time
import pynini
from loguru import logger
from parC.constants import get_yaml_dir

class GraphNode:
    """
    A node in the computational graph representing a delayed Pynini FST operation.
    """
    def __init__(self, op: str, children: list, params: dict = None, config_dep: dict = None):
        self.op = op
        self.children = [c if isinstance(c, GraphNode) else GraphNode.constant(c) for c in children]
        self.params = params or {}
        self.config_dep = config_dep or {}
        self._compiled_fst = None
        self._cache_key = None

    @classmethod
    def constant(cls, val):
        """
        Creates a constant leaf node from an existing pynini.Fst or string.
        """
        if isinstance(val, GraphNode):
            return val
        if isinstance(val, pynini.Fst):
            # For a raw FST constant, we use its pointer/ID or string representation to identify it uniquely
            return GraphNode(op="constant_fst", children=[], params={"fst_id": id(val)})
        return GraphNode(op="constant_val", children=[], params={"val": val})

    @property
    def cache_key(self) -> str:
        if self._cache_key is not None:
            return self._cache_key

        # Calculate SHA-256 of the node's op, parameters, config dependencies, and children's keys
        hasher = hashlib.sha256()
        hasher.update(self.op.encode("utf-8"))
        
        # Add parameter representation
        for k, v in sorted(self.params.items()):
            hasher.update(f"{k}:{v}".encode("utf-8"))
            
        # Add config dependencies
        for k, v in sorted(self.config_dep.items()):
            hasher.update(f"config:{k}:{v}".encode("utf-8"))
            
        # Recurse on children cache keys
        for child in self.children:
            hasher.update(child.cache_key.encode("utf-8"))
            
        self._cache_key = hasher.hexdigest()
        return self._cache_key

    def serialize_subgraph(self, visited: dict = None) -> dict:
        """
        Serializes the sub-DAG under this node into a dictionary.
        """
        if visited is None:
            visited = {}
        key = self.cache_key
        if key in visited:
            return visited

        visited[key] = {
            "op": self.op,
            "params": self.params,
            "config_dep": self.config_dep,
            "children": [child.cache_key for child in self.children]
        }
        for child in self.children:
            child.serialize_subgraph(visited)
        return visited

    def compile(self) -> pynini.Fst:
        if self._compiled_fst is not None:
            return self._compiled_fst

        yaml_dir = get_yaml_dir()
        cache_dir = os.path.join(yaml_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = self.cache_key
        cache_path = os.path.join(cache_dir, f"{cache_key}.fst")

        # 1. Try reading from cache
        if os.path.exists(cache_path):
            start_time = time.perf_counter()
            try:
                self._compiled_fst = pynini.Fst.read(cache_path)
                duration = time.perf_counter() - start_time
                logger.debug(f"FST Cache Hit: {self} loaded in {duration:.4f}s (key: {cache_key})")
                return self._compiled_fst
            except Exception as e:
                logger.warning(f"Failed to read cached FST {cache_path}: {e}. Recompiling...")

        # 2. Compile children
        compiled_children = [child.compile() for child in self.children]

        # 3. Perform operation & measure timing
        start_time = time.perf_counter()
        fst = self._execute_op(compiled_children)
        duration = time.perf_counter() - start_time
        logger.debug(f"FST Compiled: {self} in {duration:.4f}s (key: {cache_key})")

        # 4. Save to disk cache
        try:
            fst.write(cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache FST to {cache_path}: {e}")

        # 5. Save graph dependencies metadata
        try:
            dep_path = os.path.join(cache_dir, "graph_dependencies.json")
            existing_deps = {}
            if os.path.exists(dep_path):
                with open(dep_path, "r") as f:
                    existing_deps = json.load(f)
            self.serialize_subgraph(existing_deps)
            with open(dep_path, "w") as f:
                json.dump(existing_deps, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write graph dependencies to {dep_path}: {e}")

        self._compiled_fst = fst
        return fst

    def _execute_op(self, compiled_children: list[pynini.Fst]) -> pynini.Fst:
        op = self.op
        if op == "constant_val":
            val = self.params["val"]
            if isinstance(val, str):
                return pynini.accep(val)
            return val
        elif op == "constant_fst":
            # Real raw FST constant should be preserved in compilation, let's look up by parameter or raise
            raise RuntimeError("Raw FST constant node compiled without preloaded FST. Use custom GraphNodes where possible.")
        elif op == "accep":
            return pynini.accep(self.params["string"], token_type=self.params.get("token_type"))
        elif op == "concat":
            # For a binary tree pairwise concat node
            return pynini.concat(compiled_children[0], compiled_children[1])
        elif op == "union":
            # For a binary tree pairwise union node
            return pynini.union(compiled_children[0], compiled_children[1])
        elif op == "intersect":
            return pynini.intersect(compiled_children[0], compiled_children[1])
        elif op == "difference":
            return pynini.difference(compiled_children[0], compiled_children[1])
        elif op == "compose":
            return pynini.compose(compiled_children[0], compiled_children[1])
        elif op == "cdrewrite":
            tau = compiled_children[0]
            lambda_fsa = compiled_children[1]
            rho_fsa = compiled_children[2]
            sigma_fsa = compiled_children[3]
            return pynini.cdrewrite(tau, lambda_fsa, rho_fsa, sigma_fsa)
        elif op == "cross":
            return pynini.cross(compiled_children[0], compiled_children[1])
        elif op == "project":
            return pynini.project(compiled_children[0], project_type=self.params["project_type"])
        elif op == "invert":
            return pynini.invert(compiled_children[0])
        else:
            raise NotImplementedError(f"Unsupported operation: {op}")

    # Operator Overloads
    def __add__(self, other):
        return concat(self, other)

    def __radd__(self, other):
        return concat(other, self)

    def __matmul__(self, other):
        return compose(self, other)

    def __rmatmul__(self, other):
        return compose(other, self)

    def __or__(self, other):
        return union(self, other)

    def __ror__(self, other):
        return union(other, self)

    def __str__(self):
        return f"GraphNode(op={self.op}, params={self.params})"

    def __repr__(self):
        return self.__str__()


# Drop-in Wrapper Functions

def accep(string: str, token_type=None, config_dep=None) -> GraphNode:
    # Use id of SymbolTable or serialize it if possible for params
    token_repr = str(token_type) if token_type is not None else None
    return GraphNode(
        op="accep",
        children=[],
        params={"string": string, "token_type": token_repr},
        config_dep=config_dep
    )

def compose(fst1, fst2) -> GraphNode:
    return GraphNode(op="compose", children=[fst1, fst2])

def intersect(fst1, fst2) -> GraphNode:
    return GraphNode(op="intersect", children=[fst1, fst2])

def difference(fst1, fst2) -> GraphNode:
    return GraphNode(op="difference", children=[fst1, fst2])

def cdrewrite(tau, lambda_fsa, rho_fsa, sigma_fsa) -> GraphNode:
    return GraphNode(op="cdrewrite", children=[tau, lambda_fsa, rho_fsa, sigma_fsa])

def cross(fst1, fst2) -> GraphNode:
    return GraphNode(op="cross", children=[fst1, fst2])

def project(fst, project_type: str) -> GraphNode:
    return GraphNode(op="project", children=[fst], params={"project_type": project_type})

def invert(fst) -> GraphNode:
    return GraphNode(op="invert", children=[fst])


# Binary-Tree Pairwise Folding Helper

def _fold_balanced_binary_tree(nodes: list, op_fn) -> GraphNode:
    if not nodes:
        raise ValueError("Cannot fold an empty list of nodes")
    if len(nodes) == 1:
        return nodes[0] if isinstance(nodes[0], GraphNode) else GraphNode.constant(nodes[0])
    
    mid = len(nodes) // 2
    left = _fold_balanced_binary_tree(nodes[:mid], op_fn)
    right = _fold_balanced_binary_tree(nodes[mid:], op_fn)
    return op_fn(left, right)


def union(*nodes) -> GraphNode:
    if not nodes:
        raise ValueError("union requires at least one node")
    # Resolve any list arguments passed as a single tuple
    if len(nodes) == 1 and isinstance(nodes[0], (list, set, tuple)):
        nodes = list(nodes[0])
    
    # Fold using pairwise union
    return _fold_balanced_binary_tree(nodes, lambda a, b: GraphNode(op="union", children=[a, b]))


def concat(*nodes) -> GraphNode:
    if not nodes:
        raise ValueError("concat requires at least one node")
    if len(nodes) == 1 and isinstance(nodes[0], (list, set, tuple)):
        nodes = list(nodes[0])
        
    # Fold using pairwise concat
    return _fold_balanced_binary_tree(nodes, lambda a, b: GraphNode(op="concat", children=[a, b]))


# Visualization Layer

def _generate_mermaid_code(node: GraphNode, visited=None) -> str:
    if visited is None:
        visited = set()
    key = node.cache_key
    if key in visited:
        return ""
    visited.add(key)
    
    # Clean label and params to prevent breaking Mermaid syntax
    param_str = "<br>".join(f"{k}: {v}" for k, v in node.params.items())
    if node.config_dep:
        param_str += "<br>config: " + ", ".join(f"{k}={v}" for k, v in node.config_dep.items())
    
    label = f"<b>{node.op.upper()}</b>"
    if param_str:
        label += f"<br><font size=2 color=#94a3b8>{param_str}</font>"
        
    lines = [f'    {key}["{label}"]']
    for child in node.children:
        lines.append(f'    {child.cache_key} --> {key}')
        lines.append(_generate_mermaid_code(child, visited))
        
    return "\n".join(filter(None, lines))


def export_visualization(node: GraphNode, output_path: str = None):
    if output_path is None:
        yaml_dir = get_yaml_dir()
        output_path = os.path.join(yaml_dir, "fst_graph.html")
        
    mermaid_flowchart = _generate_mermaid_code(node)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynini FST Computational Graph</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d0f18;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-color: #5154ff;
            --text-color: #e2e8f0;
            --text-muted: #94a3b8;
        }}
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        header {{
            margin-top: 40px;
            margin-bottom: 20px;
            text-align: center;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0 0 10px 0;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            margin: 0;
        }}
        .container {{
            width: 90%;
            max-width: 1400px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 50px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .mermaid {{
            background: transparent;
            width: 100%;
            display: flex;
            justify-content: center;
        }}
    </style>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</head>
<body>
    <header>
        <h1>Pynini FST Computational Graph</h1>
        <p>Interactive Dependency DAG & Compilation Pipeline</p>
    </header>
    <div class="container">
        <pre class="mermaid">
graph TD
{mermaid_flowchart}
        </pre>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info(f"Computational graph visualization successfully exported to {output_path}")
