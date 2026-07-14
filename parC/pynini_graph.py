import sys

_old_pynini = sys.modules.pop("pynini", None)
import pynini as _real_pynini

if _old_pynini is not None:
    sys.modules["pynini"] = _old_pynini
else:
    sys.modules["pynini"] = _real_pynini
import pynini
import hashlib
import os
import json
import time
import atexit
import threading
import concurrent.futures
from loguru import logger
from parC.constants import get_yaml_dir

_FST_MEM_CACHE = {}
_FST_OBJECT_HASH_CACHE = {}
_HASH_CACHE_LOCK = threading.Lock()
MAX_STATES_FOR_MEM_CACHE = 2000

# Thread-safe Cache Locks and Synchronization Map
_CACHE_LOCKS = {}
_LOCKS_LOCK = threading.Lock()
_MEM_CACHE_LOCK = threading.Lock()
_GRAPH_DEPS_LOCK = threading.Lock()


def get_node_lock(cache_key: str) -> threading.Lock:
    """Returns a thread lock specific to a cache key to prevent concurrent compilation/IO for the same node."""
    with _LOCKS_LOCK:
        if cache_key not in _CACHE_LOCKS:
            _CACHE_LOCKS[cache_key] = threading.Lock()
        return _CACHE_LOCKS[cache_key]


_GRAPH_DEPS = None
_GRAPH_DEPS_PATH = None


def _save_graph_deps_at_exit():
    global _GRAPH_DEPS, _GRAPH_DEPS_PATH
    if _GRAPH_DEPS is not None and _GRAPH_DEPS_PATH is not None:
        try:
            with open(_GRAPH_DEPS_PATH, "w") as f:
                json.dump(_GRAPH_DEPS, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write graph dependencies at exit: {e}")


def compile_graph_dynamic_dag(root: "GraphNode", max_workers: int = None) -> pynini.Fst:
    """
    Highly-optimal scheduler that compiles nodes as soon as their specific children are compiled.
    Avoids level-by-level synchronization barriers.
    """
    # 1. Collect all unique nodes in the DAG
    unique_nodes = {}
    node_counts = {}

    def collect(node: GraphNode):
        key = node.cache_key
        if key not in unique_nodes:
            unique_nodes[key] = node
            node_counts[key] = 1
            for child in node.children:
                collect(child)
        else:
            node_counts[key] += 1

    collect(root)
    # json.dump(node_counts, open(root.cache_key + "-counts.json", "w+"))
    for node, count in node_counts.items():
        unique_nodes[node].set_count(count)

    # 2. Build dependency counts and parent-notification map
    uncompiled_deps = {}  # cache_key -> count of uncompiled children
    parent_map = {}  # cache_key -> set of parent GraphNodes

    for key in unique_nodes:
        parent_map[key] = set()

    for key, node in unique_nodes.items():
        uncompiled_children = []
        for child in node.children:
            child_compiled = False
            if child.num_states is not None:
                child_compiled = True
            else:
                with _MEM_CACHE_LOCK:
                    if child.cache_key in _FST_MEM_CACHE:
                        child_compiled = True
            if (
                not child_compiled
                and child.name is not None
                and node_counts[child.cache_key] > 1
            ):
                uncompiled_children.append(child)

        uncompiled_deps[key] = len(uncompiled_children)

        for child in uncompiled_children:
            parent_map[child.cache_key].add(node)

    # Lock to coordinate atomic updates of uncompiled_deps counts
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        def on_node_compiled(completed_node: GraphNode):
            completed_key = completed_node.cache_key

            with lock:
                parents = parent_map.get(completed_key, [])
                for parent in parents:
                    parent_key = parent.cache_key
                    uncompiled_deps[parent_key] -= 1

                    # Parent has 0 remaining dependencies -> schedule it immediately
                    if uncompiled_deps[parent_key] == 0:
                        executor.submit(compile_task, parent)

        def compile_task(node: GraphNode):
            try:
                # Compile node (which is now thread-safe)
                node.compile()
                on_node_compiled(node)
            except Exception as e:
                logger.error(f"Parallel compilation failed for node {node}: {e}")
                raise e

        # Seed the thread pool with all nodes that have 0 initial uncompiled dependencies
        with lock:
            initial_nodes = []
            for key, node in unique_nodes.items():
                node_compiled = False
                if node.num_states is not None:
                    node_compiled = True
                else:
                    with _MEM_CACHE_LOCK:
                        if key in _FST_MEM_CACHE:
                            node_compiled = True
                if uncompiled_deps[key] == 0 and not node_compiled:
                    initial_nodes.append(node)

            for node in initial_nodes:
                executor.submit(compile_task, node)

    return root.compile()


atexit.register(_save_graph_deps_at_exit)


class GraphNode:
    """
    A node in the computational graph representing a delayed Pynini FST operation.
    """

    def __init__(
        self,
        op: str,
        children: list,
        params: dict = None,
        config_dep: dict = None,
        name: str = None,
        compiled_fst: _real_pynini.Fst = None,
    ):
        self.op = op
        self.children = [
            c if isinstance(c, GraphNode) else GraphNode.constant(c) for c in children
        ]
        self.params = params or {}
        self.config_dep = config_dep or {}
        # self._compiled_fst = None
        self.name = name
        self.count = 0
        self.num_states = None
        self.compiled_fst = compiled_fst

        # Calculate SHA-256 of the node's op, parameters, config dependencies, and children's keys
        hasher = hashlib.sha256()
        hasher.update(self.op.encode("utf-8"))

        # Add parameter representation
        for k, v in sorted(self.params.items()):
            hasher.update(f"{k}:{v}".encode("utf-8"))

        # Add config dependencies
        for k, v in sorted(self.config_dep.items()):
            hasher.update(f"config:{k}:{v}".encode("utf-8"))

        # Add children cache keys
        for child in self.children:
            hasher.update(child.cache_key.encode("utf-8"))

        self.cache_key = hasher.hexdigest()

    def set_num_states(self, num_states: int):
        self.num_states = num_states

    def __getattr__(self, name):
        # Delegate any unhandled attribute/method to the compiled FST
        # if self._compiled_fst is None:
        # self.compile()
        # return getattr(self._compiled_fst, name)
        raise AttributeError(
            f"No property {name} on GraphNode - did you mean to compile?"
        )

    @classmethod
    def constant(cls, val):
        """
        Creates a constant leaf node from an existing pynini.Fst or string.
        """
        if isinstance(val, GraphNode):
            return val
        if isinstance(val, _real_pynini.Fst):
            # Use SHA-256 hash of the compiled FST bytes to identify it uniquely and stably
            fst_id = id(val)
            with _HASH_CACHE_LOCK:
                if fst_id in _FST_OBJECT_HASH_CACHE:
                    fst_sha = _FST_OBJECT_HASH_CACHE[fst_id]
                else:
                    fst_bytes = val.write_to_string()
                    fst_sha = hashlib.sha256(fst_bytes).hexdigest()
                    _FST_OBJECT_HASH_CACHE[fst_id] = fst_sha
            node = GraphNode(
                op="constant_fst",
                children=[],
                params={"fst_sha": fst_sha},
                compiled_fst=val,
            )
            return node
        return GraphNode(op="constant_val", children=[], params={"val": val})

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
            "children": [child.cache_key for child in self.children],
        }
        for child in self.children:
            child.serialize_subgraph(visited)
        return visited

    def compile(self, parallel: bool = False, max_workers: int = None) -> pynini.Fst:
        if parallel:
            return compile_graph_dynamic_dag(self, max_workers=max_workers)

        if self.compiled_fst is not None:
            return self.compiled_fst

        cache_key = self.cache_key
        with _MEM_CACHE_LOCK:
            if cache_key in _FST_MEM_CACHE:
                fst = _FST_MEM_CACHE[cache_key]
                self.set_num_states(fst.num_states())
                return _FST_MEM_CACHE[cache_key]

        node_lock = get_node_lock(cache_key)
        with node_lock:
            if self.compiled_fst is not None:
                return self.compiled_fst

            with _MEM_CACHE_LOCK:
                if cache_key in _FST_MEM_CACHE:
                    fst = _FST_MEM_CACHE[cache_key]
                    self.set_num_states(fst.num_states())
                    return fst

            yaml_dir = get_yaml_dir()
            cache_dir = os.path.join(yaml_dir, ".cache")
            cache_to_disk = self.name or self.count > 1

            if cache_to_disk:
                os.makedirs(cache_dir, exist_ok=True)

                cache_path = os.path.join(cache_dir, f"{cache_key}.fst")

                # 1. Try reading from cache
                if os.path.exists(cache_path):
                    start_time = time.perf_counter()
                    try:
                        compiled_fst = _real_pynini.Fst.read(cache_path)
                        duration = time.perf_counter() - start_time
                        logger.debug(
                            f"FST Cache Hit: {self} loaded in {duration:.4f}s (key: {cache_key})"
                        )
                        with _MEM_CACHE_LOCK:
                            if compiled_fst.num_states() <= MAX_STATES_FOR_MEM_CACHE:
                                _FST_MEM_CACHE[cache_key] = compiled_fst
                        self.set_num_states(compiled_fst.num_states())
                        return compiled_fst
                    except Exception as e:
                        logger.warning(
                            f"Failed to read cached FST {cache_path}: {e}. Recompiling..."
                        )

            # 2. Compile children
            compiled_children = [child.compile() for child in self.children]

            # 3. Perform operation & measure timing
            start_time = time.perf_counter()
            fst = self._execute_op(compiled_children)
            duration = time.perf_counter() - start_time
            logger.debug(f"FST Compiled: {self} in {duration:.4f}s (key: {cache_key})")

            if cache_to_disk:
                # 4. Save to disk cache
                try:
                    fst.write(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to cache FST to {cache_path}: {e}")

            # 5. Save graph dependencies metadata
            try:
                global _GRAPH_DEPS, _GRAPH_DEPS_PATH
                dep_path = os.path.join(cache_dir, "graph_dependencies.json")
                with _GRAPH_DEPS_LOCK:
                    _GRAPH_DEPS_PATH = dep_path
                    if _GRAPH_DEPS is None:
                        _GRAPH_DEPS = {}
                        if os.path.exists(dep_path):
                            with open(dep_path, "r") as f:
                                _GRAPH_DEPS = json.load(f)
                    self.serialize_subgraph(_GRAPH_DEPS)
            except Exception as e:
                logger.warning(f"Failed to update graph dependencies: {e}")

            self.num_states = fst.num_states()
            # self._compiled_fst = fst
            with _MEM_CACHE_LOCK:
                if fst.num_states() <= MAX_STATES_FOR_MEM_CACHE:
                    _FST_MEM_CACHE[cache_key] = fst
            return fst

    def _execute_op(self, compiled_children: list) -> pynini.Fst:
        op = self.op
        for i, child in enumerate(compiled_children):
            if isinstance(child, GraphNode):
                logger.error(
                    f"CRITICAL TYPE ERROR: self.op={op}, child[{i}] is a GraphNode (op={child.op}, children={len(child.children)}), self.children[{i}] op={self.children[i].op if i < len(self.children) else 'N/A'}"
                )
        if op == "constant_val":
            val = self.params["val"]
            if isinstance(val, str):
                return _real_pynini.accep(val)
            return val
        elif op == "constant_fst":
            # Real raw FST constant should be preserved in compilation, let's look up by parameter or raise
            raise RuntimeError(
                "Raw FST constant node compiled without preloaded FST. Use custom GraphNodes where possible."
            )
        elif op == "accep":
            token_type = self.params.get("token_type")
            if isinstance(token_type, str) and (
                token_type.startswith("<SymbolTable")
                or token_type.startswith("<pynini.SymbolTable")
                or token_type.startswith("SymbolTable:")
            ):
                from parC.grammar.acceptor_compilation import get_symbol_table

                token_type = get_symbol_table()
            kwargs = {}
            if "weight" in self.params:
                kwargs["weight"] = self.params["weight"]
            for k, v in self.params.items():
                if k not in ("string", "token_type", "weight"):
                    kwargs[k] = v
            return _real_pynini.accep(
                self.params["string"], token_type=token_type, **kwargs
            )
        elif op == "concat":
            # For a binary tree pairwise concat node
            return _real_pynini.concat(compiled_children[0], compiled_children[1])
        elif op == "union":
            # For a binary tree pairwise union node
            return _real_pynini.union(compiled_children[0], compiled_children[1])
        elif op == "intersect":
            return _real_pynini.intersect(compiled_children[0], compiled_children[1])
        elif op == "difference":
            return _real_pynini.difference(compiled_children[0], compiled_children[1])
        elif op == "compose":
            return _real_pynini.compose(compiled_children[0], compiled_children[1])
        elif op == "cdrewrite":
            tau = compiled_children[0]
            lambda_fsa = compiled_children[1]
            rho_fsa = compiled_children[2]
            sigma_fsa = compiled_children[3]
            return _real_pynini.cdrewrite(tau, lambda_fsa, rho_fsa, sigma_fsa)
        elif op == "cross":
            return _real_pynini.cross(compiled_children[0], compiled_children[1])
        elif op == "project":
            return _real_pynini.project(
                compiled_children[0], project_type=self.params["project_type"]
            )
        elif op == "invert":
            return _real_pynini.invert(compiled_children[0])
        elif op == "determinize":
            return _real_pynini.determinize(compiled_children[0])
        elif op == "minimize":
            return _real_pynini.minimize(compiled_children[0])
        elif op == "optimize":
            return _real_pynini.optimize(compiled_children[0])
        elif op == "star":
            return _real_pynini.closure(compiled_children[0])
        elif op == "pynutil_delete":
            return _real_pynini.cross(compiled_children[0], "")
        elif op == "pynutil_insert":
            return _real_pynini.cross("", compiled_children[0])
        elif op == "empty_fst":
            return _real_pynini.Fst()
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
        name_part = f", name={self.name}" if self.name is not None else ""
        return f"GraphNode(op={self.op}, params={self.params}{name_part})"

    def __repr__(self):
        return self.__str__()

    def set_name(self, name: str) -> "GraphNode":
        self.name = name
        return self

    def set_count(self, count: int):
        self.count = count

    def optimize(self):
        return GraphNode(op="optimize", children=[self])

    def invert(self):
        return invert(self)

    def determinize(self):
        return determinize(self)

    def minimize(self):
        return minimize(self)

    def project(self, project_type: str):
        return project(self, project_type)

    def copy(self):
        return self

    def write(self, path: str):
        compiled = self.compile()
        compiled.write(path)

    @property
    def star(self):
        return GraphNode(op="star", children=[self])


# Drop-in Wrapper Functions


def accep(
    string: str, token_type=None, weight=None, config_dep=None, **kwargs
) -> GraphNode:
    # Use id of SymbolTable or serialize it if possible for params
    if token_type is not None:
        if isinstance(token_type, _real_pynini.SymbolTable) or hasattr(
            token_type, "write_to_string"
        ):
            token_repr = (
                "SymbolTable:"
                + hashlib.sha256(token_type.write_to_string()).hexdigest()
            )
        else:
            token_repr = str(token_type)
    else:
        token_repr = None
    params = {"string": string, "token_type": token_repr}
    if weight is not None:
        params["weight"] = weight
    params.update(kwargs)
    return GraphNode(op="accep", children=[], params=params, config_dep=config_dep)


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
    return GraphNode(
        op="project", children=[fst], params={"project_type": project_type}
    )


def invert(fst) -> GraphNode:
    return GraphNode(op="invert", children=[fst])


def determinize(fst) -> GraphNode:
    return GraphNode(op="determinize", children=[fst])


def minimize(fst) -> GraphNode:
    return GraphNode(op="minimize", children=[fst])


def optimize(fst) -> GraphNode:
    return GraphNode(op="optimize", children=[fst])


class FstMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, GraphNode) or isinstance(instance, _real_pynini.Fst)


class Fst(GraphNode, metaclass=FstMeta):
    def __init__(self):
        super().__init__(op="empty_fst", children=[], params={})
        self._compiled_fst = _real_pynini.Fst()

    @classmethod
    def read(cls, path: str):
        fst = _real_pynini.Fst.read(path)
        return GraphNode.constant(fst)


class PynutilDelayed:
    def delete(self, fst) -> GraphNode:
        return GraphNode(op="pynutil_delete", children=[fst])

    def insert(self, fst) -> GraphNode:
        return GraphNode(op="pynutil_insert", children=[fst])


pynutil = PynutilDelayed()


# Binary-Tree Pairwise Folding Helper


def _fold_balanced_binary_tree(nodes: list, op_fn) -> GraphNode:
    if not nodes:
        raise ValueError("Cannot fold an empty list of nodes")
    if len(nodes) == 1:
        return (
            nodes[0]
            if isinstance(nodes[0], GraphNode)
            else GraphNode.constant(nodes[0])
        )

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
    return _fold_balanced_binary_tree(
        nodes, lambda a, b: GraphNode(op="union", children=[a, b])
    )


def concat(*nodes) -> GraphNode:
    if not nodes:
        raise ValueError("concat requires at least one node")
    if len(nodes) == 1 and isinstance(nodes[0], (list, set, tuple)):
        nodes = list(nodes[0])

    # Fold using pairwise concat
    return _fold_balanced_binary_tree(
        nodes, lambda a, b: GraphNode(op="concat", children=[a, b])
    )


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
        param_str += "<br>config: " + ", ".join(
            f"{k}={v}" for k, v in node.config_dep.items()
        )

    if node.name:
        label = f"<b>{node.name}</b><br><font size=2>{node.op.upper()}</font>"
    else:
        label = f"<b>{node.op.upper()}</b>"

    state_text = (
        f"states: {node.num_states}"
        if node.num_states is not None
        else "pending compile"
    )
    label += f"<br><font size=2 color=#94a3b8>{state_text}</font>"

    if param_str:
        label += f"<br><font size=2 color=#64748b>{param_str}</font>"

    lines = [f'    {key}["{label}"]']
    for child in node.children:
        lines.append(f"    {child.cache_key} --> {key}")
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
    logger.info(
        f"Computational graph visualization successfully exported to {output_path}"
    )


# Dynamically expose all non-overridden symbols from real pynini for perfect backward compatibility
import sys

_current_module = sys.modules[__name__]
for _name in dir(_real_pynini):
    if not hasattr(_current_module, _name):
        setattr(_current_module, _name, getattr(_real_pynini, _name))
