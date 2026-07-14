# parallel_compile_prototype.py
# Reference prototypes for Task Implementation: Parallel FST Graph Compilation with Locks.

import threading
import concurrent.futures
import time
from loguru import logger
import pynini
from parC.pynini_graph import GraphNode, _FST_MEM_CACHE

# Thread-safe Cache Locks and Synchronization Map
_CACHE_LOCKS = {}
_LOCKS_LOCK = threading.Lock()
_MEM_CACHE_LOCK = threading.Lock()

def get_node_lock(cache_key: str) -> threading.Lock:
    """Returns a thread lock specific to a cache key to prevent concurrent compilation/IO for the same node."""
    with _LOCKS_LOCK:
        if cache_key not in _CACHE_LOCKS:
            _CACHE_LOCKS[cache_key] = threading.Lock()
        return _CACHE_LOCKS[cache_key]


# =====================================================================
# PROTOTYPE 1: LEVEL-BY-LEVEL (ROW-BY-ROW) PARALLEL SCHEDULE
# =====================================================================

def get_nodes_by_level(root: GraphNode) -> list[list[GraphNode]]:
    """
    Performs a post-order traversal to assign heights to all unique nodes in the DAG.
    Returns a list of lists, where levels[0] are leaves, levels[1] are their parents, etc.
    """
    node_heights = {}  # cache_key -> height
    unique_nodes = {}  # cache_key -> GraphNode

    def compute_height(node: GraphNode) -> int:
        key = node.cache_key
        if key in node_heights:
            return node_heights[key]
        
        unique_nodes[key] = node
        if not node.children:
            height = 0
        else:
            height = 1 + max(compute_height(child) for child in node.children)
            
        node_heights[key] = height
        return height

    compute_height(root)
    
    # Group unique nodes by height
    if not node_heights:
        return []
    max_height = max(node_heights.values())
    levels = [[] for _ in range(max_height + 1)]
    for key, height in node_heights.items():
        levels[height].append(unique_nodes[key])
        
    return levels


def compile_graph_parallel_rows(root: GraphNode, max_workers: int = None) -> pynini.Fst:
    """
    Groups unique nodes by levels and dispatches them in parallel row-by-row.
    Requires thread-safe wrapper around Node.compile() using _MEM_CACHE_LOCK and get_node_lock().
    """
    levels = get_nodes_by_level(root)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for level_idx, row in enumerate(levels):
            # Compile only those nodes not already in the memory/disk cache
            nodes_to_compile = [node for node in row if node._compiled_fst is None]
            if not nodes_to_compile:
                continue
                
            # Submit level's unique nodes concurrently
            futures = [executor.submit(node.compile) for node in nodes_to_compile]
            
            # Wait for all nodes in the row to finish compiling
            concurrent.futures.wait(futures)
            
            # Validate futures for any exceptions raised during compilation
            for future in futures:
                if future.exception() is not None:
                    raise future.exception()
                    
    return root.compile()


# =====================================================================
# PROTOTYPE 2: DYNAMIC DAG SCHEDULER (ZERO-BARRIER DEPENDENCY QUEUE)
# =====================================================================

def compile_graph_dynamic_dag(root: GraphNode, max_workers: int = None) -> pynini.Fst:
    """
    Highly-optimal scheduler that compiles nodes as soon as their specific children are compiled.
    Avoids level-by-level synchronization barriers.
    """
    # 1. Collect all unique nodes in the DAG
    unique_nodes = {}
    def collect(node: GraphNode):
        key = node.cache_key
        if key not in unique_nodes:
            unique_nodes[key] = node
            for child in node.children:
                collect(child)
    collect(root)

    # 2. Build dependency counts and parent-notification map
    uncompiled_deps = {} # cache_key -> count of uncompiled children
    parent_map = {}      # cache_key -> set of parent GraphNodes
    
    for key, node in unique_nodes.items():
        parent_map[key] = set()
        
    for key, node in unique_nodes.items():
        uncompiled_children = [c for c in node.children if c._compiled_fst is None]
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
                # Node.compile() should be modified to be thread-safe
                node.compile()
                on_node_compiled(node)
            except Exception as e:
                logger.error(f"Parallel compilation failed for node {node}: {e}")
                raise e

        # Seed the thread pool with all nodes that have 0 initial uncompiled dependencies
        with lock:
            initial_nodes = [
                node for key, node in unique_nodes.items() 
                if uncompiled_deps[key] == 0 and node._compiled_fst is None
            ]
            
            for node in initial_nodes:
                executor.submit(compile_task, node)

    return root.compile()
