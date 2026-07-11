import os
import hashlib
import json
import time
import pynini
from loguru import logger
from parC.constants import get_yaml_dir
from functools import lru_cache, wraps
from glob import glob
from frozendict import frozendict


CACHE_DIR = os.path.join(get_yaml_dir(), ".cache")
_SYMS_PATH = os.path.join(CACHE_DIR, "symbol_table.syms")

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------
# SHA-256 Dependency-Graph Caching System
# ----------------------------------------------------

def get_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_dir_sha256(directory_path: str) -> str:
    if not os.path.exists(directory_path):
        return ""
    files = []
    for root, _, filenames in os.walk(directory_path):
        for f in filenames:
            if f.endswith((".yaml", ".csv", ".xlsx")):
                files.append(os.path.join(root, f))
    files.sort()
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode("utf-8"))
        h.update(get_file_sha256(f).encode("utf-8"))
    return h.hexdigest()

def compute_cache_key(name: str, kind: str, config_dirs: list[str], child_keys: dict[str, str] = None, description: str = None) -> str:
    if child_keys is None:
        child_keys = {}
    config_hashes = {}
    for d in sorted(config_dirs):
        rel_dir = os.path.relpath(d, get_yaml_dir())
        config_hashes[rel_dir] = get_dir_sha256(d)

    metadata = {
        "name": name,
        "kind": kind,
        "config_dependencies": config_hashes,
        "child_fst_dependencies": {k: child_keys[k] for k in sorted(child_keys.keys())}
    }
    if description is not None:
        metadata["description"] = description
        
    metadata_json = json.dumps(metadata, sort_keys=True)
    cache_key = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
    
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(metadata_json)
            
    return cache_key

_compile_starts = {}

def record_cache_miss(cache_key: str) -> None:
    if cache_key in _compile_starts:
        return
    name, kind, description = "fst", "", None
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            name = metadata.get("name", "fst")
            kind = metadata.get("kind", "")
            description = metadata.get("description")
        except Exception:
            pass
    desc = description if description else (f"{kind} {name}".strip() if kind else name)
    logger.debug(f"Building {desc}...")
    _compile_starts[cache_key] = (time.time(), desc)

def record_cache_save(cache_key: str) -> None:
    if cache_key in _compile_starts:
        start_time, desc = _compile_starts.pop(cache_key)
        duration = time.time() - start_time
        logger.debug(f"Building {desc}... (done {duration:.1f}sec)")

def get_cached_fst(cache_key: str) -> pynini.Fst | list[pynini.Fst] | None:
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if not os.path.exists(meta_path):
        logger.debug(f"FST cache MISS for key {cache_key}: meta file not found")
        record_cache_miss(cache_key)
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        kind = metadata.get("kind", "unknown")
        name = metadata.get("name", "unknown")
        if metadata.get("is_sequence", False):
            fsts = []
            for i in range(metadata.get("sequence_len", 0)):
                path = os.path.join(CACHE_DIR, f"{cache_key}_seq_{i}.fst")
                if not os.path.exists(path):
                    logger.debug(f"FST cache MISS for key {cache_key}: missing sequence FST at index {i}")
                    record_cache_miss(cache_key)
                    return None
                fsts.append(pynini.Fst.read(path))
            logger.debug(f"FST cache HIT for key {cache_key} (Kind: {kind}, Name: {name})")
            return fsts
        else:
            fst_path = os.path.join(CACHE_DIR, f"{cache_key}.fst")
            if os.path.exists(fst_path):
                fst = pynini.Fst.read(fst_path)
                logger.debug(f"FST cache HIT for key {cache_key} (Kind: {kind}, Name: {name})")
                return fst
            else:
                logger.debug(f"FST cache MISS for key {cache_key}: .fst file not found")
                record_cache_miss(cache_key)
    except Exception as e:
        logger.warning(f"Failed to load cached FST for key {cache_key}: {e}")
        record_cache_miss(cache_key)
    return None

def save_cached_fst(cache_key: str, fst: pynini.Fst | list[pynini.Fst]) -> None:
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}

    if isinstance(fst, list):
        metadata["is_sequence"] = True
        metadata["sequence_len"] = len(fst)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True)
        for i, item in enumerate(fst):
            path = os.path.join(CACHE_DIR, f"{cache_key}_seq_{i}.fst")
            item.write(path)
    else:
        metadata["is_sequence"] = False
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True)
        fst_path = os.path.join(CACHE_DIR, f"{cache_key}.fst")
        fst.write(fst_path)
    record_cache_save(cache_key)

def get_cached_pattern_fsts(cache_key: str) -> dict[str, pynini.Fst] | None:
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if not os.path.exists(meta_path):
        logger.debug(f"Pattern FST cache MISS for key {cache_key}: meta file not found")
        record_cache_miss(cache_key)
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        pattern_names = metadata.get("pattern_names", [])
        if not pattern_names:
            logger.debug(f"Pattern FST cache MISS for key {cache_key}: no pattern names in meta")
            record_cache_miss(cache_key)
            return None
        pattern_fsts = {}
        for name in pattern_names:
            safe_name = name.replace("<", "").replace(">", "").replace("/", "_")
            fst_path = os.path.join(CACHE_DIR, f"{cache_key}_pattern_{safe_name}.fst")
            if not os.path.exists(fst_path):
                logger.debug(f"Pattern FST cache MISS for key {cache_key}: missing FST for pattern {name}")
                record_cache_miss(cache_key)
                return None
            pattern_fsts[name] = pynini.Fst.read(fst_path)
        logger.debug(f"Pattern FST cache HIT for key {cache_key} ({len(pattern_names)} patterns)")
        return pattern_fsts
    except Exception as e:
        logger.warning(f"Failed to load cached pattern FSTs for key {cache_key}: {e}")
        record_cache_miss(cache_key)
    return None

def save_cached_pattern_fsts(cache_key: str, pattern_fsts: dict[str, pynini.Fst]) -> None:
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.meta")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}
    metadata["pattern_names"] = list(pattern_fsts.keys())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True)
        
    for name, fst in pattern_fsts.items():
        safe_name = name.replace("<", "").replace(">", "").replace("/", "_")
        fst_path = os.path.join(CACHE_DIR, f"{cache_key}_pattern_{safe_name}.fst")
        try:
            fst.write(fst_path)
        except Exception as e:
            logger.warning(f"Failed to save pattern FST {name} to {fst_path}: {e}")
    record_cache_save(cache_key)

def get_cached_symbol_table(cache_key: str) -> pynini.SymbolTable | None:
    syms_path = os.path.join(CACHE_DIR, f"{cache_key}.syms")
    if os.path.exists(syms_path):
        try:
            syms = pynini.SymbolTable.read(syms_path)
            logger.debug(f"Symbol table cache HIT for key {cache_key}")
            return syms
        except Exception as e:
            logger.warning(f"Failed to load cached symbol table from {syms_path}: {e}")
    logger.debug(f"Symbol table cache MISS for key {cache_key}")
    return None

def save_cached_symbol_table(cache_key: str, syms: pynini.SymbolTable) -> None:
    syms_path = os.path.join(CACHE_DIR, f"{cache_key}.syms")
    try:
        syms.write(syms_path)
    except Exception as e:
        logger.warning(f"Failed to save symbol table to cache path {syms_path}: {e}")

# ----------------------------------------------------
# Compatibility / Legacy Caching Interface
# ----------------------------------------------------

def _fst_path(kind: str, name: str, fst_kind: str) -> str:
    return os.path.join(CACHE_DIR, kind, f"{name}.{fst_kind}.fst")

def _is_valid(path: str, *source_dirs: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    glob_list = []
    for source_dir in source_dirs:
        glob_list.extend(glob(os.path.join(source_dir, "*.yaml")))
        glob_list.extend(glob(os.path.join(source_dir, "*.csv")))
    return all(mtime >= os.path.getmtime(file) for file in glob_list)

def is_syms_cache_valid(*source_dirs: str) -> bool:
    if _is_valid(_SYMS_PATH, *source_dirs):
        return True
    logger.info("Symbol table cache invalidated.")
    return False

def is_fst_cache_valid(kind: str, name: str, fst_kind: str, *source_dirs: str) -> bool:
    if _is_valid(_fst_path(kind, name, fst_kind), *source_dirs):
        return True
    logger.info(f"Fst of kind {kind} {fst_kind} for {name} invalidated.")

def save_symbol_table(syms: pynini.SymbolTable) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    syms.write(_SYMS_PATH)

def load_symbol_table() -> pynini.SymbolTable | None:
    if not os.path.exists(_SYMS_PATH):
        return None
    try:
        return pynini.SymbolTable.read(_SYMS_PATH)
    except Exception:
        logger.warning(f"Failed to load symbol table from {_SYMS_PATH}")
        return None

def save_fst(kind: str, name: str, fst_kind: str, fst: pynini.Fst) -> None:
    path = _fst_path(kind, name, fst_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fst.write(path)

def load_fst(kind: str, name: str, fst_kind: str) -> pynini.Fst | None:
    path = _fst_path(kind, name, fst_kind)
    if not os.path.exists(path):
        return None
    try:
        return pynini.Fst.read(path)
    except Exception:
        logger.warning(f"Failed to load FST from {path}")
        return None

def max_directory_mtime(directory: str):
    yaml_glob = glob(os.path.join(directory, "*.yaml"))
    csv_glob = glob(os.path.join(directory, "*.csv"))
    return max(os.path.getmtime(f) for f in yaml_glob + csv_glob + [directory])

def get_hashable_args_and_kwargs(args, kwargs):
    hashable_args = []
    for arg in args:
        hashable_arg = arg
        if type(arg) is list:
            hashable_arg = tuple(arg)
        elif type(arg) is dict:
            hashable_arg = frozendict(arg)
        try:
            hash(hashable_arg)
        except Exception as e:
            raise ValueError(f"Could not hash kwarg {value} with key {key}: {e}")

        hashable_args.append(hashable_arg)

    hashable_kwargs = {}
    for key, value in kwargs.items():
        hashable_value = value
        if type(value) is list:
            hashable_value = tuple(value)
        elif type(value) is dict:
            hashable_value = frozendict(value)
        try:
            hash(hashable_value)
        except Exception as e:
            raise ValueError(f"Could not hash kwarg {value} with key {key}: {e}")

        hashable_kwargs[key] = hashable_value

    return hashable_args, hashable_kwargs

def observed_cache(directories: list[str]):
    directory_mtimes = {
        directory: max_directory_mtime(directory) for directory in directories
    }

    def decorator(func):
        cached_func = lru_cache(maxsize=128)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            clear_cache = False
            for directory, mtime in directory_mtimes.items():
                new_mtime = max_directory_mtime(directory)
                if new_mtime > mtime:
                    directory_mtimes[directory] = new_mtime
                    clear_cache = True
            if clear_cache:
                logger.info(
                    f"Invalidated cache for {func.__name__}, rebuilding output..."
                )
                cached_func.cache_clear()

            try:
                args, kwargs = get_hashable_args_and_kwargs(args, kwargs)
            except Exception as e:
                logger.exception(
                    f"Error hashing args, building function output without caching. {e}"
                )
                return func(*args, **kwargs)

            return cached_func(*args, **kwargs)

        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        return wrapper

    return decorator
